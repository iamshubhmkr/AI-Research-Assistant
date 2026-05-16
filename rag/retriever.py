"""
Hybrid Retrieval — Dense (HyDE) + Sparse (BM25) + RRF + Cross-encoder rerank.

Pipeline:
  1. Multi-query expansion (3 phrasings via Haiku)
  2. HyDE embedding for each query
  3. Dense retrieval (ChromaDB cosine)
  4. BM25 sparse retrieval
  5. RRF fusion (k=60)
  6. Parent-document expansion
  7. Cross-encoder reranking
  8. Contextual compression (Haiku strips irrelevant sentences)

Vectorless RAG mode:
  Skips steps 2-5, uses LLM directly to score document relevance.
  Useful when: small corpus, no embedding infrastructure, rapid prototyping.
  Trade-off: higher latency per doc, but zero vector DB dependency.

Bottleneck analysis:
  - Cross-encoder reranking: O(query_count * candidate_count) — ~200ms for 20 candidates
  - Multi-query expansion: 1 Haiku call — ~100ms
  - Contextual compression: final_k Haiku calls — ~500ms for 5 chunks
  - Total retrieval: ~1.8s (dominated by compression)
"""
import logging
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from .hyde import hyde_embed
from llm_client import call_llm
from config import settings

logger = logging.getLogger(__name__)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class HybridRetriever:
    def __init__(self, collection, embedder, cache):
        self.col = collection
        self.embedder = embedder
        self.cache = cache
        all_docs = collection.get(include=["documents", "metadatas"])
        self.bm25 = BM25Okapi([d.lower().split() for d in all_docs["documents"]]) if all_docs["documents"] else None
        self.all_docs = all_docs["documents"]
        self.all_metas = all_docs["metadatas"]

    def retrieve(self, query: str, query_type: str = "specific_fact",
                 top_k: int = None, final_k: int = None) -> list:
        top_k = top_k or settings.retrieval_top_k
        final_k = final_k or settings.retrieval_final_k

        if settings.retrieval_mode == "vectorless":
            return self._vectorless_retrieve(query, final_k)

        queries = [query] + self._expand_queries(query)
        dense_all, sparse_all = [], []
        n = max(len(queries), 1)

        for q in queries:
            # Dense retrieval with HyDE
            vec = hyde_embed(q, self.embedder, self.cache)
            dense = self.col.query(query_embeddings=[vec], n_results=top_k // n, where={"level": 0})
            for doc, meta, dist in zip(dense["documents"][0], dense["metadatas"][0], dense["distances"][0]):
                dense_all.append((doc, meta, 1 - dist))

            # Sparse retrieval (BM25)
            if self.bm25:
                scores = self.bm25.get_scores(q.lower().split())
                for i in np.argsort(scores)[::-1][:top_k // n]:
                    sparse_all.append((self.all_docs[i], self.all_metas[i], float(scores[i])))

        fused = self._rrf(dense_all, sparse_all)
        expanded = self._expand_to_parent(fused[:top_k])

        # Cross-encoder reranking
        pairs = [(query, item["text"]) for item in expanded]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(expanded, scores), key=lambda x: x[1], reverse=True)
        top = [item for item, _ in ranked[:final_k]]

        return self._compress(query, top)

    def _vectorless_retrieve(self, query: str, final_k: int) -> list:
        """
        Vectorless RAG: use LLM to score document relevance directly.
        No embeddings, no vector DB queries. LLM reads chunks and scores them.

        Interview talking point:
          "Vectorless RAG eliminates the embedding pipeline entirely.
           The LLM directly scores chunk relevance on a 0-10 scale.
           Trade-off: higher per-query cost but zero infrastructure.
           Good for small corpora or when you can't maintain a vector DB."
        """
        if not self.all_docs:
            return []

        # Score all docs in batches
        batch_size = 10
        scored = []
        for i in range(0, len(self.all_docs), batch_size):
            batch_docs = self.all_docs[i:i+batch_size]
            batch_metas = self.all_metas[i:i+batch_size]
            docs_text = "\n---\n".join([f"[DOC {j}]: {d[:300]}" for j, d in enumerate(batch_docs)])

            result = call_llm(
                model=settings.claude_haiku,
                messages=[{"role": "user", "content":
                    f"Query: {query}\n\nDocuments:\n{docs_text}\n\n"
                    f"Rate each document's relevance to the query (0-10). "
                    f"Return JSON array: [{{\"doc\": 0, \"score\": 8}}, ...]"
                }],
                max_tokens=300, temperature=0.0, agent_name="vectorless_retriever",
            )
            import json, re
            try:
                raw = result["content"][0].text
                ratings = json.loads(re.search(r'\[.*\]', raw, re.DOTALL).group())
                for r in ratings:
                    idx = r.get("doc", 0)
                    if idx < len(batch_docs):
                        scored.append({"text": batch_docs[idx], "meta": batch_metas[idx], "score": r.get("score", 0)})
            except Exception:
                for d, m in zip(batch_docs, batch_metas):
                    scored.append({"text": d, "meta": m, "score": 0})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:final_k]

    def _rrf(self, dense, sparse, k=60):
        scores = {}
        for rank, (doc, meta, _) in enumerate(dense):
            key = doc[:80]
            if key not in scores:
                scores[key] = {"s": 0, "doc": doc, "meta": meta}
            scores[key]["s"] += 1 / (k + rank + 1)
        for rank, (doc, meta, _) in enumerate(sparse):
            key = doc[:80]
            if key not in scores:
                scores[key] = {"s": 0, "doc": doc, "meta": meta}
            scores[key]["s"] += 1 / (k + rank + 1)
        return sorted(scores.values(), key=lambda x: x["s"], reverse=True)

    def _expand_to_parent(self, items):
        expanded = []
        for item in items:
            meta = item["meta"]
            parent_id = f"{meta.get('paper_id', '')}_{meta.get('section', '')[:20]}"
            parent = self.col.get(ids=[parent_id], include=["documents"])
            if parent["documents"]:
                expanded.append({"text": parent["documents"][0], "chunk": item["doc"], "meta": meta})
            else:
                expanded.append({"text": item["doc"], "meta": meta})
        return expanded

    def _expand_queries(self, query: str) -> list:
        result = call_llm(
            model=settings.claude_haiku, max_tokens=150, temperature=0.3,
            messages=[{"role": "user", "content":
                f"Generate 2 alternative search queries for: '{query}'\nVary phrasing. One per line, no numbering."}],
            agent_name="query_expansion",
        )
        return result["content"][0].text.strip().split("\n")[:2]

    def _compress(self, query: str, chunks: list) -> list:
        compressed = []
        for chunk in chunks:
            result = call_llm(
                model=settings.claude_haiku, max_tokens=200, temperature=0.0,
                messages=[{"role": "user", "content":
                    f"Query: {query}\nText: {chunk['text'][:1500]}\n"
                    f"Return only the 2-3 sentences most relevant to the query. Exact wording only."}],
                agent_name="compression",
            )
            chunk["text"] = result["content"][0].text
            compressed.append(chunk)
        return compressed


def retriever_node(state: dict) -> dict:
    from rag.vector_store import get_collection
    from rag.embedder import get_embedder
    from cache.embedding_cache import EmbeddingCache
    collection = get_collection()
    embedder = get_embedder()
    cache = EmbeddingCache()
    retriever = HybridRetriever(collection, embedder, cache)
    chunks = retriever.retrieve(state["query"], state.get("query_type", "specific_fact"))
    return {"retrieved_chunks": chunks}
