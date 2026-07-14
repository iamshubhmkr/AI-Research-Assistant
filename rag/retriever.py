"""
Hybrid Retriever — 8-stage pipeline:
multi-query -> HyDE -> dense (Titan) + BM25 -> RRF -> parent expansion
-> optional cross-encoder rerank -> contextual compression (parallel).
Plus vectorless mode (LLM scores chunks directly; zero vector infra).

Design decisions (v3.1):
  - BM25 corpus is filtered to level-0 chunks so sparse retrieval can't
    return RAPTOR summaries the dense path already excludes.
  - RRF dedup keys come from chunk METADATA, not text prefixes — the chunker
    prepends overlap text, so consecutive chunks share their first ~200 chars
    and a doc[:80] key silently merged distinct chunks.
  - The cross-encoder rerank stage is OPTIONAL (config.enable_cross_encoder):
    it is the only local-model dependency left, so default deployments stay
    torch-free and degrade to RRF order.
"""
import json
import re
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from .hyde import hyde_embed
from llm_client import call_llm
from config import settings

logger = logging.getLogger(__name__)

_reranker = None


def _get_reranker():
    """Lazy, optional cross-encoder. Missing dependency -> RRF order."""
    global _reranker
    if not settings.enable_cross_encoder:
        return None
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            logger.warning(f"cross-encoder unavailable ({e}); using RRF order")
            _reranker = False
    return _reranker or None


def rrf_fuse(dense: list, sparse: list, k: int | None = None) -> list[dict]:
    """Reciprocal Rank Fusion over (doc, meta, score) lists, keyed by chunk meta."""
    k = k or settings.rrf_k

    def _key(doc, meta):
        m = meta or {}
        if "paper_id" in m:
            return f"{m.get('paper_id')}|{m.get('section')}|{m.get('chunk_index')}"
        return doc[:80]

    scores = {}
    for ranked in (dense, sparse):
        for rank, (doc, meta, _) in enumerate(ranked):
            e = scores.setdefault(_key(doc, meta), {"s": 0.0, "doc": doc, "meta": meta})
            e["s"] += 1 / (k + rank + 1)
    return sorted(scores.values(), key=lambda x: x["s"], reverse=True)


class HybridRetriever:
    def __init__(self, collection, embedder, cache):
        self.col, self.embedder, self.cache = collection, embedder, cache
        # Only raw L0 chunks belong in the sparse corpus.
        docs = collection.get(where={"level": 0}, include=["documents", "metadatas"])
        self.all_docs = docs["documents"] or []
        self.all_metas = docs["metadatas"] or []
        self.bm25 = BM25Okapi([d.lower().split() for d in self.all_docs]) if self.all_docs else None

    def retrieve(self, query, query_type="specific_fact", top_k=None, final_k=None):
        if not self.all_docs:
            logger.warning("[retriever] empty index — nothing to retrieve")
            return []
        top_k = top_k or settings.retrieval_top_k
        final_k = final_k or settings.retrieval_final_k
        if settings.retrieval_mode == "vectorless":
            return self._vectorless(query, final_k)

        queries = [query] + self._expand(query)
        dense_all, sparse_all = [], []
        per_q = max(top_k // max(len(queries), 1), 3)

        for q in queries:
            vec = hyde_embed(q, self.embedder, self.cache)
            n = min(per_q, len(self.all_docs))
            dense = self.col.query(query_embeddings=[vec], n_results=n, where={"level": 0})
            for doc, meta, dist in zip(dense["documents"][0], dense["metadatas"][0], dense["distances"][0]):
                dense_all.append((doc, meta, 1 - dist))
            if self.bm25:
                scores = self.bm25.get_scores(q.lower().split())
                for i in np.argsort(scores)[::-1][:per_q]:
                    sparse_all.append((self.all_docs[i], self.all_metas[i], float(scores[i])))

        fused = rrf_fuse(dense_all, sparse_all)
        expanded = self._expand_parent(fused[:top_k])

        reranker = _get_reranker()
        if reranker is not None:
            scores = reranker.predict([(query, item["text"]) for item in expanded])
            top = [item for item, _ in sorted(zip(expanded, scores), key=lambda x: x[1], reverse=True)[:final_k]]
        else:
            top = expanded[:final_k]
        return self._compress_parallel(query, top)

    def _vectorless(self, query, final_k):
        """LLM-as-judge retrieval. Good for <1000 docs / no vector infra."""
        scored = []
        for i in range(0, len(self.all_docs), 10):
            batch_d = self.all_docs[i:i + 10]
            batch_m = self.all_metas[i:i + 10]
            docs_txt = "\n---\n".join(f"[DOC {j}]: {d[:300]}" for j, d in enumerate(batch_d))
            r = call_llm(model=settings.claude_haiku, max_tokens=300, temperature=0.0,
                         agent_name="vectorless_retriever",
                         messages=[{"role": "user", "content":
                             f"Query: {query}\n\nDocuments:\n{docs_txt}\n\n"
                             f'Rate relevance 0-10 each. JSON array: [{{"doc":0,"score":8}}]'}])
            try:
                for rating in json.loads(re.search(r"\[.*\]", r["content"][0].text, re.DOTALL).group()):
                    idx = rating.get("doc", 0)
                    if idx < len(batch_d):
                        scored.append({"text": batch_d[idx], "meta": batch_m[idx],
                                       "score": rating.get("score", 0)})
            except Exception:
                scored.extend({"text": d, "meta": m, "score": 0}
                              for d, m in zip(batch_d, batch_m))
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:final_k]

    def _expand_parent(self, items):
        out = []
        for item in items:
            meta = item["meta"] or {}
            pid = f"{meta.get('paper_id','')}_sec_{meta.get('section','')[:20]}"
            parent = self.col.get(ids=[pid], include=["documents"])
            if parent["documents"]:
                out.append({"text": parent["documents"][0], "chunk": item["doc"], "meta": meta})
            else:
                out.append({"text": item["doc"], "meta": meta})
        return out

    def _expand(self, query):
        try:
            r = call_llm(model=settings.claude_haiku, max_tokens=150, temperature=0.3,
                         agent_name="query_expansion",
                         messages=[{"role": "user", "content":
                             f"Generate 2 alternative search queries for: '{query}'.\n"
                             f"Return ONLY the 2 queries, one per line, no numbering or preamble."}])
            lines = [ln.strip(" -•") for ln in r["content"][0].text.strip().split("\n")]
            return [ln for ln in lines if len(ln) > 5][:2]
        except Exception as e:
            logger.warning(f"query expansion failed ({e}); using original query only")
            return []

    def _compress_one(self, args):
        query, chunk = args
        try:
            r = call_llm(model=settings.claude_haiku, max_tokens=200, temperature=0.0,
                         agent_name="compression",
                         messages=[{"role": "user", "content":
                             f"Query: {query}\nText: {chunk['text'][:1500]}\n"
                             f"Return only the 2-3 sentences most relevant. Exact wording."}])
            compressed = r["content"][0].text.strip()
            if compressed:
                chunk["text"] = compressed
        except Exception as e:
            logger.warning(f"compression failed ({e}); keeping full chunk")
        return chunk

    def _compress_parallel(self, query, chunks):
        with ThreadPoolExecutor(max_workers=5) as pool:
            return list(pool.map(self._compress_one, [(query, c) for c in chunks]))


def retriever_node(state: dict) -> dict:
    from rag.vector_store import get_collection
    from rag.embedder import get_embedder
    from cache.embedding_cache import EmbeddingCache
    retriever = HybridRetriever(get_collection(), get_embedder(), EmbeddingCache())
    chunks = retriever.retrieve(state["query"], state.get("query_type", "specific_fact"))
    return {"retrieved_chunks": chunks}
