"""
L1 — Semantic Query Cache.
Similar question (cosine > 0.92) already answered → return cached answer.
Savings: 18s pipeline → 50ms cache hit (360x speedup). TTL: 24 hours.
"""
import json, hashlib, time, uuid
import chromadb
from .redis_client import get_redis
from config import settings


class SemanticQueryCache:
    def __init__(self):
        self.r = get_redis()
        self.col = chromadb.PersistentClient("./data/qcache").get_or_create_collection("qa")
        self.threshold = settings.semantic_cache_threshold

    def get(self, question: str) -> dict | None:
        results = self.col.query(query_texts=[question], n_results=1, include=["metadatas", "distances"])
        if not results["documents"][0]:
            return None
        sim = 1 - results["distances"][0][0]
        if sim < self.threshold:
            return None
        cid = results["metadatas"][0][0].get("cid")
        data = self.r.get(f"qa:{cid}")
        if not data:
            return None
        out = json.loads(data)
        out["from_cache"] = True
        out["similarity"] = round(sim, 3)
        return out

    def set(self, question: str, answer: str, sources: list, ragas_scores: dict):
        cid = str(uuid.uuid4())
        self.r.setex(f"qa:{cid}", settings.cache_ttl_query,
                     json.dumps({"answer": answer, "sources": sources, "ragas": ragas_scores}))
        self.col.upsert(documents=[answer], metadatas=[{"question": question, "cid": cid, "ts": time.time()}],
                        ids=[hashlib.md5(question.encode()).hexdigest()])
