"""L1 — semantic query cache: similar question (cosine>0.92) -> cached answer. 24h.
v3.1: questions are embedded with Titan (through the L2 cache) so Chroma
never instantiates a local default embedding model; cache failures degrade
to a miss instead of failing the request."""
import json
import hashlib
import time
import uuid
import logging
import chromadb
from .redis_client import get_redis
from .embedding_cache import EmbeddingCache
from config import settings

logger = logging.getLogger(__name__)


class SemanticQueryCache:
    def __init__(self):
        self.r = get_redis()
        self.col = chromadb.PersistentClient("./data/qcache").get_or_create_collection("qa")
        self.emb_cache = EmbeddingCache()
        self.threshold = settings.semantic_cache_threshold

    def _embed(self, question):
        from rag.embedder import embed_texts
        return embed_texts([question], self.emb_cache)[0]

    def get(self, question):
        try:
            if self.col.count() == 0:
                return None
            res = self.col.query(query_embeddings=[self._embed(question)], n_results=1,
                                 include=["metadatas", "distances"])
            if not res["ids"][0]:
                return None
            sim = 1 - res["distances"][0][0]
            if sim < self.threshold:
                return None
            data = self.r.get(f"qa:{res['metadatas'][0][0].get('cid')}")
            if not data:
                return None
            out = json.loads(data)
            out.update({"from_cache": True, "similarity": round(sim, 3)})
            return out
        except Exception as e:
            logger.warning(f"semantic cache get failed ({e}); treating as miss")
            return None

    def set(self, question, answer, sources, ragas_scores):
        try:
            cid = str(uuid.uuid4())
            self.r.setex(f"qa:{cid}", settings.cache_ttl_query,
                         json.dumps({"answer": answer, "sources": sources, "ragas": ragas_scores}))
            self.col.upsert(documents=[question],
                            embeddings=[self._embed(question)],
                            metadatas=[{"cid": cid, "ts": time.time()}],
                            ids=[hashlib.md5(question.encode()).hexdigest()])
        except Exception as e:
            logger.warning(f"semantic cache set failed ({e}); answer not cached")
