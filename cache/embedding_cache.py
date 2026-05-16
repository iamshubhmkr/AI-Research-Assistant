"""L2 — Embedding Cache. Same text → same embedding. TTL: 7 days."""
import pickle, hashlib
from .redis_client import get_redis
from config import settings

class EmbeddingCache:
    def __init__(self):
        self.r = get_redis()

    def _key(self, text: str) -> str:
        return f"emb:{hashlib.sha256(text.encode()).hexdigest()}"

    def get(self, text: str) -> list | None:
        data = self.r.get(self._key(text))
        return pickle.loads(data) if data else None

    def set(self, text: str, embedding: list, ttl: int = None):
        self.r.setex(self._key(text), ttl or settings.cache_ttl_embedding, pickle.dumps(embedding))
