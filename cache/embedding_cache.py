"""L2 — embedding cache: same text -> same vector. 7 days, SHA256 keys.
Stored as JSON, not pickle — pickle.loads from a shared Redis is an RCE
vector if the cache is ever compromised."""
import json
import hashlib
from .redis_client import get_redis
from config import settings


class EmbeddingCache:
    def __init__(self):
        self.r = get_redis()

    def _key(self, text):
        return f"emb:{hashlib.sha256(text.encode()).hexdigest()}"

    def get(self, text):
        data = self.r.get(self._key(text))
        return json.loads(data) if data else None

    def set(self, text, embedding, ttl=None):
        self.r.setex(self._key(text), ttl or settings.cache_ttl_embedding,
                     json.dumps(list(embedding)))
