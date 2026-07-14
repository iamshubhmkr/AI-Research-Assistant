"""L3 — LLM response cache. ONLY for temperature=0 (deterministic) calls. 1h."""
import json
import hashlib
from .redis_client import get_redis
from config import settings


class LLMCache:
    def __init__(self):
        self.r = get_redis()

    def _key(self, model, msgs, system):
        s = json.dumps({"m": model, "msgs": msgs, "sys": system}, sort_keys=True)
        return f"llm:{hashlib.sha256(s.encode()).hexdigest()}"

    def get(self, model, msgs, system=""):
        data = self.r.get(self._key(model, msgs, system))
        return json.loads(data) if data else None

    def set(self, model, msgs, response, system=""):
        self.r.setex(self._key(model, msgs, system), settings.cache_ttl_llm, json.dumps(response))
