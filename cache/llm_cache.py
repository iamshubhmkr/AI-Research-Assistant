"""
L3 — LLM Response Cache. Same prompt hash → same response.
ONLY for temperature=0 calls (deterministic). TTL: 1 hour.
"""
import json, hashlib
from .redis_client import get_redis
from config import settings

class LLMCache:
    def __init__(self):
        self.r = get_redis()

    def _key(self, model: str, msgs: list, system: str) -> str:
        s = json.dumps({"m": model, "msgs": msgs, "sys": system}, sort_keys=True)
        return f"llm:{hashlib.sha256(s.encode()).hexdigest()}"

    def get(self, model: str, msgs: list, system: str = "") -> str | None:
        data = self.r.get(self._key(model, msgs, system))
        return json.loads(data) if data else None

    def set(self, model: str, msgs: list, response: str, system: str = ""):
        self.r.setex(self._key(model, msgs, system), settings.cache_ttl_llm, json.dumps(response))
