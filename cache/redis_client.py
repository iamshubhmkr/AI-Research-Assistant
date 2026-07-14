"""Shared Redis connection pool (one pool, all cache layers)."""
import redis
from config import settings

_pool = redis.ConnectionPool.from_url(settings.redis_url, max_connections=20, decode_responses=False)

def get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=_pool)
