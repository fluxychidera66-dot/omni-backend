"""
Cache layer for Omni API results.
Uses Redis if available, falls back to in-memory dict for MVP.
"""
import os
import json
import hashlib
import time
from typing import Optional

# In-memory fallback cache: { key: { "result": ..., "expires_at": float } }
_memory_cache: dict = {}

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))  # 1 hour default

# Try to connect to Redis if configured
_redis_client = None
try:
    import redis
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        print("Cache: Connected to Redis")
    else:
        print("Cache: No REDIS_URL set, using in-memory cache")
except Exception as e:
    print(f"Cache: Redis unavailable ({e}), using in-memory cache")
    _redis_client = None


def make_cache_key(task_type: str, input_text: str, constraints: dict) -> str:
    """Generate a deterministic cache key from task params."""
    raw = f"{task_type}::{input_text}::{json.dumps(constraints, sort_keys=True)}"
    return "omni::" + hashlib.sha256(raw.encode()).hexdigest()


def get_cached(key: str) -> Optional[dict]:
    """Return cached result dict or None."""
    if _redis_client:
        try:
            val = _redis_client.get(key)
            if val:
                return json.loads(val)
        except Exception:
            pass
    else:
        entry = _memory_cache.get(key)
        if entry and entry["expires_at"] > time.time():
            return entry["result"]
        elif entry:
            del _memory_cache[key]
    return None


def set_cached(key: str, result: dict, ttl: int = CACHE_TTL_SECONDS):
    """Store result in cache with TTL."""
    if _redis_client:
        try:
            _redis_client.setex(key, ttl, json.dumps(result))
        except Exception:
            pass
    else:
        _memory_cache[key] = {
            "result": result,
            "expires_at": time.time() + ttl,
        }


def invalidate(key: str):
    """Remove a single key from cache."""
    if _redis_client:
        try:
            _redis_client.delete(key)
        except Exception:
            pass
    else:
        _memory_cache.pop(key, None)
