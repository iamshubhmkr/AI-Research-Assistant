"""
Tiny shared helpers.

Design decision: paper/chunk IDs must be STABLE across processes and restarts
(Python's built-in hash() is salted per process, which silently broke Chroma
upserts under multiple workers) — so all IDs derive from sha256.
"""
import hashlib


def stable_id(text: str, length: int = 16) -> str:
    """Deterministic short ID for a string (URL, query, chunk text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
