"""
services/redis.py — Redis client, query-result cache, and retrieval cache.

Cache layers
------------
1. Query cache   key: "query::<sha256(normalised_query)>"
                 value: JSON-serialised PipelineResponse.to_dict()
                 TTL: QUERY_CACHE_TTL (default 1 hour)

2. Retrieval cache  key: "retrieval::<sha256(query_text + params)>"
                    value: JSON-serialised list[RetrievalResult.to_dict()]
                    TTL: RETRIEVAL_CACHE_TTL (default 30 minutes)

Both caches degrade gracefully: if Redis is unavailable, every call is a
cache miss and the application continues without caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)

# ── TTLs (seconds) ────────────────────────────────────────────────────────────
QUERY_CACHE_TTL: int     = int(os.getenv("QUERY_CACHE_TTL",     "3600"))   # 1 hour
RETRIEVAL_CACHE_TTL: int = int(os.getenv("RETRIEVAL_CACHE_TTL", "1800"))   # 30 min

# ── Key prefixes ──────────────────────────────────────────────────────────────
_QUERY_PREFIX     = "query::"
_RETRIEVAL_PREFIX = "retrieval::"

# ── Singleton client ──────────────────────────────────────────────────────────
_client = None


def _get_client():
    """Return a shared Redis client, or None if Redis is unavailable."""
    global _client
    if _client is not None:
        return _client

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        import redis as redis_lib
        client = redis_lib.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()          # fail fast if Redis is down
        _client = client
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — caching disabled.", exc)
        _client = None

    return _client


def is_available() -> bool:
    """Return True if Redis is reachable."""
    return _get_client() is not None


def _make_key(prefix: str, *parts: str) -> str:
    """Build a namespaced cache key using a SHA-256 hash of the parts."""
    combined = "|".join(parts)
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}"


def _normalise_query(query: str) -> str:
    """Lower-case and collapse whitespace so minor variations share a cache entry."""
    return " ".join(query.lower().split())


# ── Query cache ───────────────────────────────────────────────────────────────

def get_query_cache(query: str) -> Optional[dict]:
    """
    Return the cached PipelineResponse dict for *query*, or None on a miss.
    The returned dict is identical to what PipelineResponse.to_dict() produces.
    """
    client = _get_client()
    if client is None:
        return None
    key = _make_key(_QUERY_PREFIX, _normalise_query(query))
    try:
        raw = client.get(key)
        if raw:
            logger.debug("Query cache HIT: %s", key)
            return json.loads(raw)
    except Exception as exc:
        logger.warning("Redis get_query_cache error: %s", exc)
    return None


def set_query_cache(query: str, response_dict: dict, ttl: int = QUERY_CACHE_TTL) -> None:
    """
    Store *response_dict* (from PipelineResponse.to_dict()) keyed by *query*.
    Silently no-ops if Redis is unavailable.
    """
    client = _get_client()
    if client is None:
        return
    key = _make_key(_QUERY_PREFIX, _normalise_query(query))
    try:
        client.setex(key, ttl, json.dumps(response_dict))
        logger.debug("Query cache SET: %s (ttl=%ds)", key, ttl)
    except Exception as exc:
        logger.warning("Redis set_query_cache error: %s", exc)


def invalidate_query_cache(query: str) -> bool:
    """Delete the cached entry for a specific query. Returns True if deleted."""
    client = _get_client()
    if client is None:
        return False
    key = _make_key(_QUERY_PREFIX, _normalise_query(query))
    try:
        return bool(client.delete(key))
    except Exception as exc:
        logger.warning("Redis invalidate_query_cache error: %s", exc)
        return False


def flush_all_query_cache() -> int:
    """
    Delete ALL query-cache entries (keys matching 'query::*').
    Returns the number of keys deleted.
    Called after index rebuilds so stale answers are evicted.
    """
    client = _get_client()
    if client is None:
        return 0
    try:
        keys = client.keys(f"{_QUERY_PREFIX}*")
        if keys:
            return client.delete(*keys)
    except Exception as exc:
        logger.warning("Redis flush_all_query_cache error: %s", exc)
    return 0


# ── Retrieval cache ───────────────────────────────────────────────────────────

def get_retrieval_cache(query_text: str, final_top_k: int, domain_filter: Optional[str]) -> Optional[list]:
    """
    Return cached retrieval results (list of dicts) for the given parameters,
    or None on a miss.
    """
    client = _get_client()
    if client is None:
        return None
    key = _make_key(
        _RETRIEVAL_PREFIX,
        _normalise_query(query_text),
        str(final_top_k),
        domain_filter or "",
    )
    try:
        raw = client.get(key)
        if raw:
            logger.debug("Retrieval cache HIT: %s", key)
            return json.loads(raw)
    except Exception as exc:
        logger.warning("Redis get_retrieval_cache error: %s", exc)
    return None


def set_retrieval_cache(
    query_text: str,
    final_top_k: int,
    domain_filter: Optional[str],
    results: list,
    ttl: int = RETRIEVAL_CACHE_TTL,
) -> None:
    """
    Store retrieval results (list of RetrievalResult.to_dict()) in the cache.
    Silently no-ops if Redis is unavailable.
    """
    client = _get_client()
    if client is None:
        return
    key = _make_key(
        _RETRIEVAL_PREFIX,
        _normalise_query(query_text),
        str(final_top_k),
        domain_filter or "",
    )
    try:
        client.setex(key, ttl, json.dumps(results))
        logger.debug("Retrieval cache SET: %s (ttl=%ds)", key, ttl)
    except Exception as exc:
        logger.warning("Redis set_retrieval_cache error: %s", exc)


def flush_all_retrieval_cache() -> int:
    """
    Delete ALL retrieval-cache entries.
    Called after index rebuilds to prevent stale vector results.
    """
    client = _get_client()
    if client is None:
        return 0
    try:
        keys = client.keys(f"{_RETRIEVAL_PREFIX}*")
        if keys:
            return client.delete(*keys)
    except Exception as exc:
        logger.warning("Redis flush_all_retrieval_cache error: %s", exc)
    return 0


def flush_all_caches() -> dict:
    """
    Flush both query and retrieval caches.
    Returns a summary dict with keys deleted per cache type.
    """
    q = flush_all_query_cache()
    r = flush_all_retrieval_cache()
    logger.info("Cache flush: %d query keys, %d retrieval keys deleted.", q, r)
    return {"query_keys_deleted": q, "retrieval_keys_deleted": r}


# ── Job-status store (used by RabbitMQ worker) ────────────────────────────────
# Stores a JSON blob for each async ingestion job keyed by job_id.
# TTL: 24 hours — long enough to poll status after processing completes.

_JOB_PREFIX = "job::"
JOB_TTL: int = int(os.getenv("JOB_STATUS_TTL", "86400"))   # 24 hours


def set_job_status(job_id: str, status: dict, ttl: int = JOB_TTL) -> None:
    """Persist the status payload for *job_id*."""
    client = _get_client()
    if client is None:
        return
    key = f"{_JOB_PREFIX}{job_id}"
    try:
        client.setex(key, ttl, json.dumps(status))
    except Exception as exc:
        logger.warning("Redis set_job_status error: %s", exc)


def get_job_status(job_id: str) -> Optional[dict]:
    """Return the status dict for *job_id*, or None if not found."""
    client = _get_client()
    if client is None:
        return None
    key = f"{_JOB_PREFIX}{job_id}"
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.warning("Redis get_job_status error: %s", exc)
        return None
