"""
routes/retrieval_cache.py
=========================
Cache management endpoints for the retrieval pipeline's Redis semantic cache.

All endpoints are under /internal/cache and visible in Swagger.

Endpoints
---------
  GET   /internal/cache/stats/{course_id}      Per-course cache stats
  DELETE /internal/cache/{course_id}           Invalidate all entries for a course
  GET   /internal/cache/health                 Redis connectivity check
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/cache", tags=["Retrieval Cache"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_cache():
    """Import cache module (lazy so Redis errors don't crash startup)."""
    try:
        from cache import retrieval_cache
        return retrieval_cache
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cache module unavailable: {exc}",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", summary="Redis connectivity check")
def cache_health() -> dict[str, Any]:
    """
    Check whether Redis is reachable and the cache module is loaded.

    Returns connection status, cache config (TTL, threshold, max entries),
    and whether caching is enabled via RETRIEVAL_CACHE_ENABLED env var.
    """
    try:
        rc = _get_cache()
        redis_conn = rc._get_redis()
        pong = redis_conn.ping()
        return {
            "redis_ok":          pong,
            "cache_enabled":     rc.CACHE_ENABLED,
            "ttl_seconds":       rc.CACHE_TTL,
            "hit_threshold":     rc.CACHE_HIT_THRESHOLD,
            "max_entries":       rc.CACHE_MAX_ENTRIES,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Cache health check failed: %s", exc)
        return {
            "redis_ok":      False,
            "cache_enabled": False,
            "error":         str(exc),
        }


@router.get("/stats/{course_id}", summary="Cache statistics for a course")
def course_cache_stats(course_id: str) -> dict[str, Any]:
    """
    Return the number of cached entries per mode for a given course.

    Example response:
    ```json
    {
      "course_id": "CS301",
      "modes": {
        "ask":       {"entries": 12},
        "quiz":      {"entries": 3},
        "summarize": {"entries": 5}
      }
    }
    ```

    Each entry represents one previously retrieved query result stored
    with its query embedding for semantic similarity matching.
    """
    rc = _get_cache()
    try:
        return rc.cache_stats(course_id)
    except Exception as exc:
        logger.error("Cache stats error for course_id=%s: %s", course_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/{course_id}", summary="Invalidate all cache entries for a course")
def invalidate_course_cache(course_id: str) -> dict[str, Any]:
    """
    Delete ALL Redis cache entries for the given course (all modes).

    **Use this when:**
    - New content has been ingested for a course (workers finished)
    - You want to force fresh Pinecone retrieval for all subsequent queries
    - Cache entries are stale due to pipeline changes

    Returns the number of Redis keys deleted.

    Example response:
    ```json
    {
      "course_id": "CS301",
      "keys_deleted": 47,
      "status": "ok"
    }
    ```
    """
    rc = _get_cache()
    try:
        deleted = rc.invalidate_course(course_id)
        logger.info("Cache invalidated for course_id=%s  deleted=%d keys", course_id, deleted)
        return {
            "course_id":    course_id,
            "keys_deleted": deleted,
            "status":       "ok",
        }
    except Exception as exc:
        logger.error("Cache invalidation error for course_id=%s: %s", course_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
