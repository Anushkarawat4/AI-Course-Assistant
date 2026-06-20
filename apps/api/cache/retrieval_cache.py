"""
retrieval_cache.py
==================
Redis-backed semantic cache for the retrieval pipeline.

How it works
------------
On every retrieve() call, BEFORE hitting Pinecone:

  1. Compute a SHA-256 hash of normalised(query + mode).
     → Instant exact-match lookup with no API call.

  2. If no exact match, embed the query with Gemini (RETRIEVAL_QUERY task,
     ~150-300 ms) and compute cosine similarity against every stored entry
     for (course_id, mode) in Redis.

  3. Cache HIT  (similarity >= CACHE_HIT_THRESHOLD, default 0.90):
     Deserialise the cached RetrievedChunks + QueryPlan and return a
     RetrievalResult immediately — no Pinecone, no Gemini planner call.

  4. Cache MISS (similarity < threshold):
     Run the full pipeline. After retrieval, embed the original query and
     store the result (chunks + plan + sub-queries + embedding) in Redis
     with a configurable TTL (default 24 h).

Redis key layout
----------------
  rc:{course_id}:{mode}:idx           List[entry_id]    ← index of all entries
  rc:{course_id}:{mode}:{entry_id}    JSON blob         ← one cache entry

Each entry JSON:
{
  "entry_id":         str,
  "query":            str,
  "query_hash":       str,       # sha256 of normalised query+mode
  "course_id":        str,
  "mode":             str,
  "top_k":            int,
  "query_emb":        [float x 1536],
  "sub_queries":      [str, ...],   # planner sub-query texts
  "plan":             {...},        # QueryPlan.model_dump()
  "chunks":           [{...}, ...], # RetrievedChunk serialised
  "source_breakdown": {...},
  "total_candidates": int,
  "latency_ms":       float,
  "created_at":       str,         # ISO-8601
}

Environment variables
---------------------
  RETRIEVAL_CACHE_ENABLED         "true" / "false"  (default: true)
  RETRIEVAL_CACHE_TTL_SECONDS     int               (default: 86400 = 24 h)
  RETRIEVAL_CACHE_HIT_THRESHOLD   float 0-1         (default: 0.90)
  RETRIEVAL_CACHE_MAX_ENTRIES     int per bucket     (default: 200)
  REDIS_URL                       redis://...        (default: redis://localhost:6379/0)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config (read once at module load, override-able via env)
# ─────────────────────────────────────────────────────────────────────────────

def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("1", "true", "yes")

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

CACHE_ENABLED       : bool  = _env_bool ("RETRIEVAL_CACHE_ENABLED",       True)
CACHE_TTL           : int   = _env_int  ("RETRIEVAL_CACHE_TTL_SECONDS",    86_400)
CACHE_HIT_THRESHOLD : float = _env_float("RETRIEVAL_CACHE_HIT_THRESHOLD",  0.90)
CACHE_MAX_ENTRIES   : int   = _env_int  ("RETRIEVAL_CACHE_MAX_ENTRIES",    200)

_KEY_PREFIX = "rc"   # short prefix so Redis keys stay readable


# ─────────────────────────────────────────────────────────────────────────────
# Redis connection (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_redis_client: Optional["Redis"] = None


def _get_redis() -> "Redis":
    global _redis_client
    if _redis_client is None:
        from redis import Redis as RedisClient
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = RedisClient.from_url(url, decode_responses=True)
    return _redis_client


# ─────────────────────────────────────────────────────────────────────────────
# Maths helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))

def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


# ─────────────────────────────────────────────────────────────────────────────
# Query normalisation + hashing (for instant exact-match without embedding)
# ─────────────────────────────────────────────────────────────────────────────

def _query_hash(query: str, mode: str) -> str:
    normalised = " ".join(query.lower().split()) + "|" + mode.lower()
    return hashlib.sha256(normalised.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Gemini embedding (RETRIEVAL_QUERY — matches ingestion RETRIEVAL_DOCUMENT)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_query(text: str) -> list[float]:
    from google import genai
    from google.genai import types as gtypes

    api_key = os.getenv("GEMINI_API_KEY", "")
    model   = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    dim     = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
    client  = genai.Client(api_key=api_key.strip())

    result = client.models.embed_content(
        model=model,
        contents=text.strip() or " ",
        config=gtypes.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=dim,
        ),
    )
    return list(result.embeddings[0].values)


# ─────────────────────────────────────────────────────────────────────────────
# Redis key helpers
# ─────────────────────────────────────────────────────────────────────────────

def _idx_key(course_id: str, mode: str) -> str:
    return f"{_KEY_PREFIX}:{course_id}:{mode}:idx"

def _entry_key(course_id: str, mode: str, entry_id: str) -> str:
    return f"{_KEY_PREFIX}:{course_id}:{mode}:{entry_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def lookup(
    query:     str,
    course_id: str,
    mode:      str,   # "ask" | "quiz" | "summarize"
    top_k:     int,
) -> Optional[dict]:
    """
    Search the Redis cache for a semantically similar previous query.

    Returns the cached entry dict (with "chunks", "plan", "sub_queries",
    "source_breakdown", "total_candidates", "latency_ms") if a hit is found,
    or None on a miss.

    Hit conditions
    --------------
    • Exact hash match  → instant, no embedding call
    • Cosine similarity >= CACHE_HIT_THRESHOLD (default 0.90)

    The returned dict is ALWAYS a cache hit and can be used directly to
    reconstruct a RetrievalResult.
    """
    if not CACHE_ENABLED:
        return None

    try:
        rc = _get_redis()
        idx_key = _idx_key(course_id, mode)

        entry_ids: list[str] = rc.lrange(idx_key, 0, -1)  # type: ignore[arg-type]
        if not entry_ids:
            return None

        q_hash = _query_hash(query, mode)

        # ── Pass 1: exact hash match (no embedding) ───────────────────────
        for eid in entry_ids:
            raw = rc.get(_entry_key(course_id, mode, eid))
            if not raw:
                continue
            entry: dict = json.loads(raw)
            if entry.get("query_hash") == q_hash:
                logger.info(
                    "Cache HIT (exact)  course=%s mode=%s  entry=%s",
                    course_id, mode, eid,
                )
                entry["cache_hit"]  = True
                entry["hit_type"]   = "exact"
                entry["similarity"] = 1.0
                return entry

        # ── Pass 2: semantic similarity (embed query) ─────────────────────
        t0 = time.perf_counter()
        q_emb = _embed_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        best_sim  : float = 0.0
        best_entry: Optional[dict] = None

        for eid in entry_ids:
            raw = rc.get(_entry_key(course_id, mode, eid))
            if not raw:
                continue
            entry = json.loads(raw)
            stored_emb = entry.get("query_emb", [])
            if not stored_emb:
                continue
            sim = _cosine(q_emb, stored_emb)
            if sim > best_sim:
                best_sim   = sim
                best_entry = entry

        if best_entry and best_sim >= CACHE_HIT_THRESHOLD:
            logger.info(
                "Cache HIT (semantic)  course=%s mode=%s  similarity=%.4f  embed_ms=%.0f",
                course_id, mode, best_sim, embed_ms,
            )
            best_entry["cache_hit"]  = True
            best_entry["hit_type"]   = "semantic"
            best_entry["similarity"] = round(best_sim, 4)
            return best_entry

        logger.info(
            "Cache MISS  course=%s mode=%s  best_sim=%.4f  embed_ms=%.0f",
            course_id, mode, best_sim, embed_ms,
        )
        return None

    except Exception as exc:
        # Never let cache errors break the retrieval pipeline
        logger.warning("Cache lookup error (skipped): %s", exc)
        return None


def store(
    query:            str,
    course_id:        str,
    mode:             str,
    top_k:            int,
    chunks:           list,   # list of RetrievedChunk
    plan:             object, # QueryPlan (Pydantic model)
    source_breakdown: dict,
    total_candidates: int,
    latency_ms:       float,
) -> None:
    """
    Embed the query and store the retrieval result in Redis.

    Called AFTER a successful full retrieval so future similar queries
    can be served from cache.

    Parameters
    ----------
    chunks : list[RetrievedChunk]
        Real retrieved chunks. Each must have a .to_dict() method.
    plan : QueryPlan
        The Pydantic plan object (.model_dump() is called to serialise it).
    """
    if not CACHE_ENABLED:
        return

    try:
        rc = _get_redis()

        # Embed original query (RETRIEVAL_QUERY task)
        q_emb   = _embed_query(query)
        q_hash  = _query_hash(query, mode)
        entry_id = str(uuid.uuid4())

        # Serialise chunks
        chunk_dicts: list[dict] = []
        for c in chunks:
            if hasattr(c, "to_dict"):
                chunk_dicts.append(c.to_dict())
            elif hasattr(c, "__dict__"):
                chunk_dicts.append(vars(c))
            else:
                chunk_dicts.append(str(c))

        # Serialise plan
        plan_dict: dict = {}
        if hasattr(plan, "model_dump"):
            plan_dict = plan.model_dump()
        elif hasattr(plan, "dict"):
            plan_dict = plan.dict()
        elif hasattr(plan, "__dict__"):
            plan_dict = vars(plan)

        sub_queries = [sq.text for sq in getattr(plan, "sub_queries", [])]

        entry = {
            "entry_id":         entry_id,
            "query":            query,
            "query_hash":       q_hash,
            "course_id":        course_id,
            "mode":             mode,
            "top_k":            top_k,
            "query_emb":        q_emb,
            "sub_queries":      sub_queries,
            "plan":             plan_dict,
            "chunks":           chunk_dicts,
            "source_breakdown": source_breakdown,
            "total_candidates": total_candidates,
            "latency_ms":       latency_ms,
            "created_at":       datetime.now(timezone.utc).isoformat(),
        }

        idx_key = _idx_key(course_id, mode)
        ekey    = _entry_key(course_id, mode, entry_id)

        rc.set(ekey, json.dumps(entry), ex=CACHE_TTL)

        # Add to index, enforce max entries (LRU-style eviction)
        rc.lpush(idx_key, entry_id)
        rc.expire(idx_key, CACHE_TTL)

        current_len = rc.llen(idx_key)
        if current_len > CACHE_MAX_ENTRIES:
            evicted_id = rc.rpop(idx_key)
            if evicted_id:
                rc.delete(_entry_key(course_id, mode, str(evicted_id)))
                logger.debug("Evicted oldest cache entry: %s", evicted_id)

        logger.info(
            "Cache STORE  course=%s mode=%s  entry=%s  chunks=%d  sub_queries=%d",
            course_id, mode, entry_id, len(chunk_dicts), len(sub_queries),
        )

    except Exception as exc:
        # Never let cache errors break the retrieval pipeline
        logger.warning("Cache store error (skipped): %s", exc)


def invalidate_course(course_id: str) -> int:
    """
    Delete ALL cached entries for a course (all modes).

    Useful when new content is ingested for a course so stale cached
    results are flushed. Returns number of keys deleted.
    """
    if not CACHE_ENABLED:
        return 0
    try:
        rc = _get_redis()
        pattern = f"{_KEY_PREFIX}:{course_id}:*"
        keys = rc.keys(pattern)
        if keys:
            rc.delete(*keys)
        logger.info("Cache invalidated  course=%s  deleted=%d keys", course_id, len(keys))
        return len(keys)
    except Exception as exc:
        logger.warning("Cache invalidation error: %s", exc)
        return 0


def cache_stats(course_id: str) -> dict:
    """
    Return cache statistics for a given course.
    """
    if not CACHE_ENABLED:
        return {"enabled": False}
    try:
        rc = _get_redis()
        stats: dict = {"enabled": True, "course_id": course_id, "modes": {}}
        for mode in ("ask", "quiz", "summarize"):
            idx_key = _idx_key(course_id, mode)
            count   = rc.llen(idx_key)
            stats["modes"][mode] = {"entries": count}
        return stats
    except Exception as exc:
        logger.warning("Cache stats error: %s", exc)
        return {"enabled": True, "error": str(exc)}
