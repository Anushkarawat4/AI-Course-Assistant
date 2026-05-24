from __future__ import annotations

import os

from redis import Redis

from apps.worker.core.config import load_environment


DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def get_redis_url() -> str:
    load_environment()
    return os.getenv("REDIS_URL", DEFAULT_REDIS_URL)


def get_redis_connection() -> Redis:
    return Redis.from_url(get_redis_url(), decode_responses=False)
