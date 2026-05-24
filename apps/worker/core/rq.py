from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from apps.worker.core.redis import get_redis_connection


@dataclass(frozen=True)
class QueueSettings:
    name: str
    timeout_seconds: int = 1800
    result_ttl_seconds: int = 86400
    failure_ttl_seconds: int = 604800


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def get_queue(settings: QueueSettings) -> Queue:
    return Queue(
        name=settings.name,
        connection=get_redis_connection(),
        default_timeout=settings.timeout_seconds,
    )


def fetch_job(job_id: str) -> Job | None:
    try:
        return Job.fetch(job_id, connection=get_redis_connection())
    except NoSuchJobError:
        return None


def datetime_to_iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def serialize_job(job: Job, include_result: bool = False) -> dict:
    status = job.get_status(refresh=True)
    payload = {
        "id": job.id,
        "status": status.value if hasattr(status, "value") else str(status),
        "queue_name": job.origin,
        "created_at": datetime_to_iso(job.created_at),
        "enqueued_at": datetime_to_iso(job.enqueued_at),
        "started_at": datetime_to_iso(job.started_at),
        "ended_at": datetime_to_iso(job.ended_at),
        "timeout": job.timeout,
        "meta": job.meta,
        "exc_info": job.exc_info,
    }

    if include_result:
        payload["result"] = job.result

    return payload
