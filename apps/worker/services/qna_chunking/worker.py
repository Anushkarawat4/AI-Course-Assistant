from __future__ import annotations

import os
import sys

from redis import Redis
from rq import Queue, SimpleWorker
from rq.job import Job
from rq.timeouts import TimerDeathPenalty

from apps.worker.core.redis import get_redis_url
from apps.worker.core.rq import QueueSettings, env_int, get_queue


QUEUE_NAME = os.getenv("QNA_CHUNKING_QUEUE", "qna-chunking")
JOB_FUNCTION = (
    "apps.worker.services.qna_chunking.jobs."
    "process_qna_chunking_request"
)


def get_queue_settings() -> QueueSettings:
    return QueueSettings(
        name=QUEUE_NAME,
        timeout_seconds=env_int("QNA_CHUNKING_JOB_TIMEOUT", 3600),
        result_ttl_seconds=env_int("QNA_CHUNKING_RESULT_TTL", 86400),
        failure_ttl_seconds=env_int("QNA_CHUNKING_FAILURE_TTL", 604800),
    )


def get_service_queue() -> Queue:
    return get_queue(get_queue_settings())


def enqueue_job(
    file_paths: list[str],
    course_id: str,
) -> Job:
    settings = get_queue_settings()
    queue = get_service_queue()
    kwargs = {
        "file_paths": file_paths,
        "course_id": course_id,
    }
    return queue.enqueue(
        JOB_FUNCTION,
        kwargs=kwargs,
        job_timeout=settings.timeout_seconds,
        result_ttl=settings.result_ttl_seconds,
        failure_ttl=settings.failure_ttl_seconds,
        meta={**kwargs, "queue": queue.name},
    )


def run_worker() -> None:
    connection = Redis.from_url(get_redis_url(), decode_responses=False)
    worker = SimpleWorker([QUEUE_NAME], connection=connection)
    if sys.platform.startswith("win"):
        worker.death_penalty_class = TimerDeathPenalty
    worker.work()
