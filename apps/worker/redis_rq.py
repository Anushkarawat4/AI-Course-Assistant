from __future__ import annotations

from apps.worker.core.redis import get_redis_connection, get_redis_url
from apps.worker.core.rq import fetch_job, serialize_job
from apps.worker.services.image_chunking.worker import (
    QUEUE_NAME as DEFAULT_IMAGE_CHUNKING_QUEUE,
    enqueue_job as enqueue_image_chunking_job,
    get_service_queue as get_image_chunking_queue,
)


def get_image_chunking_queue_name() -> str:
    return DEFAULT_IMAGE_CHUNKING_QUEUE
