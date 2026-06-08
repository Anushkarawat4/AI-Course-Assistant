from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError
from rq.job import Job


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.worker.core.rq import fetch_job, serialize_job  # noqa: E402
from apps.worker.services.qna_chunking.run_pipeline import SUPPORTED_EXTENSIONS  # noqa: E402
from apps.worker.services.qna_chunking.worker import (  # noqa: E402
    enqueue_job as enqueue_qna_chunking_job,
    get_service_queue,
)


class QnAChunkingJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_paths: list[str] = Field(..., min_length=1, description="Paths to QnA PDF/DOCX/PPTX files")
    course_id: str = Field(..., min_length=1, description="Course identifier used to select the Pinecone index")


router = APIRouter(prefix="/qna-chunking/jobs", tags=["qna-chunking-jobs"])


def validate_submission_paths(file_paths: list[str]) -> list[Path]:
    paths = []
    for file_path in file_paths:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
            )
        paths.append(path)
    return paths


def enqueue_job(request: QnAChunkingJobRequest) -> dict:
    paths = validate_submission_paths(request.file_paths)
    try:
        job = enqueue_qna_chunking_job(
            file_paths=[str(path) for path in paths],
            course_id=request.course_id,
        )
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Could not connect to Redis/RQ: {exc}") from exc
    return serialize_job(job)


def get_job_or_404(job_id: str) -> Job:
    try:
        job = fetch_job(job_id)
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Could not connect to Redis/RQ: {exc}") from exc
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.post("")
async def add_qna_chunking_job(request: QnAChunkingJobRequest) -> dict:
    return await run_in_threadpool(enqueue_job, request)


@router.get("")
async def list_qna_chunking_jobs(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict:
    def list_jobs() -> dict:
        try:
            queue = get_service_queue()
            job_ids = queue.job_ids[:limit]
            jobs = [Job.fetch(job_id, connection=queue.connection) for job_id in job_ids]
            return {"queue": queue.name, "count": len(jobs), "jobs": [serialize_job(job) for job in jobs]}
        except RedisError as exc:
            raise HTTPException(status_code=503, detail=f"Could not connect to Redis/RQ: {exc}") from exc
    return await run_in_threadpool(list_jobs)


@router.get("/{job_id}")
async def get_qna_chunking_job(job_id: str) -> dict:
    return serialize_job(await run_in_threadpool(get_job_or_404, job_id))


@router.get("/{job_id}/result")
async def get_qna_chunking_job_result(job_id: str) -> dict:
    job = await run_in_threadpool(get_job_or_404, job_id)
    payload = serialize_job(job, include_result=True)
    if payload["status"] != "finished":
        payload["result"] = None
    return payload


@router.delete("/{job_id}")
async def delete_qna_chunking_job(job_id: str) -> dict:
    def delete_job() -> dict:
        job = get_job_or_404(job_id)
        if job.get_status(refresh=True) in {"queued", "deferred", "scheduled"}:
            job.cancel()
        job.delete()
        return {"id": job_id, "deleted": True}
    return await run_in_threadpool(delete_job)
