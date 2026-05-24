from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from redis.exceptions import RedisError
from rq.job import Job


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.worker.core.rq import fetch_job, serialize_job  # noqa: E402
from apps.worker.services.document_chunking.run_pipeline import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
)
from apps.worker.services.document_chunking.worker import (  # noqa: E402
    enqueue_job as enqueue_document_chunking_job,
    get_service_queue,
)


class DocumentChunkingJobRequest(BaseModel):
    file_paths: list[str] = Field(
        ...,
        min_length=1,
        description="Absolute or relative paths to PDF/DOC/PPT/TXT/MD files",
    )
    namespace: str | None = Field(None, description="Pinecone namespace override")


router = APIRouter(prefix="/document-chunking/jobs", tags=["document-chunking-jobs"])


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
                detail=(
                    "Unsupported file type. Supported: "
                    + ", ".join(sorted(SUPPORTED_EXTENSIONS))
                ),
            )

        paths.append(path)

    return paths


def enqueue_job(request: DocumentChunkingJobRequest) -> dict:
    paths = validate_submission_paths(request.file_paths)
    try:
        job = enqueue_document_chunking_job(
            file_paths=[str(path) for path in paths],
            namespace=request.namespace,
        )
    except RedisError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Redis/RQ: {exc}",
        ) from exc

    return serialize_job(job)


def get_job_or_404(job_id: str) -> Job:
    try:
        job = fetch_job(job_id)
    except RedisError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Redis/RQ: {exc}",
        ) from exc

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return job


@router.post("")
async def add_document_chunking_job(request: DocumentChunkingJobRequest) -> dict:
    return await run_in_threadpool(enqueue_job, request)


@router.get("")
async def list_document_chunking_jobs(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict:
    def list_jobs() -> dict:
        try:
            queue = get_service_queue()
            job_ids = queue.job_ids[:limit]
            jobs = [Job.fetch(job_id, connection=queue.connection) for job_id in job_ids]
            return {
                "queue": queue.name,
                "count": len(jobs),
                "jobs": [serialize_job(job) for job in jobs],
            }
        except RedisError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Could not connect to Redis/RQ: {exc}",
            ) from exc

    return await run_in_threadpool(list_jobs)


@router.get("/{job_id}")
async def get_document_chunking_job(job_id: str) -> dict:
    job = await run_in_threadpool(get_job_or_404, job_id)
    return serialize_job(job)


@router.get("/{job_id}/result")
async def get_document_chunking_job_result(job_id: str) -> dict:
    job = await run_in_threadpool(get_job_or_404, job_id)
    payload = serialize_job(job, include_result=True)
    if payload["status"] != "finished":
        payload["result"] = None
    return payload


@router.delete("/{job_id}")
async def delete_document_chunking_job(job_id: str) -> dict:
    def delete_job() -> dict:
        job = get_job_or_404(job_id)
        status = job.get_status(refresh=True)
        if status in {"queued", "deferred", "scheduled"}:
            job.cancel()
        job.delete()
        return {"id": job_id, "deleted": True}

    return await run_in_threadpool(delete_job)
