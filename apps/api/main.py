from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.worker.core.config import load_environment

load_environment("api")

from routes.agentic_retrieval import router as agentic_retrieval_router
from routes.audio_chunking_jobs import router as audio_chunking_jobs_router
from routes.document_chunking_jobs import router as document_chunking_jobs_router
from routes.handwritten_chunking_jobs import router as handwritten_chunking_jobs_router
from routes.image_chunking_jobs import router as image_chunking_jobs_router
from routes.qna_chunking_jobs import router as qna_chunking_jobs_router
from routes.temp_chunk_preview import router as temp_chunk_preview_router
from routes.youtube_chunking_jobs import router as youtube_chunking_jobs_router

try:
    from routes.image_processing import router as image_processing_router
except ImportError:
    image_processing_router = None

app = FastAPI(title="AI Course Assistant API")

app.include_router(agentic_retrieval_router)
app.include_router(audio_chunking_jobs_router)
app.include_router(document_chunking_jobs_router)
app.include_router(handwritten_chunking_jobs_router)
app.include_router(image_chunking_jobs_router)
app.include_router(qna_chunking_jobs_router)
app.include_router(youtube_chunking_jobs_router)
# TEMPORARY: delete this include and the import above when chunk preview is removed.
app.include_router(temp_chunk_preview_router)
if image_processing_router is not None:
    app.include_router(image_processing_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
