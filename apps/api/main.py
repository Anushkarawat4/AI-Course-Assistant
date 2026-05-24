from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.worker.core.config import load_environment
from routes.document_chunking_jobs import router as document_chunking_jobs_router
from routes.image_chunking_jobs import router as image_chunking_jobs_router
from routes.temp_chunk_preview import router as temp_chunk_preview_router

try:
    from routes.image_processing import router as image_processing_router
except ImportError:
    image_processing_router = None


load_environment("api")

app = FastAPI(title="AI Course Assistant API")

app.include_router(document_chunking_jobs_router)
app.include_router(image_chunking_jobs_router)
# TEMPORARY: delete this include and the import above when chunk preview is removed.
app.include_router(temp_chunk_preview_router)
if image_processing_router is not None:
    app.include_router(image_processing_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
