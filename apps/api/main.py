from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.worker.core.config import load_environment
from routes.image_chunking_jobs import router as image_chunking_jobs_router

try:
    from routes.image_processing import router as image_processing_router
except ImportError:
    image_processing_router = None


load_environment("api")

app = FastAPI(title="AI Course Assistant API")

app.include_router(image_chunking_jobs_router)
if image_processing_router is not None:
    app.include_router(image_processing_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
