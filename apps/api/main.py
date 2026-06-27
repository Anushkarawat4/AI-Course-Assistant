from __future__ import annotations


from fastapi.middleware.cors import CORSMiddleware
import sys
from app.routers import auth
from app.routers.courses import router as courses_router
from app.routers.materials import router as materials_router
from app.routers.chat import router as chat_router
from app.routers.quiz import router as quiz_router


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
from routes.evals import router as evals_router
from routes.retrieval_cache import router as retrieval_cache_router

try:
    from routes.image_processing import router as image_processing_router
except ImportError:
    image_processing_router = None


app = FastAPI(title="CourseGPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(courses_router)
app.include_router(materials_router)
app.include_router(chat_router)
app.include_router(quiz_router)
print("AUTH ROUTER REGISTERED")
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


# LOCAL EVAL — shown in OpenAPI docs for easy Swagger testing.
app.include_router(evals_router)

# RETRIEVAL CACHE management — stats, invalidation, health check.
app.include_router(retrieval_cache_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "message": "CourseGPT API is running",
        "docs": "/docs",
        "health": "/health"
    }