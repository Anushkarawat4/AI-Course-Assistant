from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field


# TEMPORARY CHUNK PREVIEW ROUTES.
# Delete this entire file when chunk inspection is no longer needed.
# Also delete the temp_chunk_preview_router import/include from apps/api/main.py.

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCUMENT_PIPELINE_DIR = (
    PROJECT_ROOT
    / "apps"
    / "api"
    / "pipelines"
    / "chunking pipeline"
    / "document pipeline"
)
DOCUMENT_PIPELINE_FILE = DOCUMENT_PIPELINE_DIR / "ingestion_pipeline_pinecone.py"
IMAGE_PIPELINE_DIR = (
    PROJECT_ROOT
    / "apps"
    / "api"
    / "pipelines"
    / "chunking pipeline"
    / "image pipeline"
)
IMAGE_OCR_WORKER_FILE = IMAGE_PIPELINE_DIR / "ocr_worker.py"

DOCUMENT_EXTENSIONS = {".pdf", ".pptx", ".ppt", ".docx", ".doc", ".txt", ".md"}
IMAGE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


router = APIRouter(prefix="/temp/chunks", tags=["temporary-chunk-preview"])


class DocumentChunkPreviewRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative document path")
    include_metadata: bool = Field(True, description="Return chunk metadata")


class ImageChunkPreviewRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative image/PDF path")
    dpi: int = Field(300, ge=72, le=600, description="PDF render DPI")
    include_pages: bool = Field(True, description="Return full page OCR response")


def resolve_file_path(file_path: str, supported_extensions: set[str]) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")
    if path.suffix.lower() not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Supported: "
                + ", ".join(sorted(supported_extensions))
            ),
        )

    return path


def load_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise ImportError(f"Module file not found: {module_path}")

    if str(module_path.parent) not in sys.path:
        sys.path.insert(0, str(module_path.parent))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def get_document_pipeline():
    return load_module("temp_document_ingestion_pipeline", DOCUMENT_PIPELINE_FILE)


@lru_cache(maxsize=1)
def get_ocr_worker_class():
    module = load_module("temp_ocr_worker", IMAGE_OCR_WORKER_FILE)
    return module.OCRWorker


def preview_document_chunks(request: DocumentChunkPreviewRequest) -> dict[str, Any]:
    path = resolve_file_path(request.file_path, DOCUMENT_EXTENSIONS)
    pipeline = get_document_pipeline()

    elements = pipeline.partition_document(str(path))
    chunks = pipeline.create_chunks(elements)
    documents = pipeline.build_documents(chunks, str(path))

    preview_chunks = []
    for index, document in enumerate(documents):
        item = {
            "chunk_index": index,
            "text": document.page_content,
            "char_count": len(document.page_content),
        }
        if request.include_metadata:
            item["metadata"] = document.metadata
        preview_chunks.append(item)

    return {
        "pipeline": "document",
        "source": {
            "file_path": str(path),
            "file_type": path.suffix.lower().lstrip("."),
        },
        "chunk_count": len(preview_chunks),
        "chunks": preview_chunks,
    }


def preview_image_chunks(request: ImageChunkPreviewRequest) -> dict[str, Any]:
    path = resolve_file_path(request.file_path, IMAGE_EXTENSIONS)
    OCRWorker = get_ocr_worker_class()

    worker = OCRWorker(debug=False)
    pages = worker.process_file(path, dpi=request.dpi)
    response = worker.to_response(pages)

    preview_chunks = []
    for page in pages:
        for block_index, block in enumerate(page.blocks):
            preview_chunks.append(
                {
                    "chunk_index": len(preview_chunks),
                    "page_number": page.page_number,
                    "block_index": block_index,
                    "text": block.text,
                    "char_count": len(block.text),
                    "metadata": {
                        "confidence": block.confidence,
                        "bbox": {
                            "left": block.left,
                            "top": block.top,
                            "width": block.width,
                            "height": block.height,
                        },
                        "block_num": block.block_num,
                        "block_type": block.block_type,
                        "language": block.language,
                        "is_code": block.is_code,
                        "page_mean_confidence": page.mean_confidence,
                        "page_skew_angle": page.skew_angle,
                        "page_image_type": page.image_type,
                    },
                }
            )

    result = {
        "pipeline": "image",
        "source": {
            "file_path": str(path),
            "file_type": path.suffix.lower().lstrip("."),
            "dpi": request.dpi,
        },
        "page_count": response["page_count"],
        "chunk_count": len(preview_chunks),
        "chunks": preview_chunks,
        "full_text": response["full_text"],
    }

    if request.include_pages:
        result["pages"] = response["pages"]

    return result


@router.post("/document")
async def get_document_chunks(request: DocumentChunkPreviewRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(preview_document_chunks, request)
    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline import failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document chunk preview failed: {exc}") from exc


@router.post("/image")
async def get_image_chunks(request: ImageChunkPreviewRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(preview_image_chunks, request)
    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline import failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image chunk preview failed: {exc}") from exc
