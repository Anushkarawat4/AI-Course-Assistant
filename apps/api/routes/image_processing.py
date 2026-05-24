# from __future__ import annotations

# import sys
# from functools import lru_cache
# from pathlib import Path
# from typing import Annotated

# from fastapi import APIRouter, HTTPException, Query
# from fastapi.concurrency import run_in_threadpool
# from pydantic import BaseModel, Field


# IMAGE_PIPELINE_DIR = (
#     Path(__file__).resolve().parents[1]
#     / "pipelines"
#     / "chunking pipeline"
#     / "image pipeline"
# )
# if str(IMAGE_PIPELINE_DIR) not in sys.path:
#     sys.path.insert(0, str(IMAGE_PIPELINE_DIR))

# from ocr_worker import OCRWorker  # noqa: E402


# SUPPORTED_EXTENSIONS = {
#     ".pdf",
#     ".png",
#     ".jpg",
#     ".jpeg",
#     ".tif",
#     ".tiff",
#     ".bmp",
#     ".webp",
# }


# class OCRPathRequest(BaseModel):
#     file_path: str = Field(..., description="Absolute or relative path to a PDF/image")
#     dpi: int = Field(300, ge=72, le=600, description="PDF render DPI")


# router = APIRouter(prefix="/image-processing", tags=["image-processing"])


# @lru_cache(maxsize=1)
# def get_ocr_worker() -> OCRWorker:
#     return OCRWorker(debug=False)


# def validate_file_path(file_path: str) -> Path:
#     path = Path(file_path).expanduser()
#     if not path.is_absolute():
#         path = (Path.cwd() / path).resolve()

#     if not path.exists():
#         raise HTTPException(status_code=404, detail=f"File not found: {path}")
#     if not path.is_file():
#         raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")
#     if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Unsupported file type. Supported: "
#                 + ", ".join(sorted(SUPPORTED_EXTENSIONS))
#             ),
#         )

#     return path


# def process_ocr_path(file_path: str, dpi: int) -> dict:
#     path = validate_file_path(file_path)
#     worker = get_ocr_worker()
#     pages = worker.process_file(path, dpi=dpi)
#     response = worker.to_response(pages)
#     response["source"] = {
#         "file_path": str(path),
#         "file_type": path.suffix.lower().lstrip("."),
#         "dpi": dpi,
#     }
#     return response


# @router.post("/ocr")
# async def process_image_ocr(request: OCRPathRequest) -> dict:
#     return await run_in_threadpool(process_ocr_path, request.file_path, request.dpi)


# @router.get("/ocr")
# async def process_image_ocr_from_query(
#     file_path: Annotated[str, Query(description="Absolute or relative path to a PDF/image")],
#     dpi: Annotated[int, Query(ge=72, le=600)] = 300,
# ) -> dict:
#     return await run_in_threadpool(process_ocr_path, file_path, dpi)
