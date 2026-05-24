from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
IMAGE_PIPELINE_DIR = (
    PROJECT_ROOT
    / "apps"
    / "api"
    / "pipelines"
    / "chunking pipeline"
    / "image pipeline"
)

for import_path in (PROJECT_ROOT, IMAGE_PIPELINE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def load_ocr_worker_class() -> type:
    ocr_worker_path = IMAGE_PIPELINE_DIR / "ocr_worker.py"
    spec = importlib.util.spec_from_file_location("ocr_worker", ocr_worker_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OCR worker from {ocr_worker_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ocr_worker", module)
    spec.loader.exec_module(module)
    return module.OCRWorker


OCRWorker = load_ocr_worker_class()


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


def validate_file_path(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )

    return path


def process_image_chunking_request(
    file_path: str,
    dpi: int = 300,
) -> dict[str, Any]:
    path = validate_file_path(file_path)
    worker = OCRWorker(debug=False)
    pages = worker.process_file(path, dpi=dpi)
    response = worker.to_response(pages)
    response["source"] = {
        "file_path": str(path),
        "file_type": path.suffix.lower().lstrip("."),
        "dpi": dpi,
    }
    return response
