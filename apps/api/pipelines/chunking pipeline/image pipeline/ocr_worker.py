"""
FastAPI worker integration
===========================
Drop this into your background ingestion worker.
Handles: single images, multi-page scanned PDFs (each page as an image),
         and returns structured OCR results with per-chunk metadata.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator

try:
    import cv2
    import numpy as np
    IMAGE_LIBS_AVAILABLE = True
    IMAGE_LIBS_IMPORT_ERROR = None
except ImportError as exc:
    cv2 = None
    np = None
    IMAGE_LIBS_AVAILABLE = False
    IMAGE_LIBS_IMPORT_ERROR = exc

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    from tesseract_engine import TesseractOCREngine
    OCR_ENGINE_AVAILABLE = True
    OCR_ENGINE_IMPORT_ERROR = None
except ImportError as exc:
    TesseractOCREngine = None
    OCR_ENGINE_AVAILABLE = False
    OCR_ENGINE_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


@dataclass
class OCRBlock:
    """A single detected text block with spatial metadata."""
    text:        str
    page:        int
    confidence:  float          # 0-100 from Tesseract
    left:        int
    top:         int
    width:       int
    height:      int
    block_num:   int
    block_type:  str = "unknown"
    language:    str = "unknown"
    is_code:     bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageOCRResult:
    """Full OCR result for one page/image."""
    page_number:    int
    raw_text:       str
    blocks:         list[OCRBlock] = field(default_factory=list)
    mean_confidence: float = 0.0
    skew_angle:     float  = 0.0
    image_type:     str    = ""
    stages_applied: list   = field(default_factory=list)
    warnings:       list   = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "text": self.raw_text,
            "metadata": {
                "mean_confidence": self.mean_confidence,
                "skew_angle": self.skew_angle,
                "image_type": self.image_type,
                "stages_applied": self.stages_applied,
                "warnings": self.warnings,
                "block_count": len(self.blocks),
            },
            "blocks": [block.to_dict() for block in self.blocks],
        }


class OCRWorker:
    """
    Main entry point for the ingestion pipeline.
    Accepts image bytes or a scanned PDF and returns structured OCR results.
    """

    MIN_CONFIDENCE = 40    # Discard Tesseract blocks below this confidence
    MIN_BLOCK_LEN  = 2     # Discard single-character noise blocks

    def __init__(self, debug: bool = False):
        if not IMAGE_LIBS_AVAILABLE:
            raise RuntimeError(
                "Image processing dependencies are not installed. "
                "Install apps/api/requirements.txt first."
            ) from IMAGE_LIBS_IMPORT_ERROR

        if not OCR_ENGINE_AVAILABLE:
            raise RuntimeError(
                "Tesseract OCR engine dependencies are not installed. "
                "Install apps/api/requirements.txt and ensure Tesseract is "
                "available on PATH."
            ) from OCR_ENGINE_IMPORT_ERROR

        assert TesseractOCREngine is not None
        self.engine = TesseractOCREngine(debug=debug)

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def process_image(self, image_bytes: bytes,
                      page_number: int = 1) -> PageOCRResult:
        """Process a single image (JPEG, PNG, TIFF, BMP, WEBP)."""
        result = self.engine.run(image_bytes, page=page_number)
        return self._to_page_result(result)

    def process_file(self, file_path: str | Path,
                     dpi: int = 300) -> list[PageOCRResult]:
        """Process a local image or PDF file and return page-wise OCR results."""
        path = Path(file_path)
        content = path.read_bytes()

        if path.suffix.lower() == ".pdf":
            return self.process_scanned_pdf(content, dpi=dpi)

        return [self.process_image(content, page_number=1)]

    def process_scanned_pdf(self,
                            pdf_bytes: bytes,
                            dpi: int = 300) -> list[PageOCRResult]:
        """
        Process a scanned PDF by rendering each page as an image
        and running the full pipeline on each page.
        """
        if not FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF processing. "
                               "pip install pymupdf")

        results = []
        for page_num, page_img in self._pdf_to_images(pdf_bytes, dpi):
            logger.info(f"Processing page {page_num}")
            result = self.engine.run(page_img, page=page_num)
            results.append(self._to_page_result(result))

        return results

    def get_full_text(self, pages: list[PageOCRResult]) -> str:
        """Concatenate all page texts with page-break markers."""
        parts = []
        for page in pages:
            if page.raw_text.strip():
                parts.append(f"[Page {page.page_number}]\n{page.raw_text}")
        return "\n\n".join(parts)

    def to_response(self, pages: list[PageOCRResult]) -> dict:
        """Build a JSON-ready response with page-wise OCR text and metadata."""
        return {
            "page_count": len(pages),
            "pages": [page.to_dict() for page in pages],
            "full_text": self.get_full_text(pages),
        }

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _to_page_result(self, result: Any) -> PageOCRResult:
        """Adapt the OCR engine result to the worker response shape."""
        blocks = [
            OCRBlock(
                text=block.text,
                page=block.page,
                confidence=block.confidence,
                left=block.bbox[0],
                top=block.bbox[1],
                width=block.bbox[2],
                height=block.bbox[3],
                block_num=block.block_id,
                block_type=block.block_type.value,
                language=block.language.value if block.language else "unknown",
                is_code=block.is_code,
            )
            for block in result.blocks
            if block.text.strip()
        ]
        return PageOCRResult(
            page_number    = result.page,
            raw_text       = result.full_text,
            blocks         = blocks,
            mean_confidence= result.mean_confidence,
            skew_angle     = result.skew_angle,
            image_type     = result.image_type,
            stages_applied = result.stages_applied,
            warnings       = result.warnings,
        )

    def _pdf_to_images(self, pdf_bytes: bytes,
                       dpi: int) -> Generator[tuple[int, Any], None, None]:
        """Render each page of a PDF as a numpy image."""
        assert cv2 is not None
        assert np is not None
        assert fitz is not None

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mat = fitz.Matrix(dpi / 72, dpi / 72)   # 72 dpi is PyMuPDF default
        for page_num, page in enumerate(doc, start=1):
            pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_bytes = pix.tobytes("png")
            arr  = np.frombuffer(img_bytes, dtype=np.uint8)
            img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            yield page_num, img
        doc.close()


# ─────────────────────────────────────────────────────
# FastAPI route example (drop into your ingestion router)
# ─────────────────────────────────────────────────────
"""
from fastapi import UploadFile, HTTPException
from ocr_worker import OCRWorker

worker = OCRWorker()

@router.post("/ingest/scanned-image")
async def ingest_scanned_image(file: UploadFile):
    content = await file.read()
    if file.content_type == "application/pdf":
        pages = worker.process_scanned_pdf(content, dpi=300)
    else:
        pages = [worker.process_image(content, page_number=1)]

    # Warn if OCR quality is low
    low_conf_pages = [p.page_number for p in pages if p.mean_confidence < 60]
    if low_conf_pages:
        print(f"Warning: Low OCR confidence on pages {low_conf_pages}")

    full_text = worker.get_full_text(pages)
    # → pass full_text into your chunking pipeline
    return {"pages": len(pages), "text_length": len(full_text)}
"""
