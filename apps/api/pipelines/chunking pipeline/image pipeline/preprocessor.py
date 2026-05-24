"""
Advanced OCR Preprocessing Pipeline
====================================
Handles: scanned docs, photographed pages, low-light images,
         skewed/rotated pages, noisy prints, mixed-quality scans,
         stamps, watermarks, tables, handwritten annotations.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

class ImageType(Enum):
    CLEAN_SCAN     = "clean_scan"        # High-quality flatbed scan
    PHOTO          = "photo"             # Phone/camera photo of page
    LOW_LIGHT      = "low_light"         # Poorly lit photograph
    NOISY_PRINT    = "noisy_print"       # Old/degraded printed document
    WATERMARKED    = "watermarked"       # Document with watermarks/stamps
    FORM           = "form"              # Structured form with lines/boxes
    NEWSPAPER      = "newspaper"         # Low-contrast newsprint
    MIXED          = "mixed"             # Cannot be clearly classified
    UNKNOWN        = "unknown"


@dataclass
class DiagnosticReport:
    """Per-image diagnostic collected during classification."""
    detected_type:      ImageType = ImageType.UNKNOWN
    brightness:         float = 0.0      # 0-255 mean
    contrast:           float = 0.0      # std-dev of grayscale
    blur_score:         float = 0.0      # Laplacian variance (higher = sharper)
    skew_angle:         float = 0.0      # Detected rotation in degrees
    has_watermark:      bool  = False
    has_border:         bool  = False
    has_dark_bg:        bool  = False
    noise_level:        str   = "low"   # low / medium / high
    resolution_ok:      bool  = True
    warnings:           list  = field(default_factory=list)


@dataclass
class PipelineResult:
    """Output of the full pipeline."""
    image:              np.ndarray = None   # Final preprocessed image (BGR)
    image_gray:         np.ndarray = None   # Grayscale version
    image_binary:       np.ndarray = None   # Binary (for Tesseract)
    tesseract_config:   str = ""            # Recommended --psm and --oem flags
    diagnostics:        DiagnosticReport = field(default_factory=DiagnosticReport)
    stages_applied:     list = field(default_factory=list)


# ─────────────────────────────────────────────
# Stage 1 — Image loading and validation
# ─────────────────────────────────────────────

class ImageLoader:
    MIN_DIM = 300      # px — smaller than this is too blurry to OCR reliably
    TARGET_DPI = 300   # effective DPI we aim for

    @staticmethod
    def load(source) -> np.ndarray:
        """Load from file path, bytes, or existing numpy array."""
        if isinstance(source, np.ndarray):
            img = source.copy()
        elif isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                raise ValueError(f"Cannot read image: {source}")
        elif isinstance(source, bytes):
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
        return img

    @staticmethod
    def validate_and_upscale(img: np.ndarray, diag: DiagnosticReport) -> tuple[np.ndarray, list]:
        """Ensure minimum resolution; upscale small images for better OCR."""
        stages = []
        h, w = img.shape[:2]

        if h < ImageLoader.MIN_DIM or w < ImageLoader.MIN_DIM:
            diag.warnings.append(f"Image very small ({w}×{h}). OCR quality may be limited.")
            diag.resolution_ok = False

        # Upscale if smaller than 1000px on shortest side (sweet spot for Tesseract)
        short_side = min(h, w)
        if short_side < 1000:
            scale = 1000 / short_side
            # Use INTER_CUBIC for upscaling text — preserves edges better than LINEAR
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
            stages.append(f"upscale_x{scale:.2f}")

        return img, stages


# ─────────────────────────────────────────────
# Stage 2 — Image type classification
# ─────────────────────────────────────────────

class ImageClassifier:
    """
    Heuristic classifier that drives which preprocessing path to take.
    Avoids running expensive operations on already-clean images.
    """

    @staticmethod
    def classify(img: np.ndarray) -> DiagnosticReport:
        diag = DiagnosticReport()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # ── Brightness and contrast
        diag.brightness = float(gray.mean())
        diag.contrast   = float(gray.std())

        # ── Blur score (Laplacian variance — higher = sharper)
        diag.blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # ── Noise level via high-freq energy ratio
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_energy = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
        if noise_energy < 2:
            diag.noise_level = "low"
        elif noise_energy < 6:
            diag.noise_level = "medium"
        else:
            diag.noise_level = "high"

        # ── Dark background detection (inverted documents)
        diag.has_dark_bg = diag.brightness < 80

        # ── Watermark/stamp detection via frequency domain
        dft  = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(np.abs(dft_shift) + 1)
        # Watermarks show as periodic patterns — elevated mid-frequency energy
        cy, cx = h // 2, w // 2
        mid_band = magnitude[cy-h//6:cy+h//6, cx-w//6:cx+w//6]
        high_band = magnitude[cy-h//3:cy+h//3, cx-w//3:cx+w//3]
        if mid_band.mean() > high_band.mean() * 0.82:
            diag.has_watermark = True

        # ── Border detection (thick black frame around scan)
        border_width = max(5, int(min(h, w) * 0.03))
        edges_region = np.concatenate([
            gray[:border_width, :].ravel(),
            gray[-border_width:, :].ravel(),
            gray[:, :border_width].ravel(),
            gray[:, -border_width:].ravel()
        ])
        diag.has_border = float(edges_region.mean()) < 60

        # ── Classification logic
        if diag.brightness < 70 and diag.contrast > 40:
            diag.detected_type = ImageType.LOW_LIGHT
        elif diag.brightness > 200 and diag.contrast < 35 and diag.blur_score > 400:
            diag.detected_type = ImageType.CLEAN_SCAN
        elif diag.noise_level == "high" and diag.contrast < 50:
            diag.detected_type = ImageType.NOISY_PRINT
        elif diag.has_watermark:
            diag.detected_type = ImageType.WATERMARKED
        elif diag.brightness < 130 and diag.blur_score < 80:
            diag.detected_type = ImageType.PHOTO
        elif diag.contrast < 30 and 90 < diag.brightness < 180:
            diag.detected_type = ImageType.NEWSPAPER
        else:
            diag.detected_type = ImageType.MIXED

        logger.debug(f"Classification: {diag.detected_type.value} | "
                     f"brightness={diag.brightness:.1f} contrast={diag.contrast:.1f} "
                     f"blur={diag.blur_score:.1f} noise={diag.noise_level}")
        return diag


# ─────────────────────────────────────────────
# Stage 3 — Color normalisation and channel work
# ─────────────────────────────────────────────

class ColorNormalizer:

    @staticmethod
    def normalize(img: np.ndarray, diag: DiagnosticReport) -> tuple[np.ndarray, list]:
        stages = []

        # ── Invert dark-background images
        if diag.has_dark_bg:
            img = cv2.bitwise_not(img)
            stages.append("invert_dark_bg")

        # ── For photos and low-light: apply CLAHE in LAB color space
        # (operates on Lightness channel only — preserves hue/saturation)
        if diag.detected_type in (ImageType.PHOTO, ImageType.LOW_LIGHT, ImageType.MIXED):
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # clipLimit and tileGrid are tuned for document text (not natural images)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            stages.append("clahe_lab")

        # ── For newspaper / low-contrast: stronger CLAHE on grayscale
        elif diag.detected_type == ImageType.NEWSPAPER:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            stages.append("clahe_gray_newspaper")

        # ── Global gamma correction for dark images
        if diag.brightness < 100:
            gamma = 1.8
            lut = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                            for i in range(256)], dtype=np.uint8)
            img = cv2.LUT(img, lut)
            stages.append(f"gamma_correction_{gamma}")

        return img, stages


# ─────────────────────────────────────────────
# Stage 4 — Noise reduction
# ─────────────────────────────────────────────

class NoiseReducer:

    @staticmethod
    def denoise(img: np.ndarray, diag: DiagnosticReport) -> tuple[np.ndarray, list]:
        stages = []

        if diag.noise_level == "high":
            # Non-local means — best for heavy salt-and-pepper / scan noise
            # h=10 is the filter strength; higher removes more noise but blurs edges
            img = cv2.fastNlMeansDenoisingColored(img, None,
                                                   h=10, hColor=10,
                                                   templateWindowSize=7,
                                                   searchWindowSize=21)
            stages.append("nlm_denoise_strong")

        elif diag.noise_level == "medium":
            # Bilateral filter: smooths noise while PRESERVING edges (critical for text)
            img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            stages.append("bilateral_denoise")

        # For low noise: nothing — avoid blurring clean text

        return img, stages


# ─────────────────────────────────────────────
# Stage 5 — Border and shadow removal
# ─────────────────────────────────────────────

class BorderShadowRemover:

    @staticmethod
    def remove_border(img: np.ndarray, diag: DiagnosticReport) -> tuple[np.ndarray, list]:
        stages = []
        if not diag.has_border:
            return img, stages

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Find the largest bright rectangle inside dark borders
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            # Only crop if the contour is meaningfully smaller than the full image
            if cw < w * 0.98 or ch < h * 0.98:
                img = img[y:y+ch, x:x+cw]
                stages.append("border_crop")

        return img, stages

    @staticmethod
    def remove_shadow(img: np.ndarray, diag: DiagnosticReport) -> tuple[np.ndarray, list]:
        """
        Shadow removal via background normalisation.
        Works by estimating the background illumination and dividing it out.
        """
        stages = []
        if diag.detected_type not in (ImageType.PHOTO, ImageType.LOW_LIGHT):
            return img, stages

        # Dilate to estimate background (shadows are low-frequency)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        bg = cv2.dilate(img, kernel)
        # Divide original by background to normalise illumination
        norm = cv2.divide(img.astype(np.float32),
                          bg.astype(np.float32),
                          scale=255)
        img = np.clip(norm, 0, 255).astype(np.uint8)
        stages.append("shadow_normalisation")
        return img, stages


# ─────────────────────────────────────────────
# Stage 6 — Skew and rotation correction
# ─────────────────────────────────────────────

class SkewCorrector:
    MAX_ANGLE = 45.0  # beyond this it's probably a different orientation, not skew

    @staticmethod
    def detect_and_correct(img: np.ndarray,
                           diag: DiagnosticReport) -> tuple[np.ndarray, list]:
        stages = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Method 1: Hough lines (best for documents with clear text baselines)
        angle_hough = SkewCorrector._hough_angle(gray)

        # ── Method 2: Minimum bounding rectangle of text blobs (fallback)
        angle_mser  = SkewCorrector._mser_angle(gray)

        # ── Consensus: prefer Hough if confident, else average
        if abs(angle_hough) < SkewCorrector.MAX_ANGLE and abs(angle_hough) > 0.1:
            angle = angle_hough
            method = "hough"
        elif abs(angle_mser) < SkewCorrector.MAX_ANGLE and abs(angle_mser) > 0.1:
            angle = angle_mser
            method = "mser"
        else:
            diag.skew_angle = 0.0
            return img, stages

        diag.skew_angle = angle
        if abs(angle) < 0.3:   # Sub-pixel skew — not worth rotating
            return img, stages

        img = SkewCorrector._rotate(img, angle)
        stages.append(f"skew_correction_{method}_{angle:.2f}deg")
        return img, stages

    @staticmethod
    def _hough_angle(gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return 0.0
        angles = []
        for line in lines[:50]:
            theta = line[0][1]
            angle = np.degrees(theta) - 90
            if abs(angle) < 45:
                angles.append(angle)
        return float(np.median(angles)) if angles else 0.0

    @staticmethod
    def _mser_angle(gray: np.ndarray) -> float:
        """Use minimum-area bounding rect of connected text components."""
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        all_points = np.concatenate([c.reshape(-1, 2)
                                     for c in contours if cv2.contourArea(c) > 50])
        if len(all_points) < 5:
            return 0.0
        rect = cv2.minAreaRect(all_points)
        angle = rect[2]
        if angle < -45:
            angle += 90
        return float(angle)

    @staticmethod
    def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        # Expand canvas so corners don't get clipped
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy
        return cv2.warpAffine(img, M, (new_w, new_h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


# ─────────────────────────────────────────────
# Stage 7 — Watermark and stamp suppression
# ─────────────────────────────────────────────

class WatermarkSuppressor:

    @staticmethod
    def suppress(img: np.ndarray, diag: DiagnosticReport) -> tuple[np.ndarray, list]:
        if not diag.has_watermark:
            return img, []

        # Strategy: watermarks are typically light-colored semi-transparent overlays.
        # Convert to HSV and desaturate highly-saturated mid-brightness regions
        # (stamps are often red/blue; watermarks are gray-ish).
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # Mask: mid-saturation, mid-value (typical watermark profile)
        watermark_mask = cv2.inRange(hsv,
                                     np.array([0, 20, 100]),
                                     np.array([180, 150, 220]))
        # Whiten the watermark regions (blend toward white)
        img_float = img.astype(np.float32)
        white = np.full_like(img_float, 255)
        alpha = (watermark_mask.astype(np.float32) / 255.0)[..., np.newaxis]
        img_float = img_float * (1 - alpha * 0.7) + white * (alpha * 0.7)
        return np.clip(img_float, 0, 255).astype(np.uint8), ["watermark_suppress"]


# ─────────────────────────────────────────────
# Stage 8 — Sharpening
# ─────────────────────────────────────────────

class Sharpener:

    @staticmethod
    def sharpen(img: np.ndarray, diag: DiagnosticReport) -> tuple[np.ndarray, list]:
        stages = []

        if diag.blur_score > 500:
            # Image already sharp — light sharpen only to avoid ringing
            kernel = np.array([[0, -0.5, 0],
                                [-0.5, 3, -0.5],
                                [0, -0.5, 0]])
            img = cv2.filter2D(img, -1, kernel)
            stages.append("sharpen_light")

        elif diag.blur_score < 100:
            # Blurry image — unsharp masking (more controlled than a raw Laplacian)
            gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
            img = cv2.addWeighted(img, 1.6, gaussian, -0.6, 0)
            stages.append("sharpen_unsharp_mask")

        else:
            # Moderate sharpening
            kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)
            stages.append("sharpen_standard")

        return img, stages


# ─────────────────────────────────────────────
# Stage 9 — Binarisation (final step before Tesseract)
# ─────────────────────────────────────────────

class Binarizer:
    """
    Multiple binarisation strategies tried; best one selected by
    estimating which produces the cleanest text/background separation.
    """

    @staticmethod
    def binarize(gray: np.ndarray,
                 diag: DiagnosticReport) -> tuple[np.ndarray, str]:

        candidates = {}

        # ── Otsu global threshold (best for bimodal histograms)
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates["otsu"] = otsu

        # ── Adaptive Gaussian (best for uneven illumination / shadows)
        block = Binarizer._adaptive_block_size(gray)
        adaptive_gauss = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block, C=11)
        candidates["adaptive_gauss"] = adaptive_gauss

        # ── Adaptive Mean (more aggressive — good for low-contrast)
        adaptive_mean = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=block, C=8)
        candidates["adaptive_mean"] = adaptive_mean

        # ── Sauvola-inspired (local mean + std threshold — best for degraded docs)
        sauvola = Binarizer._sauvola(gray, window=51, k=0.2, r=128)
        candidates["sauvola"] = sauvola

        # ── Select best by scoring text-pixel ratio and connectivity
        best_name, best_img = Binarizer._select_best(candidates, diag)
        return best_img, best_name

    @staticmethod
    def _adaptive_block_size(gray: np.ndarray) -> int:
        """Block size scales with image resolution for consistent results."""
        h, w = gray.shape
        base = max(11, int(min(h, w) * 0.03))
        return base if base % 2 == 1 else base + 1

    @staticmethod
    def _sauvola(gray: np.ndarray, window: int = 51,
                 k: float = 0.2, r: float = 128) -> np.ndarray:
        """
        Sauvola binarization: threshold = mean * (1 + k * (std/r - 1))
        Particularly strong on degraded historical documents.
        """
        gray_f = gray.astype(np.float32)
        mean = cv2.boxFilter(gray_f, -1, (window, window))
        sq_mean = cv2.boxFilter(gray_f ** 2, -1, (window, window))
        std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
        threshold = mean * (1.0 + k * ((std / r) - 1.0))
        binary = np.where(gray_f >= threshold, 255, 0).astype(np.uint8)
        return binary

    @staticmethod
    def _select_best(candidates: dict,
                     diag: DiagnosticReport) -> tuple[str, np.ndarray]:
        """
        Score each binarized image. Best score = best foreground/background
        separation with connected text components (not scattered noise).
        """
        scores = {}
        for name, img in candidates.items():
            fg_ratio = np.sum(img == 0) / img.size
            # Text documents: black pixels typically 5-25% of area
            ratio_score = 1.0 - abs(fg_ratio - 0.12) * 4
            # Penalise extreme ratios
            if fg_ratio < 0.01 or fg_ratio > 0.5:
                ratio_score = -1.0
            scores[name] = ratio_score

        best = max(scores, key=scores.get)

        # Strategy override for known conditions
        if diag.detected_type == ImageType.NEWSPAPER:
            best = "sauvola"
        elif diag.detected_type == ImageType.CLEAN_SCAN and diag.contrast > 60:
            best = "otsu"
        elif diag.detected_type in (ImageType.PHOTO, ImageType.LOW_LIGHT):
            best = "adaptive_gauss"

        logger.debug(f"Binarization scores: {scores} → selected: {best}")
        return best, candidates[best]


# ─────────────────────────────────────────────
# Stage 10 — Morphological cleanup
# ─────────────────────────────────────────────

class MorphologicalCleaner:

    @staticmethod
    def clean(binary: np.ndarray,
              diag: DiagnosticReport) -> tuple[np.ndarray, list]:
        stages = []

        # ── Remove isolated noise pixels (speckles) via opening
        speckle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, speckle_kernel)
        stages.append("morph_open_despeckle")

        # ── Fill small holes in characters (broken strokes from noise)
        if diag.noise_level in ("medium", "high"):
            fill_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, fill_kernel)
            stages.append("morph_close_fill_gaps")

        # ── Remove very small connected components (stray dots, dust)
        binary = MorphologicalCleaner._remove_small_blobs(binary, min_area=15)
        stages.append("remove_small_blobs")

        return binary, stages

    @staticmethod
    def _remove_small_blobs(binary: np.ndarray, min_area: int = 15) -> np.ndarray:
        inv = cv2.bitwise_not(binary)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inv, connectivity=8)
        cleaned = np.zeros_like(inv)
        for i in range(1, num_labels):  # 0 = background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255
        return cv2.bitwise_not(cleaned)


# ─────────────────────────────────────────────
# Stage 11 — Tesseract config advisor
# ─────────────────────────────────────────────

class TesseractConfigAdvisor:
    """
    Recommends the optimal Tesseract --psm and --oem flags
    based on image characteristics and type.
    """

    @staticmethod
    def recommend(diag: DiagnosticReport) -> str:
        # OEM 3 = LSTM + legacy (best quality; OEM 1 = LSTM only for speed)
        oem = 3

        # PSM (Page Segmentation Mode) selection:
        # 3 = Fully automatic (default)
        # 4 = Assume single column of variable sizes
        # 6 = Assume uniform block of text
        # 11 = Sparse text (find as much as possible)
        # 12 = Sparse text with OSD

        if diag.detected_type == ImageType.FORM:
            psm = 6   # Forms have uniform text blocks in cells
        elif diag.noise_level == "high" or diag.blur_score < 50:
            psm = 11  # Sparse mode is more forgiving on degraded docs
        elif diag.detected_type in (ImageType.CLEAN_SCAN, ImageType.NEWSPAPER):
            psm = 4   # Single column assumption for scanned pages
        else:
            psm = 3   # General automatic segmentation

        config_parts = [f"--oem {oem}", f"--psm {psm}"]

        # Whitelist for numeric-heavy content (optional, can comment out)
        # config_parts.append("-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,-")

        config_parts.append("-c preserve_interword_spaces=1")
        config_parts.append("-c textord_heavy_nr=1")  # Better for noisy docs

        return " ".join(config_parts)


# ─────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────

class OCRPreprocessingPipeline:
    """
    Orchestrates all stages in order, collecting a diagnostic report
    and list of applied stages throughout.

    Usage:
        pipeline = OCRPreprocessingPipeline()
        result = pipeline.run("lecture_scan.jpg")

        # Pass to Tesseract:
        import pytesseract
        text = pytesseract.image_to_string(
            result.image_binary,
            config=result.tesseract_config
        )
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)

    def run(self, source) -> PipelineResult:
        result = PipelineResult()
        all_stages = []

        # ── Stage 1: Load and validate
        img = ImageLoader.load(source)
        img, stages = ImageLoader.validate_and_upscale(img,
                                                        DiagnosticReport())
        all_stages.extend(stages)

        # ── Stage 2: Classify
        diag = ImageClassifier.classify(img)
        result.diagnostics = diag

        if self.debug:
            self._save_debug(img, "s0_original")

        # ── Stage 3: Color normalization
        img, stages = ColorNormalizer.normalize(img, diag)
        all_stages.extend(stages)
        if self.debug: self._save_debug(img, "s3_color_norm")

        # ── Stage 4: Noise reduction
        img, stages = NoiseReducer.denoise(img, diag)
        all_stages.extend(stages)
        if self.debug: self._save_debug(img, "s4_denoised")

        # ── Stage 5a: Border removal
        img, stages = BorderShadowRemover.remove_border(img, diag)
        all_stages.extend(stages)

        # ── Stage 5b: Shadow removal
        img, stages = BorderShadowRemover.remove_shadow(img, diag)
        all_stages.extend(stages)
        if self.debug: self._save_debug(img, "s5_shadow_removed")

        # ── Stage 6: Skew correction
        img, stages = SkewCorrector.detect_and_correct(img, diag)
        all_stages.extend(stages)
        if self.debug: self._save_debug(img, "s6_deskewed")

        # ── Stage 7: Watermark suppression
        img, stages = WatermarkSuppressor.suppress(img, diag)
        all_stages.extend(stages)
        if self.debug: self._save_debug(img, "s7_no_watermark")

        # ── Stage 8: Sharpening
        img, stages = Sharpener.sharpen(img, diag)
        all_stages.extend(stages)
        if self.debug: self._save_debug(img, "s8_sharpened")

        # ── Convert to grayscale for binarization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Stage 9: Binarization
        binary, bin_method = Binarizer.binarize(gray, diag)
        all_stages.append(f"binarize_{bin_method}")
        if self.debug: self._save_debug(binary, "s9_binary")

        # ── Stage 10: Morphological cleanup
        binary, stages = MorphologicalCleaner.clean(binary, diag)
        all_stages.extend(stages)
        if self.debug: self._save_debug(binary, "s10_morph_cleaned")

        # ── Stage 11: Tesseract config
        result.tesseract_config = TesseractConfigAdvisor.recommend(diag)

        # ── Finalize
        result.image        = img
        result.image_gray   = gray
        result.image_binary = binary
        result.stages_applied = all_stages

        logger.info(f"Pipeline complete. Stages: {all_stages}")
        logger.info(f"Tesseract config: {result.tesseract_config}")
        logger.info(f"Type: {diag.detected_type.value} | "
                    f"Skew: {diag.skew_angle:.2f}° | "
                    f"Noise: {diag.noise_level} | "
                    f"Blur: {diag.blur_score:.1f}")

        return result

    def _save_debug(self, img: np.ndarray, name: str):
        path = f"/tmp/ocr_debug_{name}.jpg"
        cv2.imwrite(path, img)
        logger.debug(f"Debug image saved: {path}")
