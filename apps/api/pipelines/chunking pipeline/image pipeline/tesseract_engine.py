"""
Tesseract OCR Engine
=====================
Plugs directly onto the preprocessing pipeline (preprocessor.py).
Handles:
  - Typed academic text (paragraphs, headings, lists)
  - Source code (Python, Java, C++, JS, SQL, pseudocode, etc.)
  - Mixed content (text + code blocks on the same page)
  - Tables, formulas, numbered lists
  - Multi-column layouts

Returns structured output with per-block metadata ready to pass
into the chunking pipeline.
"""

import re
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import the preprocessing pipeline
from preprocessor import OCRPreprocessingPipeline, PipelineResult, ImageType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Enums and data classes
# ─────────────────────────────────────────────

class BlockType(Enum):
    HEADING      = "heading"
    PARAGRAPH    = "paragraph"
    CODE         = "code"
    INLINE_CODE  = "inline_code"
    TABLE_CELL   = "table_cell"
    LIST_ITEM    = "list_item"
    FORMULA      = "formula"
    CAPTION      = "caption"
    UNKNOWN      = "unknown"


class CodeLanguage(Enum):
    PYTHON      = "python"
    JAVA        = "java"
    CPP         = "cpp"
    C           = "c"
    JAVASCRIPT  = "javascript"
    TYPESCRIPT  = "typescript"
    SQL         = "sql"
    BASH        = "bash"
    PSEUDOCODE  = "pseudocode"
    UNKNOWN     = "unknown"


@dataclass
class TextBlock:
    """A single logical block of OCR'd content."""
    block_id:     int
    block_type:   BlockType
    text:         str
    raw_text:     str                   # Before post-processing
    confidence:   float                 # 0-100
    page:         int
    bbox:         tuple                 # (x, y, w, h) in pixels
    language:     Optional[CodeLanguage] = None   # Set if block_type == CODE
    is_code:      bool = False
    line_count:   int = 0
    warnings:     list = field(default_factory=list)


@dataclass
class OCRResult:
    """Final structured result for one image/page."""
    page:             int
    blocks:           list[TextBlock]
    full_text:        str           # Clean plain text (all blocks concatenated)
    code_blocks:      list[TextBlock]  # Subset: only CODE type
    mean_confidence:  float
    image_type:       str
    skew_corrected:   bool
    skew_angle:       float
    tesseract_config: str
    stages_applied:   list
    warnings:         list = field(default_factory=list)

    def has_code(self) -> bool:
        return len(self.code_blocks) > 0

    def get_text_only(self) -> str:
        return "\n\n".join(
            b.text for b in self.blocks
            if b.block_type not in (BlockType.CODE, BlockType.TABLE_CELL)
            and b.text.strip()
        )

    def get_code_only(self) -> list[dict]:
        return [
            {"language": b.language.value if b.language else "unknown",
             "code": b.text,
             "confidence": b.confidence,
             "page": b.page}
            for b in self.code_blocks
        ]


# ─────────────────────────────────────────────
# Tesseract configuration builder
# ─────────────────────────────────────────────

class TesseractConfig:
    """
    Builds optimal Tesseract config strings for different content types.
    All configs target maximum accuracy over speed since this runs as
    a background worker (not a real-time API path).
    """

    # Base: LSTM engine (OEM 1) is faster; OEM 3 (LSTM + legacy) is more accurate
    # For a university RAG system accuracy matters more than speed
    OEM = 3

    @staticmethod
    def for_general_text(multicolumn: bool = False) -> str:
        """Standard typed academic text."""
        psm = 4 if multicolumn else 6
        return (
            f"--oem {TesseractConfig.OEM} "
            f"--psm {psm} "
            "-c preserve_interword_spaces=1 "
            "-c textord_heavy_nr=1 "
            "-c tessedit_do_invert=0"
        )

    @staticmethod
    def for_code() -> str:
        """
        Source code requires:
        - PSM 6 (uniform block of text) — code files are one uniform block
        - Disable word dictionary entirely (code is not English)
        - Preserve every space and indentation (indentation is semantically meaningful)
        - Allow all printable ASCII characters (no quote chars in whitelist to
          avoid shell-quoting issues inside pytesseract's subprocess call)
        """
        # Note: We intentionally omit a tessedit_char_whitelist here.
        # Whitelisting in pytesseract passes the value through a shell command,
        # and embedded quotes/backslashes break the invocation.
        # Instead we rely on OEM 3 + disabled dictionaries for code accuracy.
        return (
            f"--oem {TesseractConfig.OEM} "
            "--psm 6 "
            "-c preserve_interword_spaces=1 "
            "-c load_system_dawg=0 "    # Disable English dictionary (kills code)
            "-c load_freq_dawg=0 "      # Disable frequency-based word guessing
            "-c segment_penalty_dict_nonword=0 "
            "-c textord_heavy_nr=1"
        )

    @staticmethod
    def for_sparse_text() -> str:
        """Degraded or low-density text pages."""
        return (
            f"--oem {TesseractConfig.OEM} "
            "--psm 11 "
            "-c preserve_interword_spaces=1 "
            "-c textord_heavy_nr=1"
        )

    @staticmethod
    def for_single_line() -> str:
        """Single line (captions, headings)."""
        return f"--oem {TesseractConfig.OEM} --psm 7"

    @staticmethod
    def for_table() -> str:
        """Table cells — treat each cell as a single uniform block."""
        return (
            f"--oem {TesseractConfig.OEM} "
            "--psm 6 "
            "-c preserve_interword_spaces=1"
        )


# ─────────────────────────────────────────────
# Code language detector
# ─────────────────────────────────────────────

class CodeLanguageDetector:
    """
    Heuristic language detection on raw OCR'd text.
    Works without any ML model — purely pattern-based.
    Fast enough to run on every block.
    """

    PATTERNS = {
        CodeLanguage.PYTHON: [
            r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+[\s:(]',
            r'\bimport\s+\w+', r'\bfrom\s+\w+\s+import',
            r'\bif\s+__name__\s*==',
            r':\s*$',                    # Python block endings with colon
            r'\bprint\s*\(',
            r'\bself\.',
            r'->\s*\w+:',               # Type hints
            r'#.*$',                     # Python comments
        ],
        CodeLanguage.JAVA: [
            r'\bpublic\s+(static\s+)?(class|void|int|String)',
            r'\bprivate\s+\w+\s+\w+',
            r'\bSystem\.out\.print',
            r'@Override', r'@Autowired',
            r'\bnew\s+\w+\s*\(',
            r'import\s+java\.',
            r';\s*$',                    # Java semicolons
        ],
        CodeLanguage.CPP: [
            r'#include\s*[<"]',
            r'\bstd::', r'\bcout\s*<<', r'\bcin\s*>>',
            r'\busing\s+namespace\s+std',
            r'\bvoid\s+\w+\s*\(',
            r'->\w+',                    # Pointer access
            r'::\w+',                    # Scope resolution
        ],
        CodeLanguage.C: [
            r'#include\s*[<"]',
            r'\bprintf\s*\(', r'\bscanf\s*\(',
            r'\bmalloc\s*\(', r'\bfree\s*\(',
            r'\bint\s+main\s*\(',
            r'\*\w+',                    # Pointer declarations
        ],
        CodeLanguage.JAVASCRIPT: [
            r'\bconst\s+\w+\s*=',
            r'\blet\s+\w+\s*=',
            r'\bvar\s+\w+\s*=',
            r'\bfunction\s+\w+\s*\(',
            r'=>\s*{',                   # Arrow functions
            r'\bconsole\.log\s*\(',
            r'===', r'!==',              # Strict equality
            r'\bpromise\b', r'\basync\b', r'\bawait\b',
            r'document\.', r'window\.',
        ],
        CodeLanguage.SQL: [
            r'\bSELECT\b.*\bFROM\b',
            r'\bINSERT\s+INTO\b',
            r'\bUPDATE\b.*\bSET\b',
            r'\bDELETE\s+FROM\b',
            r'\bCREATE\s+(TABLE|DATABASE|INDEX)\b',
            r'\bWHERE\b', r'\bJOIN\b', r'\bGROUP\s+BY\b',
        ],
        CodeLanguage.BASH: [
            r'^\s*#!.*/(bash|sh|zsh)',
            r'\becho\s+', r'\bexport\s+',
            r'\$\w+',                    # Variable references
            r'\|\s*\w+',                 # Pipes
            r'chmod\s+', r'sudo\s+',
            r'&&\s*\n', r'\\\s*\n',      # Line continuation
        ],
        CodeLanguage.PSEUDOCODE: [
            r'\bBEGIN\b', r'\bEND\b',
            r'\bIF\b.*\bTHEN\b', r'\bELSE\b',
            r'\bFOR\b.*\bDO\b', r'\bWHILE\b.*\bDO\b',
            r'\bRETURN\b', r'\bPRINT\b',
            r'←',                       # Assignment arrow in pseudocode
            r'\bPROCEDURE\b', r'\bFUNCTION\b',
        ],
    }

    @classmethod
    def detect(cls, text: str) -> CodeLanguage:
        if not text.strip():
            return CodeLanguage.UNKNOWN
        scores = {lang: 0 for lang in CodeLanguage}
        for lang, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
                scores[lang] += len(matches)
        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]
        return best_lang if best_score >= 2 else CodeLanguage.UNKNOWN


# ─────────────────────────────────────────────
# Code block detector
# ─────────────────────────────────────────────

class CodeBlockDetector:
    """
    Determines whether a region of text is code BEFORE running Tesseract
    on it, so we can switch to the code-optimized config.
    Uses visual features of the image region (monospace font, indentation).
    """

    @staticmethod
    def is_code_region(region: np.ndarray) -> bool:
        """
        Heuristics on the grayscale region image:
        1. Monospace fonts produce uniform character spacing → low horizontal variance
        2. Code blocks often have leading whitespace (left indentation)
        3. Line lengths are more uniform than prose
        4. Background is often slightly different (syntax highlighting remnants)
        """
        if region is None or region.size == 0:
            return False

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) \
               if len(region.shape) == 3 else region
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = binary.shape

        # ── Check left-margin indentation variance (code has indented lines)
        row_profiles = []
        for r in range(h):
            row = binary[r, :]
            nonzero = np.where(row > 0)[0]
            if len(nonzero) > 0:
                row_profiles.append(nonzero[0])   # First ink pixel per row
        if not row_profiles:
            return False

        indent_variance = float(np.var(row_profiles))

        # ── Check line-length uniformity (code lines end at similar x positions)
        line_lengths = []
        for r in range(h):
            row = binary[r, :]
            nonzero = np.where(row > 0)[0]
            if len(nonzero) > 2:
                line_lengths.append(nonzero[-1] - nonzero[0])

        length_variance = float(np.var(line_lengths)) if line_lengths else 0

        # ── Check for monospace: character bounding box widths should be uniform
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        char_widths = [cv2.boundingRect(c)[2] for c in contours
                       if 5 < cv2.boundingRect(c)[2] < w * 0.08]
        char_width_variance = float(np.var(char_widths)) if len(char_widths) > 10 else 999

        # Decision: code has high indent variance + low char width variance
        is_code = (indent_variance > 50 and char_width_variance < 40) \
               or (indent_variance > 200)

        logger.debug(f"CodeRegion: indent_var={indent_variance:.1f} "
                     f"char_var={char_width_variance:.1f} → is_code={is_code}")
        return is_code

    @staticmethod
    def has_code_markers_in_text(text: str) -> bool:
        """
        Quick text-level check: does the OCR'd text look like code?
        Used as a secondary verification after visual detection.
        """
        code_indicators = [
            r'[{}()\[\];]',             # Brackets and semicolons
            r'\b(def|class|import|for|while|if|else|return|function|var|let|const)\b',
            r'==|!=|<=|>=|->|=>|:=',   # Operators
            r'#include|#define',
            r'\bNULL\b|\bnull\b|\bNone\b|\bundefined\b',
            r'[a-z_]+\([^)]*\)',        # Function calls
            r'^\s{2,}',                 # Leading whitespace (indented code)
        ]
        score = sum(
            len(re.findall(pat, text, re.MULTILINE))
            for pat in code_indicators
        )
        return score >= 3


# ─────────────────────────────────────────────
# OCR post-processor
# ─────────────────────────────────────────────

class OCRPostProcessor:
    """
    Cleans common Tesseract errors on typed text and code.
    Runs AFTER Tesseract, before chunking.
    """

    # Common single-character misreads
    CHAR_CORRECTIONS = {
        # In code context: Tesseract confuses these constantly
        '\u00b0': '°',   # degree sign preserved
        '\u2019': "'",   # right single quote → apostrophe
        '\u201c': '"',   # curly open quote → straight
        '\u201d': '"',   # curly close quote → straight
        '\u2014': '-',   # em dash → hyphen (in code)
        '\u2013': '-',   # en dash → hyphen
        '\u00d7': '*',   # multiplication × → *
        '\u00f7': '/',   # division ÷ → /
        '\ufb01': 'fi',  # fi ligature
        '\ufb02': 'fl',  # fl ligature
        '\u2022': '-',   # bullet → dash
    }

    # Tesseract specific character substitution errors in code
    CODE_CHAR_FIXES = [
        (r'(?<=[a-zA-Z0-9_])\|(?=[a-zA-Z0-9_])', 'l'),   # | → l between words
        (r'(?<!\|)\|(?!\|)', 'l'),                          # lone | → l (usually)
        (r'\bO(?=[0-9])', '0'),     # O followed by digit → 0
        (r'(?<=[0-9])O\b', '0'),    # O after digit → 0
        (r'\bl(?=[0-9])', '1'),     # l before digit → 1
        (r'(?<=[0-9])l\b', '1'),    # l after digit → 1
        (r'rn\b', 'm'),             # rn → m (very common Tesseract error)
        (r'\bI(?=[a-z])', 'l'),     # Capital I before lowercase → lowercase l
        (r'(?<=\s)_(?=\s)', '-'),   # Lone _ → - (dashes misread)
        (r'``', '"'),               # Double backticks → quote
        (r"''", '"'),               # Double apostrophes → quote
    ]

    # Tesseract errors specific to text (not code)
    TEXT_FIXES = [
        (r'\b([A-Z])\s(?=[a-z])', r'\1'),   # Remove spurious space after capital
        (r'\s{2,}', ' '),                    # Collapse multiple spaces
        (r'(?<=[a-z])-\s*\n\s*(?=[a-z])', ''),  # Rejoin hyphenated line breaks
    ]

    @classmethod
    def clean_text(cls, text: str, is_code: bool = False) -> str:
        if not text:
            return text

        # ── Unicode normalization for common OCR artifacts
        for bad_char, good_char in cls.CHAR_CORRECTIONS.items():
            text = text.replace(bad_char, good_char)

        if is_code:
            # Apply code-specific character fixes
            for pattern, replacement in cls.CODE_CHAR_FIXES:
                text = re.sub(pattern, replacement, text)

            # Preserve indentation — DO NOT strip leading whitespace
            # Only remove trailing whitespace from lines
            lines = text.split('\n')
            lines = [line.rstrip() for line in lines]

            # Remove completely blank leading/trailing lines
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()

            text = '\n'.join(lines)

        else:
            # Apply text-specific fixes
            for pattern, replacement in cls.TEXT_FIXES:
                text = re.sub(pattern, replacement, text)

            # Clean up excessive whitespace in prose
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()

        return text

    @staticmethod
    def detect_block_type(text: str, bbox: tuple,
                          image_height: int) -> BlockType:
        """
        Infer the semantic type of a block from text content and position.
        """
        stripped = text.strip()
        if not stripped:
            return BlockType.UNKNOWN

        x, y, w, h = bbox

        # Heading: short, at top of page, possibly all caps or large font
        if (y < image_height * 0.15 and len(stripped) < 120
                and '\n' not in stripped):
            return BlockType.HEADING

        # List item: starts with bullet, number, or letter+period
        if re.match(r'^[\s]*[-•*]\s', stripped) \
        or re.match(r'^[\s]*\d+[.)]\s', stripped) \
        or re.match(r'^[\s]*[a-zA-Z][.)]\s', stripped):
            return BlockType.LIST_ITEM

        # Caption: short, near bottom of image
        if y > image_height * 0.85 and len(stripped) < 200:
            return BlockType.CAPTION

        # Formula: contains math symbols
        math_symbols = re.findall(r'[∑∫∂∇×÷±√∞αβγδεζηθ=<>]', stripped)
        if len(math_symbols) >= 2:
            return BlockType.FORMULA

        return BlockType.PARAGRAPH


# ─────────────────────────────────────────────
# Column layout detector
# ─────────────────────────────────────────────

class ColumnDetector:
    """
    Detects whether a page has single or multi-column layout.
    Multi-column requires PSM 4 instead of PSM 6 in Tesseract.
    """

    @staticmethod
    def is_multicolumn(binary: np.ndarray,
                       min_gap_fraction: float = 0.05) -> bool:
        """
        Project the binary image onto the x-axis (vertical projection).
        A wide vertical valley in the middle = column separator.
        """
        projection = np.sum(binary == 0, axis=0).astype(np.float32)
        w = len(projection)
        if w == 0:
            return False

        # Smooth the projection
        kernel = np.ones(max(1, w // 50)) / max(1, w // 50)
        smoothed = np.convolve(projection, kernel, mode='same')

        # Look for a valley in the middle third of the page
        mid_start = w // 3
        mid_end   = 2 * w // 3
        mid_region = smoothed[mid_start:mid_end]

        if mid_region.size == 0 or smoothed.max() == 0:
            return False

        # Normalise
        norm = mid_region / smoothed.max()
        min_val = norm.min()

        # A valley below 5% density = column gap
        is_multi = bool(min_val < min_gap_fraction)
        logger.debug(f"ColumnDetect: mid_min={min_val:.3f} → multicolumn={is_multi}")
        return is_multi


# ─────────────────────────────────────────────
# Main Tesseract OCR engine
# ─────────────────────────────────────────────

class TesseractOCREngine:
    """
    Full OCR engine combining:
    - The preprocessing pipeline (preprocessor.py)
    - Content-type detection (code vs text)
    - Multi-pass Tesseract with per-region configs
    - Post-processing and structured output

    Usage:
        engine = TesseractOCREngine()
        result = engine.run("page.jpg")
        print(result.full_text)
        for block in result.code_blocks:
            print(f"[{block.language.value}]\\n{block.text}")
    """

    # Discard blocks with confidence below this
    MIN_CONFIDENCE = 45
    # Discard blocks shorter than this (noise)
    MIN_TEXT_LENGTH = 3

    def __init__(self, debug: bool = False):
        self.preprocessor = OCRPreprocessingPipeline(debug=debug)
        self.debug = debug

    # ─────────────────────────────────────────
    # Primary entry point
    # ─────────────────────────────────────────

    def run(self, source, page: int = 1) -> OCRResult:
        """
        Full pipeline:
        Load → preprocess → detect layout → multi-pass OCR → post-process
        """
        # Step 1: Preprocess
        prep: PipelineResult = self.preprocessor.run(source)
        binary  = prep.image_binary
        img_bgr = prep.image
        diag    = prep.diagnostics

        h, w = binary.shape[:2]

        # Step 2: Detect column layout
        is_multi = ColumnDetector.is_multicolumn(binary)

        # Step 3: Detect whether the whole page is code
        page_is_code = CodeBlockDetector.is_code_region(img_bgr)

        # Step 4: Select primary Tesseract config
        if page_is_code:
            primary_config = TesseractConfig.for_code()
        elif diag.detected_type == ImageType.NOISY_PRINT:
            primary_config = TesseractConfig.for_sparse_text()
        else:
            primary_config = TesseractConfig.for_general_text(
                multicolumn=is_multi)

        # Step 5: Run primary OCR pass to get block layout
        raw_data = pytesseract.image_to_data(
            binary,
            config=primary_config,
            output_type=Output.DICT
        )

        # Step 6: Group into logical blocks and run per-block passes
        blocks = self._build_blocks(raw_data, binary, img_bgr, h, w, page)

        # Step 7: Assemble final output
        return self._assemble_result(blocks, prep, primary_config, page)

    # ─────────────────────────────────────────
    # Block building
    # ─────────────────────────────────────────

    def _build_blocks(self, raw_data: dict, binary: np.ndarray,
                      img_bgr: np.ndarray, h: int, w: int,
                      page: int) -> list[TextBlock]:
        """
        Groups Tesseract word-level output into logical blocks,
        then runs a second targeted OCR pass on blocks detected as code.
        """
        # Group words by Tesseract block_num
        block_groups: dict[int, list] = {}
        n = len(raw_data["text"])
        for i in range(n):
            word = str(raw_data["text"][i]).strip()
            conf = float(raw_data["conf"][i])
            bn   = int(raw_data["block_num"][i])

            if conf < 0:   # Tesseract uses -1 for non-word rows
                continue

            if bn not in block_groups:
                block_groups[bn] = {
                    "words": [], "confs": [],
                    "x1": [], "y1": [], "x2": [], "y2": []
                }
            block_groups[bn]["words"].append(word)
            block_groups[bn]["confs"].append(conf)
            block_groups[bn]["x1"].append(raw_data["left"][i])
            block_groups[bn]["y1"].append(raw_data["top"][i])
            block_groups[bn]["x2"].append(
                raw_data["left"][i] + raw_data["width"][i])
            block_groups[bn]["y2"].append(
                raw_data["top"][i] + raw_data["height"][i])

        blocks: list[TextBlock] = []
        for bid, grp in block_groups.items():
            if not grp["words"]:
                continue

            # Compute aggregate bounding box
            bx1 = min(grp["x1"])
            by1 = min(grp["y1"])
            bx2 = max(grp["x2"])
            by2 = max(grp["y2"])
            bbox = (bx1, by1, bx2 - bx1, by2 - by1)

            raw_text = " ".join(w for w in grp["words"] if w)
            mean_conf = float(np.mean([c for c in grp["confs"] if c >= 0])) \
                        if grp["confs"] else 0.0

            if mean_conf < self.MIN_CONFIDENCE:
                continue
            if len(raw_text.strip()) < self.MIN_TEXT_LENGTH:
                continue

            # Extract the image region for this block
            pad = 4
            rx1 = max(0, bx1 - pad)
            ry1 = max(0, by1 - pad)
            rx2 = min(w, bx2 + pad)
            ry2 = min(h, by2 + pad)
            region_bgr = img_bgr[ry1:ry2, rx1:rx2]
            region_bin = binary[ry1:ry2, rx1:rx2]

            # Decide if this block is code
            is_code_visual  = CodeBlockDetector.is_code_region(region_bgr) \
                              if region_bgr.size > 0 else False
            is_code_text    = CodeBlockDetector.has_code_markers_in_text(raw_text)
            is_code_block   = is_code_visual or is_code_text

            # Re-run OCR with code config if needed
            if is_code_block and not is_code_visual:
                # Text heuristic said code but visual didn't; do a targeted re-run
                better_text = pytesseract.image_to_string(
                    region_bin if region_bin.size > 0 else binary,
                    config=TesseractConfig.for_code()
                )
                raw_text = better_text if better_text.strip() else raw_text
            elif is_code_visual:
                # Visual said code; always re-run with code config for accuracy
                better_text = pytesseract.image_to_string(
                    region_bin if region_bin.size > 0 else binary,
                    config=TesseractConfig.for_code()
                )
                raw_text = better_text if better_text.strip() else raw_text

            # Post-process the text
            cleaned = OCRPostProcessor.clean_text(raw_text, is_code=is_code_block)

            # Detect language if code
            lang = None
            if is_code_block:
                lang = CodeLanguageDetector.detect(cleaned)

            # Detect block type
            block_type = BlockType.CODE if is_code_block else \
                         OCRPostProcessor.detect_block_type(cleaned, bbox, h)

            line_count = len([l for l in cleaned.split('\n') if l.strip()])

            block = TextBlock(
                block_id    = bid,
                block_type  = block_type,
                text        = cleaned,
                raw_text    = raw_text,
                confidence  = mean_conf,
                page        = page,
                bbox        = bbox,
                language    = lang,
                is_code     = is_code_block,
                line_count  = line_count,
            )
            blocks.append(block)

        # Sort blocks top-to-bottom, left-to-right (reading order)
        blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        return blocks

    # ─────────────────────────────────────────
    # Result assembly
    # ─────────────────────────────────────────

    def _assemble_result(self, blocks: list[TextBlock],
                         prep: PipelineResult,
                         config_used: str,
                         page: int) -> OCRResult:
        code_blocks = [b for b in blocks if b.is_code]
        all_text_parts = []

        for b in blocks:
            if not b.text.strip():
                continue
            if b.is_code:
                lang = b.language.value if b.language else "unknown"
                all_text_parts.append(f"[CODE:{lang}]\n{b.text}\n[/CODE]")
            else:
                all_text_parts.append(b.text)

        full_text = "\n\n".join(all_text_parts)
        all_confs = [b.confidence for b in blocks]
        mean_conf = float(np.mean(all_confs)) if all_confs else 0.0

        warnings = list(prep.diagnostics.warnings)
        if mean_conf < 60:
            warnings.append(
                f"Low mean OCR confidence ({mean_conf:.1f}). "
                "Consider improving scan quality."
            )
        if any(b.confidence < 50 for b in code_blocks):
            warnings.append(
                "Some code blocks have low confidence. "
                "Verify code output manually."
            )

        return OCRResult(
            page             = page,
            blocks           = blocks,
            full_text        = full_text,
            code_blocks      = code_blocks,
            mean_confidence  = mean_conf,
            image_type       = prep.diagnostics.detected_type.value,
            skew_corrected   = abs(prep.diagnostics.skew_angle) > 0.3,
            skew_angle       = prep.diagnostics.skew_angle,
            tesseract_config = config_used,
            stages_applied   = prep.stages_applied,
            warnings         = warnings,
        )
