"""
===================================================================
  UNIVERSAL DOCUMENT INGESTION PIPELINE  →  PINECONE
  No AI summarisation — rich deterministic metadata per chunk
===================================================================
Supported formats : PDF | PPTX | DOCX | TXT | MD
Vector store      : Pinecone (serverless or pod-based)
Embeddings        : Gemini gemini-embedding-2 (1536-dim by default)
===================================================================

QUICK START
-----------
1. Install dependencies:
      pip install -r requirements.txt

   System deps (Linux):
      apt-get install poppler-utils tesseract-ocr libmagic-dev

   System deps (Mac):
      brew install poppler tesseract libmagic

2. Create a .env file (copy .env.example and fill in your keys).

3. Run:
      python ingestion_pipeline_pinecone.py
   or import and call run_pipeline() from your own code.
===================================================================
"""

print("TOP OF FILE")
import asyncio
import hashlib
import json
import os
import re
import sys
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

print("AFTER LANGCHAIN IMPORTS")
# ── LangChain ────────────────────────────────────────────────────
from langchain_core.documents import Document
from pydantic import BaseModel, Field

print("AFTER GOOGLE IMPORTS")
# ── Gemini embeddings ────────────────────────────────────────────
from google import genai
from google.genai import types


# ── Pinecone v3+ ─────────────────────────────────────────────────
from pinecone import Pinecone, ServerlessSpec
print("AFTER PINECONE IMPORTS")
# ── Unstructured (universal parser) ──────────────────────────────
print("IMPORTING partition")
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
print("IMPORTING partition")
from unstructured.chunking.title import chunk_by_title
print("AFTER UNSTRUCTURED IMPORTS")

# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
PINECONE_ENV        = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "documents")

EMBEDDING_MODEL       = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM         = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))
EMBEDDING_CONCURRENCY = int(os.getenv("EMBEDDING_CONCURRENCY", "10"))  # max concurrent Gemini embed calls
IMAGE_SUMMARY_MODEL = os.getenv("GEMINI_IMAGE_SUMMARY_MODEL", "gemini-2.5-flash-lite")
ENABLE_IMAGE_SUMMARIES = os.getenv("ENABLE_IMAGE_SUMMARIES", "true").lower() == "true"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
IMAGE_SUMMARY_PROMPT_FILE = PROMPTS_DIR / "image_summary_few_shot.md"

# PDF parsing strategy. hi_res is the default because it preserves richer
# layout/table/image structure for downstream retrieval quality.
PDF_PARTITION_STRATEGY = os.getenv("DOCUMENT_PDF_STRATEGY", "hi_res").lower()

# Use hi_res only when image summarization is enabled AND caller requests it.
# hi_res is 10-20x slower (runs layout detection + Tesseract OCR).
# For text-native PDFs (most course slides/papers), fast is sufficient.
PDF_USE_HI_RES_FOR_IMAGES = os.getenv("PDF_USE_HI_RES_FOR_IMAGES", "true").lower() == "true"

# Chunking knobs
CHUNK_MAX_CHARS = 3000
CHUNK_NEW_AFTER = 2400
CHUNK_MIN_CHARS = 500
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "200"))

# Keyword extraction
MAX_KEYWORDS    = 12    # top N keywords stored per chunk
MIN_KEYWORD_LEN = 4     # ignore tokens shorter than this

# Pinecone upsert batch size
UPSERT_BATCH    = 100


class ImageSummary(BaseModel):
    """Structured summary for images extracted from source documents."""

    image_type: str = Field(..., description="chart, diagram, table_image, photo, screenshot, formula, or other")
    summary: str = Field(..., description="Concise description of the image content")
    detected_text: str = Field("", description="Any readable text/OCR-like content visible in the image")
    educational_value: str = Field(..., description="Why this image matters for learning or retrieval")
    key_entities: List[str] = Field(default_factory=list, description="Important visible objects, labels, concepts, or entities")
    relationships: List[str] = Field(default_factory=list, description="Important relationships, flows, comparisons, or trends")
    searchable_keywords: List[str] = Field(default_factory=list, description="Short retrieval keywords for this image")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in the summary")


# ════════════════════════════════════════════════════════════════
#  ENGLISH STOPWORD LIST  (no NLTK required)
# ════════════════════════════════════════════════════════════════

_STOPWORDS: set = {
    "a","about","above","after","again","against","all","also","an","and",
    "any","are","as","at","be","because","been","before","being","below",
    "between","both","but","by","can","did","do","does","doing","down",
    "during","each","few","for","from","further","get","got","had","has",
    "have","having","he","her","here","him","his","how","i","if","in",
    "into","is","it","its","itself","just","like","more","most","my",
    "no","not","now","of","on","once","only","or","other","our","out",
    "over","own","s","same","she","should","so","some","such","than",
    "that","the","their","them","then","there","these","they","this",
    "those","through","to","too","under","until","up","us","very","was",
    "we","were","what","when","where","which","while","who","with","you",
    "your","will","would","could","may","might","shall","its","said",
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "new","use","used","using","make","made","know","see","many","much",
    "well","also","within","without","per","etc","via","based","across",
}


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def _extract_keywords(text: str, top_n: int = MAX_KEYWORDS) -> str:
    """
    Zero-dependency keyword extraction via token frequency.

    Approach:
      1. Lowercase + strip punctuation
      2. Remove stopwords and short tokens
      3. Return the top-N by frequency as a comma-joined string
    """
    tokens = re.findall(r"[a-z]{" + str(MIN_KEYWORD_LEN) + r",}", text.lower())
    freq: dict = {}
    for tok in tokens:
        if tok not in _STOPWORDS:
            freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq, key=lambda k: freq[k], reverse=True)[:top_n]
    return ", ".join(ranked)


def _safe_page(el) -> Optional[int]:
    """Try to get a page number from an element's metadata; return None if absent."""
    try:
        return el.metadata.page_number  # type: ignore[attr-defined]
    except AttributeError:
        return None


def _chunk_id(source: str, chunk_index: int) -> str:
    """Stable, deterministic ID so re-ingesting the same file overwrites rather than duplicates."""
    raw = f"{Path(source).resolve()}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _get_metadata_value(el, *names: str):
    metadata = getattr(el, "metadata", None)
    if metadata is None:
        return None

    for name in names:
        value = getattr(metadata, name, None)
        if value:
            return value

    if hasattr(metadata, "to_dict"):
        data = metadata.to_dict()
        for name in names:
            value = data.get(name)
            if value:
                return value

    return None


def _extract_image_payload(el) -> tuple[bytes, str] | None:
    metadata = getattr(el, "metadata", None)
    if metadata is None:
        return None

    # Priority 1: image_path on disk (unstructured >= 0.13 with hi_res)
    image_path = _get_metadata_value(el, "image_path")
    if image_path:
        try:
            img_p = Path(str(image_path))
            if img_p.exists():
                img_bytes = img_p.read_bytes()
                suffix = img_p.suffix.lower().lstrip(".")
                mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                            "png": "image/png",  "gif": "image/gif",
                            "webp": "image/webp", "tiff": "image/tiff"}
                mime = mime_map.get(suffix, "image/png")
                print(f"   ✅  Image loaded from disk: {img_p.name}  {len(img_bytes)} bytes  mime={mime}")
                return img_bytes, mime
            else:
                print(f"   ⚠️  image_path attribute found but file does not exist: {image_path}")
        except Exception as e:
            print(f"   ⚠️  Failed to read image from disk '{image_path}': {e}")

    # Debug: print all metadata keys so we can identify what the installed
    # version of unstructured actually stores
    if hasattr(metadata, "to_dict"):
        all_keys = list(metadata.to_dict().keys())
        img_keys = [k for k in all_keys if "image" in k.lower() or "base64" in k.lower() or "figure" in k.lower()]
        if img_keys:
            print(f"   🔍 Potential image keys in metadata: {img_keys}")
        else:
            print(f"   ⚠️  No image/base64 keys found. All metadata keys: {all_keys}")

    # Priority 2: Check for base64 (legacy Unstructured)
    raw_payload = _get_metadata_value(
        el,
        "image_base64",
        "image_base64_data",
        # older versions
        "image_payload",
        # additional fallback names
        "base64_image",
        "image_data",
    )

    if not raw_payload:
        print("   ⚠️  Image payload not found via image_path or any base64 attribute.")
        return None

    mime_type = _get_metadata_value(
        el, "image_mime_type", "mime_type", "image_type"
    ) or "image/png"

    payload = str(raw_payload)
    if "," in payload and payload.lower().startswith("data:"):
        header, payload = payload.split(",", 1)
        if ";" in header:
            mime_type = header[5:].split(";", 1)[0] or mime_type

    try:
        decoded = base64.b64decode(payload)
        print(f"   ✅  Image payload decoded: {len(decoded)} bytes, mime: {mime_type}")
        return decoded, mime_type
    except Exception as e:
        print(f"   ⚠️  base64 decode failed: {e}")
        return None


def _summarize_image_with_gemini(image_bytes: bytes, mime_type: str) -> ImageSummary:
    prompt = _load_prompt(IMAGE_SUMMARY_PROMPT_FILE)
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=IMAGE_SUMMARY_MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ImageSummary,
            temperature=0.2,
        ),
    )

    if getattr(response, "parsed", None) is not None:
        return response.parsed

    return ImageSummary.model_validate_json(response.text)


def _image_summary_to_embedding_text(summary: ImageSummary) -> str:
    parts = [
        f"[IMAGE TYPE: {summary.image_type}]",
        f"[IMAGE SUMMARY: {summary.summary}]",
        f"[IMAGE EDUCATIONAL VALUE: {summary.educational_value}]",
    ]
    if summary.detected_text:
        parts.append(f"[IMAGE DETECTED TEXT: {summary.detected_text}]")
    if summary.key_entities:
        parts.append("[IMAGE KEY ENTITIES: " + ", ".join(summary.key_entities) + "]")
    if summary.relationships:
        parts.append("[IMAGE RELATIONSHIPS: " + " | ".join(summary.relationships) + "]")
    if summary.searchable_keywords:
        parts.append("[IMAGE KEYWORDS: " + ", ".join(summary.searchable_keywords) + "]")
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════
#  STEP 1 – PARTITION
# ════════════════════════════════════════════════════════════════

def partition_document(file_path: str) -> list:
    """
    Route any supported file type through unstructured's auto-partitioner.

    PDFs  → hi_res strategy + table structure + image extraction
    PPTX  → slide-aware with page breaks
    DOCX  → table structure inference
    TXT/MD → plain text
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    print(f"\n📄  Partitioning [{ext.upper()}]  {path.name}")

    kwargs: dict = {"filename": str(path)}

    if ext == ".pdf":
        # Determine strategy:
        # hi_res extracts image base64 payloads but is 10-20x slower.
        # Only use it when image summarization is enabled AND explicitly requested.
        # For text-native PDFs (most course materials), fast is correct.
        use_hi_res = ENABLE_IMAGE_SUMMARIES and PDF_USE_HI_RES_FOR_IMAGES
        pdf_strategy = "hi_res" if use_hi_res else "fast"
        print(f"   PDF strategy: {pdf_strategy} (ENABLE_IMAGE_SUMMARIES={ENABLE_IMAGE_SUMMARIES}, PDF_USE_HI_RES_FOR_IMAGES={PDF_USE_HI_RES_FOR_IMAGES})")
        kwargs.update(strategy=pdf_strategy)
        if use_hi_res:
            kwargs.update(extract_images_in_pdf=True)
    elif ext in {".pptx", ".ppt"}:
        kwargs.update(include_page_breaks=True)
    elif ext in {".docx", ".doc"}:
        kwargs.update(infer_table_structure=True)
    # .txt / .md → no extra kwargs needed

    if ext == ".pdf":
        elements = partition_pdf(**kwargs)
    elif ext in {".pptx", ".ppt"}:
        elements = partition_pptx(**kwargs)
    elif ext in {".docx", ".doc"}:
        elements = partition_docx(**kwargs)
    elif ext in {".txt", ".md"}:
        elements = partition_text(filename=str(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    print(f"   ✅  {len(elements)} elements extracted")
    return elements


def _cleanup_figures_dir(figures_dir: Path | None = None) -> None:
    """
    Delete the figures/ temp directory that unstructured creates when
    extract_images_in_pdf=True.  Must be called AFTER build_documents()
    has finished reading image bytes — not before.

    Safe to call even if the directory does not exist.
    """
    import shutil
    target = figures_dir or (Path.cwd() / "figures")
    if target.exists() and target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        print(f"   🗑️   Cleaned up temp figures dir: {target}")


# ════════════════════════════════════════════════════════════════
#  STEP 2 – CHUNK
# ════════════════════════════════════════════════════════════════

def create_chunks(elements: list) -> list:
    """
    Title-aware chunking.
    A new chunk starts whenever a Title element is encountered
    or when the soft-limit (CHUNK_NEW_AFTER) is hit.
    Small fragments are merged up to CHUNK_MIN_CHARS.
    """
    print("\n🔨  Chunking elements …")
    chunks = chunk_by_title(
        elements,
        max_characters=CHUNK_MAX_CHARS,
        new_after_n_chars=CHUNK_NEW_AFTER,
        combine_text_under_n_chars=CHUNK_MIN_CHARS,
        overlap=CHUNK_OVERLAP,
        overlap_all=False,
    )
    print(f"   ✅  {len(chunks)} chunks created")
    return chunks


# ════════════════════════════════════════════════════════════════
#  STEP 3 – BUILD RICH DOCUMENTS  (no AI, zero cost)
# ════════════════════════════════════════════════════════════════

def build_documents(
    chunks: list,
    source_file: str,
    course_id: str = "",
) -> List[Document]:
    """
    Convert unstructured chunks → LangChain Documents with maximum metadata.

    EMBEDDING TEXT  (what gets vectorised)
    ──────────────────────────────────────
    Prepend the section title so the embedding "knows" the context,
    then the raw chunk text, then plain-text of any tables.
    This gives richer semantic signal with zero extra cost.

    METADATA SCHEMA
    ───────────────
    Identity        chunk_id, source, file_name, file_type
    Position        chunk_index, total_chunks, page_start, page_end
    Structure       section_title, element_types
    Content flags   has_table, has_image, has_list
                    table_count, image_count, list_item_count
    Statistics      char_count, word_count
    Quick display   text_preview (first 300 chars)
    Retrieval aids  keywords (frequency-based, top 12)
    Raw content     raw_text, tables_text (plain text of tables)
    Audit           ingested_at (ISO-8601 UTC)
    """
    path     = Path(source_file)
    file_ext = path.suffix.lower().lstrip(".")
    total    = len(chunks)

    print(f"\n🗂️   Building documents for '{path.name}' ({total} chunks) …")
    docs: List[Document] = []

    for i, chunk in enumerate(chunks):

        # ── 3a. Parse orig_elements ──────────────────────────────
        orig_elements = []
        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
            orig_elements = chunk.metadata.orig_elements or []

        element_type_names: List[str] = []
        section_title      = ""
        tables_html: List[str] = []
        tables_text: List[str] = []
        image_summaries: List[ImageSummary] = []
        image_count        = 0
        list_item_count    = 0
        page_numbers: List[int] = []

        for el in orig_elements:
            kind = type(el).__name__
            element_type_names.append(kind)

            # Section title – first Title element wins
            if kind == "Title" and not section_title:
                section_title = el.text.strip()

            # Tables – collect both HTML and plain text
            elif kind == "Table":
                html = getattr(el.metadata, "text_as_html", "") or ""
                plain = el.text.strip()
                if html:
                    tables_html.append(html)
                if plain:
                    tables_text.append(plain)

            # Images – summarize extracted payloads; never store base64 in Pinecone.
            elif kind == "Image":
                image_count += 1
                if ENABLE_IMAGE_SUMMARIES:
                    payload = _extract_image_payload(el)
                    if payload:
                        image_bytes, mime_type = payload
                        try:
                            image_summaries.append(
                                _summarize_image_with_gemini(image_bytes, mime_type)
                            )
                        except Exception as exc:
                            print(f"   ⚠️  Image summary failed: {exc}")
                else:
                    print(f"   ⚠️  Image {image_count} skipped — no base64 payload found")

            # List items
            elif kind in {"ListItem", "List"}:
                list_item_count += 1

            # Page numbers
            pg = _safe_page(el)
            if pg is not None:
                page_numbers.append(pg)

        # If no Title found in orig_elements, fall back to chunk-level metadata
        if not section_title:
            try:
                section_title = chunk.metadata.section or ""
            except AttributeError:
                section_title = ""

        # ── 3b. Deduplicate element type list ────────────────────
        seen: set = set()
        unique_types: List[str] = []
        for t in element_type_names:
            if t not in seen:
                seen.add(t)
                unique_types.append(t)

        # ── 3c. Page range ───────────────────────────────────────
        page_start = min(page_numbers) if page_numbers else -1
        page_end   = max(page_numbers) if page_numbers else -1

        # ── 3d. Text stats ───────────────────────────────────────
        raw_text   = chunk.text or ""
        char_count = len(raw_text)
        word_count = len(raw_text.split())

        # ── 3e. Keywords (frequency-based, no AI) ───────────────
        full_text_for_kw = raw_text + " " + " ".join(tables_text)
        keywords = _extract_keywords(full_text_for_kw)

        # ── 3f. Build embedding text ─────────────────────────────
        # Layout:
        #   [SECTION: <title>]          ← context anchor
        #   <raw chunk text>            ← primary content
        #   [TABLE: <plain text>] ...   ← table content (searchable)
        parts: List[str] = []
        if section_title:
            parts.append(f"[SECTION: {section_title}]")
        parts.append(raw_text)
        for tbl in tables_text:
            parts.append(f"[TABLE: {tbl}]")
        for image_summary in image_summaries:
            parts.append(_image_summary_to_embedding_text(image_summary))
        embedding_text = "\n\n".join(p for p in parts if p.strip())

        # ── 3g. Metadata dict ────────────────────────────────────
        metadata = {
            # Identity
            "chunk_id"        : _chunk_id(source_file, i),
            "source"          : str(path),
            "file_name"       : path.name,
            "file_type"       : file_ext,
            "course_id"       : course_id,

            # Position
            "chunk_index"     : i,
            "total_chunks"    : total,
            "page_start"      : page_start,
            "page_end"        : page_end,

            # Structure
            "section_title"   : section_title,
            "element_types"   : ", ".join(unique_types),

            # Content flags (great for Pinecone metadata filters)
            "has_table"       : len(tables_html) > 0,
            "has_image"       : image_count > 0,
            "has_list"        : list_item_count > 0,
            "table_count"     : len(tables_html),
            "image_count"     : image_count,
            "list_item_count" : list_item_count,
            "image_summary_count": len(image_summaries),

            # Statistics
            "char_count"      : char_count,
            "word_count"      : word_count,

            # Quick display (no need to re-fetch the full vector)
            "text_preview"    : raw_text[:300],

            # Retrieval aids
            "keywords"        : keywords,

            # Raw content (available at retrieval time)
            "raw_text"        : raw_text,
            "tables_text"     : " | ".join(tables_text),   # pipe-separated plain tables
            "image_summaries"  : json.dumps(
                [summary.model_dump() for summary in image_summaries],
                ensure_ascii=False,
            ),

            # Audit
            "ingested_at"     : datetime.now(timezone.utc).isoformat(),
        }

        docs.append(Document(page_content=embedding_text, metadata=metadata))

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"   [{i+1}/{total}] done")

    print(f"   ✅  {len(docs)} documents built")
    return docs


# ════════════════════════════════════════════════════════════════
#  STEP 4 – PINECONE UPSERT
# ════════════════════════════════════════════════════════════════

def _get_gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")
    return genai.Client(api_key=GEMINI_API_KEY)


async def _embed_single_text_async(
    client,
    text: str,
    task_type: str,
) -> list:
    """Embed one text asynchronously using asyncio.to_thread()."""
    result = await asyncio.to_thread(
        client.models.embed_content,
        model=EMBEDDING_MODEL,
        contents=text if text.strip() else " ",
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return result.embeddings[0].values


async def _embed_texts_async(
    texts: list,
    task_type: str,
) -> list:
    """
    Embed all texts concurrently, bounded by EMBEDDING_CONCURRENCY.
    Uses asyncio.to_thread() to run the sync SDK call in a thread pool.
    Semaphore prevents QPM rate-limit hits.
    Includes exponential backoff on 429 errors.
    """
    client    = _get_gemini_client()
    semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    async def _embed_with_semaphore(text: str) -> list:
        async with semaphore:
            for attempt in range(4):
                try:
                    return await _embed_single_text_async(client, text, task_type)
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        wait = (2 ** attempt) + (0.1 * attempt)
                        print(f"   Rate limited, retrying in {wait:.1f}s...")
                        await asyncio.sleep(wait)
                    else:
                        raise
            raise RuntimeError("Embedding failed after 4 attempts")

    tasks = [_embed_with_semaphore(text) for text in texts]
    return await asyncio.gather(*tasks)


def _embed_texts_with_gemini(
    texts: List[str],
    task_type: str,
    batch_size: int = 100,   # kept for API compatibility, not used
) -> List[list]:
    """Public sync wrapper — runs the async embedder in a fresh event loop."""
    return asyncio.run(_embed_texts_async(texts, task_type))

def _get_or_create_index(pc: Pinecone, index_name: str) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"\n🌲  Creating Pinecone index '{index_name}' …")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_ENV),
        )
        print("   ✅  Index created")
    else:
        print(f"\n🌲  Using existing index '{index_name}'")


def store_in_pinecone(
    documents: List[Document],
    namespace: str = PINECONE_NAMESPACE,
    index_name: str = PINECONE_INDEX_NAME,
) -> None:
    """
    Embed documents with Gemini and upsert into Pinecone.

    Vector ID = chunk_id from metadata → deterministic, so re-ingesting
    the same file overwrites existing vectors instead of duplicating.

    Metadata stored per vector:
      All fields from build_documents() except tables_html (too large).
      raw_text is trimmed to 8 000 chars to stay safely under Pinecone's
      40 KB per-vector metadata limit.
    """
    print("\n📌  Embedding & upserting to Pinecone …")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("\n========== PINECONE DEBUG ==========")
    print("API KEY PREFIX:", PINECONE_API_KEY[:10])
    print("INDEXES:", pc.list_indexes())
    print("TARGET INDEX:", index_name)
    print("====================================\n")
    _get_or_create_index(pc, index_name=index_name)
    index = pc.Index(index_name)

    texts = [doc.page_content for doc in documents]
    print(f"   Embedding {len(texts)} texts (this may take a moment) …")
    vectors = _embed_texts_with_gemini(texts, task_type="RETRIEVAL_DOCUMENT")
    if len(vectors) != len(documents):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} vectors "
            f"for {len(documents)} documents."
        )

    upsert_data = []
    for doc, vec in zip(documents, vectors):
        m = doc.metadata.copy()
        # Pinecone 40 KB metadata cap: trim heavy fields
        m["raw_text"]     = m.get("raw_text", "")[:8_000]
        m["tables_text"]  = m.get("tables_text", "")[:4_000]
        m["image_summaries"] = m.get("image_summaries", "")[:8_000]
        m["text_preview"] = m.get("text_preview", "")[:300]

        upsert_data.append({
            "id"      : m["chunk_id"],   # deterministic — overwrites on re-ingest
            "values"  : vec,
            "metadata": m,
        })

    for start in range(0, len(upsert_data), UPSERT_BATCH):
        batch = upsert_data[start : start + UPSERT_BATCH]
        index.upsert(vectors=batch, namespace=namespace)
        end = min(start + UPSERT_BATCH, len(upsert_data))
        print(f"   Upserted {end} / {len(upsert_data)}")

    stats = index.describe_index_stats()
    print(f"\n   ✅  Index stats → {stats}")


# ════════════════════════════════════════════════════════════════
#  QUERY HELPER
# ════════════════════════════════════════════════════════════════

def query_pinecone(
    query: str,
    top_k: int = 5,
    namespace: str = PINECONE_NAMESPACE,
    index_name: str = PINECONE_INDEX_NAME,
    filter: Optional[dict] = None,
) -> List[dict]:
    """
    Semantic search against Pinecone.

    Pinecone metadata filters can be passed via `filter`, e.g.:
      filter={"file_type": {"$eq": "pdf"}}
      filter={"has_table": {"$eq": True}}
      filter={"page_start": {"$gte": 5, "$lte": 10}}
      filter={"file_name": {"$eq": "report.pdf"}}

    Returns a list of result dicts with all metadata fields.
    """
    pc       = Pinecone(api_key=PINECONE_API_KEY)
    index    = pc.Index(index_name)
    query_vec = _embed_texts_with_gemini([query], task_type="RETRIEVAL_QUERY")[0]

    kwargs: dict = dict(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    if filter:
        kwargs["filter"] = filter

    results = index.query(**kwargs)

    hits = []
    for match in results.matches:
        hits.append({"id": match.id, "score": round(match.score, 4), **match.metadata})
    return hits


# ════════════════════════════════════════════════════════════════
#  MASTER PIPELINE  –  single entry point
# ════════════════════════════════════════════════════════════════

def run_pipeline(
    file_paths: List[str],
    namespace: Optional[str] = None,
    index_name: str = PINECONE_INDEX_NAME,
    course_id: str = "",
) -> None:
    """
    Ingest one or more files into Pinecone with rich metadata.

    Parameters
    ----------
    file_paths  : list of file paths (.pdf, .pptx, .ppt, .docx, .doc, .txt, .md)
    namespace   : Pinecone namespace (defaults to PINECONE_NAMESPACE from .env)
    """
    ns       = namespace or PINECONE_NAMESPACE
    all_docs : List[Document] = []
    failures: List[str] = []

    for fp in file_paths:
        print("\n" + "=" * 60)
        print(f"  FILE  →  {fp}")
        print("=" * 60)
        try:
            print("BEFORE partition_document")
            elements = partition_document(fp)
            print("AFTER partition_document")
            print("BEFORE create_chunks")
            chunks = create_chunks(elements)
            print("AFTER create_chunks")
            print("BEFORE build_documents")
            docs = build_documents(
                chunks,
                source_file=fp,
                course_id=course_id,)
            print("AFTER build_documents")
            all_docs.extend(docs)
        except Exception as exc:
            print(f"\n❌ Failed to process '{fp}': {exc}")
            failures.append(f"{fp}: {exc}")
        finally:
            # Clean up figures/ temp dir created by unstructured hi_res strategy.
            # Must run AFTER build_documents() has read all image bytes.
            if fp.lower().endswith(".pdf"):
                _cleanup_figures_dir()
            continue

    if not all_docs:
        print("\n⚠  No documents to upsert. Exiting.")
        details = "; ".join(failures) if failures else "No chunks were produced."
        raise RuntimeError(f"Document ingestion produced no chunks. {details}")

    store_in_pinecone(all_docs, namespace=ns, index_name=index_name)

    print("\n🎉  Pipeline complete!")
    print(f"   Vectors upserted : {len(all_docs)}")
    print(f"   Pinecone index   : {index_name}")
    print(f"   Namespace        : {ns}")

    return {
        "vectors_upserted": len(all_docs),
        "pinecone_index": index_name,
        "namespace": ns,
        "failed_files": failures,
    }


# ════════════════════════════════════════════════════════════════
#  EXAMPLE USAGE
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 1. Ingest files ──────────────────────────────────────────
    FILES_TO_INGEST = [
        # "./docs/report.pdf",
        # "./docs/deck.pptx",
        # "./docs/manual.docx",
        # "./docs/notes.txt",
    ]

    if FILES_TO_INGEST:
        run_pipeline(file_paths=FILES_TO_INGEST)
    else:
        print("⚠  No files listed in FILES_TO_INGEST. Edit the script and re-run.")

    # ── 2. Query with optional metadata filters ──────────────────
    # results = query_pinecone(
    #     query="What are the main conclusions?",
    #     top_k=5,
    #     filter={"file_type": {"$eq": "pdf"}},   # optional Pinecone filter
    # )
    # for r in results:
    #     print(f"\n[{r['score']}]  {r['file_name']}  p{r['page_start']}  §{r['section_title']}")
    #     print(f"  keywords : {r['keywords']}")
    #     print(f"  preview  : {r['text_preview'][:200]}")
