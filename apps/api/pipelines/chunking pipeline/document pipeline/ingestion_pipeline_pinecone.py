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

import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


# ── LangChain ────────────────────────────────────────────────────
from langchain_core.documents import Document

# ── Gemini embeddings ────────────────────────────────────────────
from google import genai
from google.genai import types

# ── Pinecone v3+ ─────────────────────────────────────────────────
from pinecone import Pinecone, ServerlessSpec

# ── Unstructured (universal parser) ──────────────────────────────
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title


# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
PINECONE_ENV        = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "documents")

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM   = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))

# PDF parsing strategy. hi_res is the default because it preserves richer
# layout/table/image structure for downstream retrieval quality.
PDF_PARTITION_STRATEGY = os.getenv("DOCUMENT_PDF_STRATEGY", "hi_res").lower()

# Chunking knobs
CHUNK_MAX_CHARS = 3000
CHUNK_NEW_AFTER = 2400
CHUNK_MIN_CHARS = 500

# Keyword extraction
MAX_KEYWORDS    = 12    # top N keywords stored per chunk
MIN_KEYWORD_LEN = 4     # ignore tokens shorter than this

# Pinecone upsert batch size
UPSERT_BATCH    = 100


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
        kwargs.update(
            strategy=PDF_PARTITION_STRATEGY,
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
        )
    elif ext in {".pptx", ".ppt"}:
        kwargs.update(include_page_breaks=True)
    elif ext in {".docx", ".doc"}:
        kwargs.update(infer_table_structure=True)
    # .txt / .md → no extra kwargs needed

    elements = partition(**kwargs)
    print(f"   ✅  {len(elements)} elements extracted")
    return elements


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
    )
    print(f"   ✅  {len(chunks)} chunks created")
    return chunks


# ════════════════════════════════════════════════════════════════
#  STEP 3 – BUILD RICH DOCUMENTS  (no AI, zero cost)
# ════════════════════════════════════════════════════════════════

def build_documents(chunks: list, source_file: str) -> List[Document]:
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

            # Images – count only (skip base64 to stay under Pinecone 40 KB limit)
            elif kind == "Image":
                image_count += 1

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
        embedding_text = "\n\n".join(p for p in parts if p.strip())

        # ── 3g. Metadata dict ────────────────────────────────────
        metadata = {
            # Identity
            "chunk_id"        : _chunk_id(source_file, i),
            "source"          : str(path),
            "file_name"       : path.name,
            "file_type"       : file_ext,

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


def _embed_texts_with_gemini(
    texts: List[str],
    task_type: str,
    batch_size: int = 100,
) -> List[list]:
    client = _get_gemini_client()
    vectors: List[list] = []

    for text in texts:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text if text.strip() else " ",
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        vectors.append(result.embeddings[0].values)

    return vectors

def _get_or_create_index(pc: Pinecone) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"\n🌲  Creating Pinecone index '{PINECONE_INDEX_NAME}' …")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_ENV),
        )
        print("   ✅  Index created")
    else:
        print(f"\n🌲  Using existing index '{PINECONE_INDEX_NAME}'")


def store_in_pinecone(
    documents: List[Document],
    namespace: str = PINECONE_NAMESPACE,
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
    _get_or_create_index(pc)
    index = pc.Index(PINECONE_INDEX_NAME)

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
    index    = pc.Index(PINECONE_INDEX_NAME)
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
            elements = partition_document(fp)
            chunks   = create_chunks(elements)
            docs     = build_documents(chunks, source_file=fp)
            all_docs.extend(docs)
        except Exception as exc:
            print(f"\n  ❌  Failed to process '{fp}': {exc}")
            failures.append(f"{fp}: {exc}")
            continue

    if not all_docs:
        print("\n⚠  No documents to upsert. Exiting.")
        details = "; ".join(failures) if failures else "No chunks were produced."
        raise RuntimeError(f"Document ingestion produced no chunks. {details}")

    store_in_pinecone(all_docs, namespace=ns)

    print("\n🎉  Pipeline complete!")
    print(f"   Vectors upserted : {len(all_docs)}")
    print(f"   Pinecone index   : {PINECONE_INDEX_NAME}")
    print(f"   Namespace        : {ns}")

    return {
        "vectors_upserted": len(all_docs),
        "pinecone_index": PINECONE_INDEX_NAME,
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
