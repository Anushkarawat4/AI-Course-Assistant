# AI Course Assistant

Backend services for OCR, image chunking, document chunking, QnA chunking, audio transcription, YouTube transcript ingestion, handwritten-note ingestion, Gemini embeddings, Pinecone storage, and Redis/RQ background jobs.

The project has two main runtime parts:

- **FastAPI API**: accepts requests, validates file paths, creates RQ jobs, and exposes job/result routes.
- **Background workers**: consume Redis/RQ queues and run the actual processing pipelines.

## Project Structure

```text
apps/
  api/
    main.py
    routes/
      image_chunking_jobs.py
      document_chunking_jobs.py
      qna_chunking_jobs.py
      audio_chunking_jobs.py
      youtube_chunking_jobs.py
      handwritten_chunking_jobs.py
      temp_chunk_preview.py
    pipelines/
      chunking pipeline/
        image pipeline/
        document pipeline/
        QnA pipeline/
        audio pipeline/
        youtube pipeline/
        handwritten pipeline/

  worker/
    run_worker.py
    core/
      config.py
      redis.py
      rq.py
      pinecone.py
    services/
      image_chunking/
      document_chunking/
      qna_chunking/
      audio_chunking/
      youtube_chunking/
      handwritten_chunking/
```

## What Each Pipeline Does

**Image chunking**

Accepts an image or scanned PDF path, runs preprocessing + Tesseract OCR, falls back to Gemini OCR when confidence is low, creates efficient LLM-ready text chunks, enriches each chunk with Gemini metadata, embeds with Gemini, and stores vectors in Pinecone.

Supported files:

```text
.pdf, .png, .jpg, .jpeg, .tif, .tiff, .bmp, .webp
```

**Document chunking**

Accepts PDF/DOC/PPT/TXT/MD paths, partitions the document, creates semantic chunks, creates Gemini embeddings, and stores vectors in Pinecone.

Supported files:

```text
.pdf, .pptx, .ppt, .docx, .doc, .txt, .md
```

**Audio chunking**

Accepts an audio path, generates a structured Gemini transcript, creates sliding-window, concept-block, and lecture-summary chunks, embeds them with Gemini, and stores them in Pinecone.

Supported files:

```text
.aac, .flac, .m4a, .mp3, .mp4, .mpeg, .mpga, .ogg, .opus, .wav, .webm
```

**QnA chunking**

Accepts question-answer style PDFs, creates retrieval chunks for questions and answers, embeds with Gemini, and stores them in Pinecone.

**YouTube chunking**

Accepts a YouTube URL, pulls video metadata and transcript, chunks transcript text without LLM enhancement, embeds with Gemini, and stores chunks in Pinecone.

**Handwritten chunking**

Accepts handwritten images or scanned PDFs, uses Gemini structured OCR with a prompt file, chunks the extracted text, embeds with Gemini, and stores chunks in Pinecone.

## Requirements

Install these before running the full pipeline:

- Python 3.11 or newer
- Redis URL, local Redis or Upstash Redis
- Pinecone API key
- Gemini API key
- Tesseract OCR installed locally for image OCR

On Windows, install Tesseract from:

```text
https://github.com/UB-Mannheim/tesseract/wiki
```

After installing, make sure `tesseract.exe` is available on your system `PATH`.

## Environment Files

Create these files from the examples:

```powershell
Copy-Item apps\api\.env.example apps\api\.env
Copy-Item apps\worker\.env.example apps\worker\.env
```

Both API and worker should have the same Redis URL so they talk to the same queue.

### API `.env`

Path:

```text
apps/api/.env
```

Important values:

```env
REDIS_URL=redis://localhost:6379/0

PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_CLOUD=aws
PINECONE_COURSE_INDEX_PREFIX=course

GEMINI_API_KEY=your_gemini_key
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIM=1536
GEMINI_IMAGE_SUMMARY_MODEL=gemini-2.5-flash-lite
GEMINI_TRANSCRIPTION_MODEL=gemini-2.5-flash
HANDWRITTEN_FLASH_MODEL=gemini-2.5-flash
HANDWRITTEN_PRO_MODEL=gemini-2.5-pro
IMAGE_GEMINI_OCR_MODEL=gemini-2.5-flash
IMAGE_GEMINI_TEXT_MODEL=gemini-2.5-flash
IMAGE_GEMINI_ENRICH_MODEL=gemini-2.5-flash
IMAGE_LOW_CONFIDENCE_THRESHOLD=60
ENABLE_IMAGE_SUMMARIES=true

DOCUMENT_PDF_STRATEGY=hi_res
IMAGE_CHUNKING_NAMESPACE=image-chunks
DOCUMENT_CHUNKING_NAMESPACE=document-chunks
QNA_CHUNKING_NAMESPACE=qna-chunks
AUDIO_CHUNKING_NAMESPACE=audio-chunks
YOUTUBE_CHUNKING_NAMESPACE=youtube-chunks
HANDWRITTEN_CHUNKING_NAMESPACE=handwritten-chunks
AUDIO_CHUNKING_QUEUE=audio-chunking
```

### Worker `.env`

Path:

```text
apps/worker/.env
```

Important values:

```env
REDIS_URL=redis://localhost:6379/0

PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_CLOUD=aws
PINECONE_COURSE_INDEX_PREFIX=course

GEMINI_API_KEY=your_gemini_key
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIM=1536
GEMINI_IMAGE_SUMMARY_MODEL=gemini-2.5-flash-lite
GEMINI_TRANSCRIPTION_MODEL=gemini-2.5-flash
HANDWRITTEN_FLASH_MODEL=gemini-2.5-flash
HANDWRITTEN_PRO_MODEL=gemini-2.5-pro
IMAGE_GEMINI_OCR_MODEL=gemini-2.5-flash
IMAGE_GEMINI_TEXT_MODEL=gemini-2.5-flash
IMAGE_GEMINI_ENRICH_MODEL=gemini-2.5-flash
IMAGE_LOW_CONFIDENCE_THRESHOLD=60
ENABLE_IMAGE_SUMMARIES=true

DOCUMENT_PDF_STRATEGY=hi_res

IMAGE_CHUNKING_NAMESPACE=image-chunks
IMAGE_CHUNKING_QUEUE=image-chunking
IMAGE_CHUNKING_JOB_TIMEOUT=1800
IMAGE_CHUNKING_RESULT_TTL=86400
IMAGE_CHUNKING_FAILURE_TTL=604800

DOCUMENT_CHUNKING_QUEUE=document-chunking
DOCUMENT_CHUNKING_NAMESPACE=document-chunks
DOCUMENT_CHUNKING_JOB_TIMEOUT=3600
DOCUMENT_CHUNKING_RESULT_TTL=86400
DOCUMENT_CHUNKING_FAILURE_TTL=604800

AUDIO_CHUNKING_NAMESPACE=audio-chunks
AUDIO_CHUNKING_QUEUE=audio-chunking
AUDIO_CHUNKING_JOB_TIMEOUT=7200
AUDIO_CHUNKING_RESULT_TTL=86400
AUDIO_CHUNKING_FAILURE_TTL=604800

YOUTUBE_CHUNKING_NAMESPACE=youtube-chunks
YOUTUBE_CHUNKING_QUEUE=youtube-chunking
YOUTUBE_CHUNKING_JOB_TIMEOUT=3600
YOUTUBE_CHUNKING_RESULT_TTL=86400
YOUTUBE_CHUNKING_FAILURE_TTL=604800

HANDWRITTEN_CHUNKING_NAMESPACE=handwritten-chunks
HANDWRITTEN_CHUNKING_QUEUE=handwritten-chunking
HANDWRITTEN_CHUNKING_JOB_TIMEOUT=7200
HANDWRITTEN_CHUNKING_RESULT_TTL=86400
HANDWRITTEN_CHUNKING_FAILURE_TTL=604800
```

For Upstash, put the Upstash Redis connection string in `REDIS_URL`.

Each course stores vectors in its own Pinecone index. The index name is generated from `course_id`, for example:

```text
course-cs301-a1b2c3d4
```

Inside that index, fixed namespaces separate the ingestion types:

```text
document-chunks
image-chunks
qna-chunks
audio-chunks
youtube-chunks
handwritten-chunks
```

## Image Pipeline Flow

The image pipeline now keeps OCR and metadata extraction separate:

1. Preprocess the image or scanned PDF page.
2. Run Tesseract OCR first.
3. If OCR confidence is below `IMAGE_LOW_CONFIDENCE_THRESHOLD`, send that page image to Gemini OCR.
4. If OCR confidence is good, send the Tesseract text to Gemini for cleanup, summary, topic, keywords, and page metadata.
5. Build clean, section-aware text chunks from the Gemini-normalized page results.
6. Send each chunk to Gemini for summary, topic, keywords, and content signals.
7. Embed the chunk text with Gemini and upsert into the course Pinecone index under `image-chunks`.

Because Gemini extracts the semantic metadata, the local image chunker no longer tries to detect code, formulas, definitions, examples, or content type by regex.

## Setup

From the project root:

```powershell
cd "C:\Users\Dell\OneDrive\Desktop\PROJECTS\AI Course Assistant"
```

Create and activate the API virtual environment:

```powershell
cd apps\api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you also want to install worker dependencies into the same venv:

```powershell
pip install -r ..\worker\requirements.txt
```

This project currently uses the API venv to run both the API and workers.

## Run The API

Open a terminal:

```powershell
cd "C:\Users\Dell\OneDrive\Desktop\PROJECTS\AI Course Assistant\apps\api"
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

For development reload, do not let Uvicorn watch `.venv`. Because the virtualenv is inside `apps/api`, watching it can trigger endless reloads from installed packages:

```powershell
uvicorn main:app --reload --reload-dir . --reload-exclude ".venv/*" --host 127.0.0.1 --port 8000
```

## Run Background Workers

Open a new terminal from the project root.

### Image Chunking Worker

```powershell
cd "C:\Users\Dell\OneDrive\Desktop\PROJECTS\AI Course Assistant"
.\apps\api\.venv\Scripts\python.exe -m apps.worker.run_worker image-chunking
```

### Document Chunking Worker

```powershell
cd "C:\Users\Dell\OneDrive\Desktop\PROJECTS\AI Course Assistant"
.\apps\api\.venv\Scripts\python.exe -m apps.worker.run_worker document-chunking
```

### QnA Chunking Worker

```powershell
cd "C:\Users\Dell\OneDrive\Desktop\PROJECTS\AI Course Assistant"
.\apps\api\.venv\Scripts\python.exe -m apps.worker.run_worker qna-chunking
```

### Audio Chunking Worker

```powershell
cd "C:\Users\Dell\OneDrive\Desktop\PROJECTS\AI Course Assistant"
.\apps\api\.venv\Scripts\python.exe -m apps.worker.run_worker audio-chunking
```

### YouTube Chunking Worker

```powershell
cd "C:\Users\Dell\OneDrive\Desktop\PROJECTS\AI Course Assistant"
.\apps\api\.venv\Scripts\python.exe -m apps.worker.run_worker youtube-chunking
```

### Handwritten Chunking Worker

```powershell
cd "C:\Users\Dell\OneDrive\Desktop\PROJECTS\AI Course Assistant"
.\apps\api\.venv\Scripts\python.exe -m apps.worker.run_worker handwritten-chunking
```

Run each worker in a separate terminal when you want multiple queues active.

## API Routes

### Image Chunking Jobs

Create job:

```http
POST /image-chunking/jobs
```

Body:

```json
{
  "file_path": "C:\\Users\\Dell\\Downloads\\PublicWaterMassMailing.pdf",
  "dpi": 300,
  "course_id": "CS301",
  "course_name": "",
  "subject_area": ""
}
```

List jobs:

```http
GET /image-chunking/jobs
```

Get job:

```http
GET /image-chunking/jobs/{job_id}
```

Get result:

```http
GET /image-chunking/jobs/{job_id}/result
```

Delete job:

```http
DELETE /image-chunking/jobs/{job_id}
```

### Document Chunking Jobs

Create job:

```http
POST /document-chunking/jobs
```

Body:

```json
{
  "file_paths": [
    "C:\\Users\\Dell\\Downloads\\course-notes.pdf"
  ],
  "course_id": "CS301"
}
```

List jobs:

```http
GET /document-chunking/jobs
```

Get job:

```http
GET /document-chunking/jobs/{job_id}
```

Get result:

```http
GET /document-chunking/jobs/{job_id}/result
```

Delete job:

```http
DELETE /document-chunking/jobs/{job_id}
```

### QnA Chunking Jobs

Create job:

```http
POST /qna-chunking/jobs
```

Body:

```json
{
  "file_paths": [
    "C:\\Users\\Dell\\Downloads\\question-bank.pdf"
  ],
  "course_id": "CS301"
}
```

Get result:

```http
GET /qna-chunking/jobs/{job_id}/result
```

### Audio Chunking Jobs

Create job:

```http
POST /audio-chunking/jobs
```

Body:

```json
{
  "file_path": "C:\\Users\\Dell\\Downloads\\lecture-01.mp3",
  "course_id": "CS301",
  "course_name": "",
  "lecture_id": "",
  "lecture_number": 0,
  "week_number": 0,
  "lecture_title": "",
  "professor": ""
}
```

When `lecture_id` is empty, the audio file name is used. Fetch the result with:

```http
GET /audio-chunking/jobs/{job_id}/result
```

### YouTube Chunking Jobs

Create job:

```http
POST /youtube-chunking/jobs
```

Body:

```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "course_id": "CS301",
  "course_name": "",
  "subject": ""
}
```

Fetch the result with:

```http
GET /youtube-chunking/jobs/{job_id}/result
```

### Agentic Retrieval

The retrieval pipeline searches the course-specific Pinecone index generated from `course_id`.
It uses fixed namespaces for each source type, such as `document-chunks`, `image-chunks`, `qna-chunks`, `audio-chunks`, `youtube-chunks`, and `handwritten-chunks`.

Ask mode:

```http
POST /retrieval/ask
```

```json
{
  "query": "Explain TCP three-way handshake",
  "course_id": "CS301",
  "top_k": 8
}
```

Quiz mode:

```http
POST /retrieval/quiz
```

Summarize mode:

```http
POST /retrieval/summarize
```

Gemini API mode note:

The retrieval planner currently uses Gemini JSON mode and validates the JSON locally with Pydantic. This is the correct setting for the regular Gemini Developer API because that API does not support every JSON Schema feature generated by Pydantic, especially `additionalProperties` from `dict` fields.

Final response model defaults:

- `GEMINI_ASK_RESPONSE_MODEL=gemini-2.5-flash` for normal question answering, which keeps latency and cost moderate.
- `GEMINI_ASK_FALLBACK_MODEL=gemini-2.5-flash-lite` if the ask model is temporarily overloaded.
- `GEMINI_QUIZ_RESPONSE_MODEL=gemini-2.5-pro` for higher-quality quiz generation.
- `GEMINI_QUIZ_FALLBACK_MODEL=gemini-2.5-flash` if Pro quota is exhausted or unavailable.
- `GEMINI_SUMMARY_RESPONSE_MODEL=gemini-2.5-pro` for higher-quality summaries.
- `GEMINI_SUMMARY_FALLBACK_MODEL=gemini-2.5-flash` if Pro quota is exhausted or unavailable.

If you are using the free Gemini Developer API tier and `gemini-2.5-pro` returns quota errors, either switch the primary quiz/summary models to `gemini-2.5-flash` in `apps/api/.env` or enable billing/Enterprise quota. The route will automatically try the fallback model for temporary `429` and `503` errors.

You can change these values in `apps/api/.env` for local/API runs and in `apps/worker/.env` if a worker-side retrieval flow is added later.

For Gemini Enterprise Agent Platform deployment, you can switch back to server-side schema enforcement by uncommenting these lines:

- [agentic_retrieval.py](</c:/Users/Dell/OneDrive/Desktop/PROJECTS/AI Course Assistant/apps/api/pipelines/retrieval pipeline/agentic_retrieval.py:764>) uncomment `response_schema=QueryPlan`

Keep those lines commented for normal API-key deployment on local Windows or Render.

### Handwritten Chunking Jobs

Create job:

```http
POST /handwritten-chunking/jobs
```

Body:

```json
{
  "file_path": "C:\\Users\\Dell\\Downloads\\handwritten-notes.pdf",
  "course_id": "CS301",
  "course_name": "",
  "subject": "",
  "instructor": "",
  "semester": "",
  "university": "",
  "tags": []
}
```

Fetch the result with:

```http
GET /handwritten-chunking/jobs/{job_id}/result
```

## Temporary Chunk Preview Routes

These routes are only for debugging and learning the chunk output before using the background queue.

File:

```text
apps/api/routes/temp_chunk_preview.py
```

Delete this file later when the preview routes are no longer needed. Also remove the `temp_chunk_preview_router` import and `app.include_router(temp_chunk_preview_router)` from `apps/api/main.py`.

### Preview Document Chunks

```http
POST /temp/chunks/document
```

Body:

```json
{
  "file_path": "C:\\Users\\Dell\\Downloads\\course-notes.pdf",
  "include_metadata": true
}
```

This returns chunks without creating embeddings and without storing anything in Pinecone.

### Preview Image OCR Chunks

```http
POST /temp/chunks/image
```

Body:

```json
{
  "file_path": "C:\\Users\\Dell\\Downloads\\scan.pdf",
  "dpi": 300,
  "include_pages": true
}
```

This runs OCR directly and returns OCR blocks as temporary chunk-like records.

## Windows JSON Path Note

In JSON, Windows paths must use escaped backslashes:

```json
{
  "file_path": "C:\\Users\\Dell\\Downloads\\file.pdf"
}
```

Or use forward slashes:

```json
{
  "file_path": "C:/Users/Dell/Downloads/file.pdf"
}
```

Do not send this:

```json
{
  "file_path": "C:\Users\Dell\Downloads\file.pdf"
}
```

That causes `Invalid \escape`.

## Local Development Flow

Start Redis or configure `REDIS_URL` with Upstash.

Start API:

```powershell
cd "C:\Users\Dell\OneDrive\Desktop\PROJECTS\AI Course Assistant\apps\api"
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 8000
```

Start worker:

```powershell
cd "C:\Users\Dell\OneDrive\Desktop\PROJECTS\AI Course Assistant"
.\apps\api\.venv\Scripts\python.exe -m apps.worker.run_worker image-chunking
```

Create a job from Swagger or Postman.

Watch the worker terminal process the job.

Fetch the result from:

```text
GET /image-chunking/jobs/{job_id}/result
```

## Deployment Notes

For Render or another hosting platform:

- Deploy the FastAPI API as a web service.
- Deploy each worker as a separate background worker service.
- Set the same `REDIS_URL` in API and worker environments.
- Set `PINECONE_API_KEY` and `GEMINI_API_KEY` in worker environment.
- Keep queue names the same between API and worker.

Example worker start commands:

```bash
python -m apps.worker.run_worker image-chunking
python -m apps.worker.run_worker document-chunking
python -m apps.worker.run_worker qna-chunking
python -m apps.worker.run_worker audio-chunking
python -m apps.worker.run_worker youtube-chunking
python -m apps.worker.run_worker handwritten-chunking
```

If using Upstash Redis, paste the Upstash Redis URL into `REDIS_URL` for both services.

## Troubleshooting

**422 Invalid escape**

Use escaped backslashes or forward slashes in JSON file paths.

**Worker error: `os.fork` or `SIGALRM` on Windows**

The worker code uses Windows-compatible RQ worker behavior. Run workers through:

```powershell
.\apps\api\.venv\Scripts\python.exe -m apps.worker.run_worker image-chunking
```

**Tesseract not found**

Install Tesseract OCR and add it to `PATH`.

**Redis connection failed**

Check `REDIS_URL` in both `apps/api/.env` and `apps/worker/.env`.

**Document PDF job says `No module named 'unstructured_inference'`**

The high-quality Unstructured PDF strategy uses extra ML layout dependencies. Install them in the same venv used by the worker:

```powershell
.\apps\api\.venv\Scripts\python.exe -m pip install "unstructured-inference"
```

Keep:

```env
DOCUMENT_PDF_STRATEGY=hi_res
```

**Uvicorn keeps reloading files from `.venv\Lib\site-packages`**

This happens because `.venv` is inside `apps/api`, and `uvicorn --reload` watches everything under the current folder. Use the stable command without reload:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

Or use reload with `.venv` excluded:

```powershell
uvicorn main:app --reload --reload-dir . --reload-exclude ".venv/*" --host 127.0.0.1 --port 8000
```

**Pinecone or Gemini error**

Check:

```env
PINECONE_API_KEY=
PINECONE_COURSE_INDEX_PREFIX=course
GEMINI_API_KEY=
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIM=1536
```

**Pinecone created `image-chunks` as an index**

That index was created by an older worker or queued job. In the current code, `image-chunks` is only a namespace, and the index is generated from `course_id`, for example `course-cs301-68e5fce5`.

Restart both the API and worker after pulling changes:

```powershell
.\apps\api\.venv\Scripts\python.exe -m apps.worker.run_worker image-chunking
```

Then create a new job with `course_id` and without `namespace`. After confirming the new course index has the expected namespace, delete the old `image-chunks` index from the Pinecone dashboard if it only contains test data.
