# AI Course Assistant

Backend services for OCR, image chunking, document chunking, Gemini embeddings, Pinecone storage, and Redis/RQ background jobs.

The project has two main runtime parts:

- **FastAPI API**: accepts requests, validates file paths, creates RQ jobs, and exposes job/result routes.
- **Background workers**: consume Redis/RQ queues and run the actual image/document processing pipelines.

## Project Structure

```text
apps/
  api/
    main.py
    routes/
      image_chunking_jobs.py
      document_chunking_jobs.py
      temp_chunk_preview.py
    pipelines/
      chunking pipeline/
        image pipeline/
        document pipeline/

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
```

## What Each Pipeline Does

**Image chunking**

Accepts an image or scanned PDF path, runs preprocessing + Tesseract OCR, and returns page-wise OCR text with block metadata.

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
PINECONE_INDEX_NAME=rag-index
PINECONE_ENVIRONMENT=us-east-1
PINECONE_CLOUD=aws
PINECONE_NAMESPACE=documents

GEMINI_API_KEY=your_gemini_key
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIM=1536

DOCUMENT_PDF_STRATEGY=hi_res
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
PINECONE_INDEX_NAME=rag-index
PINECONE_ENVIRONMENT=us-east-1
PINECONE_CLOUD=aws
PINECONE_NAMESPACE=documents

GEMINI_API_KEY=your_gemini_key
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIM=1536

DOCUMENT_PDF_STRATEGY=hi_res

IMAGE_CHUNKING_QUEUE=image-chunking
IMAGE_CHUNKING_JOB_TIMEOUT=1800
IMAGE_CHUNKING_RESULT_TTL=86400
IMAGE_CHUNKING_FAILURE_TTL=604800

DOCUMENT_CHUNKING_QUEUE=document-chunking
DOCUMENT_CHUNKING_JOB_TIMEOUT=3600
DOCUMENT_CHUNKING_RESULT_TTL=86400
DOCUMENT_CHUNKING_FAILURE_TTL=604800
```

For Upstash, put the Upstash Redis connection string in `REDIS_URL`.

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

Run each worker in a separate terminal when you want both queues active.

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
  "dpi": 300
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
  "namespace": "documents"
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
PINECONE_INDEX_NAME=
GEMINI_API_KEY=
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIM=1536
```
