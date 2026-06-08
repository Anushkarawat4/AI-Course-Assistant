from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RETRIEVAL_PIPELINE_FILE = (
    PROJECT_ROOT
    / "apps"
    / "api"
    / "pipelines"
    / "retrieval pipeline"
    / "agentic_retrieval.py"
)
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_DEFAULT_TOP_K", "8"))
ASK_RESPONSE_MODEL = os.getenv(
    "GEMINI_ASK_RESPONSE_MODEL",
    os.getenv("GEMINI_FINAL_RESPONSE_MODEL", "gemini-2.5-flash"),
)
QUIZ_RESPONSE_MODEL = os.getenv(
    "GEMINI_QUIZ_RESPONSE_MODEL",
    os.getenv("GEMINI_FINAL_RESPONSE_MODEL", "gemini-2.5-pro"),
)
SUMMARY_RESPONSE_MODEL = os.getenv(
    "GEMINI_SUMMARY_RESPONSE_MODEL",
    os.getenv("GEMINI_FINAL_RESPONSE_MODEL", "gemini-2.5-pro"),
)
ASK_FALLBACK_MODEL = os.getenv("GEMINI_ASK_FALLBACK_MODEL", "gemini-2.5-flash-lite")
QUIZ_FALLBACK_MODEL = os.getenv("GEMINI_QUIZ_FALLBACK_MODEL", "gemini-2.5-flash")
SUMMARY_FALLBACK_MODEL = os.getenv("GEMINI_SUMMARY_FALLBACK_MODEL", "gemini-2.5-flash")


def load_retrieval_pipeline_module():
    spec = importlib.util.spec_from_file_location("agentic_retrieval_pipeline", RETRIEVAL_PIPELINE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load retrieval pipeline from {RETRIEVAL_PIPELINE_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    course_id: str = Field(..., min_length=1, description="Course identifier used to select the Pinecone index")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=30)
    include_context: bool = True
    include_answer_prompt: bool = True
    include_chunks: bool = True


class AskRetrievalRequest(RetrievalRequest):
    pass


class QuizRetrievalRequest(RetrievalRequest):
    query: str = Field(..., min_length=1, description="Topic or learning goal for quiz generation")


class SummarizeRetrievalRequest(RetrievalRequest):
    query: str = Field(..., min_length=1, description="Topic to summarize")


router = APIRouter(prefix="/retrieval", tags=["agentic-retrieval"])


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")
    return genai.Client(api_key=api_key.strip())


def is_retryable_gemini_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return "client has been closed" in message or "temporarily unavailable" in message


def controller_instruction(mode: Literal["ask", "quiz", "summarize"]) -> str:
    if mode == "quiz":
        return (
            "Generate the quiz requested by the student using only the retrieved course context. "
            "Include questions, answer key, explanations, and source citations like [1]."
        )
    if mode == "summarize":
        return (
            "Write a clear study summary using only the retrieved course context. "
            "Use structured headings, key terms, examples where useful, and source citations like [1]."
        )
    return (
        "Answer the student's question using only the retrieved course context. "
        "Be clear, educational, and cite factual claims with source citations like [1]."
    )


def generate_final_response(
    *,
    answer_prompt: str,
    mode: Literal["ask", "quiz", "summarize"],
) -> tuple[str, str]:
    primary_model = {
        "ask": ASK_RESPONSE_MODEL,
        "quiz": QUIZ_RESPONSE_MODEL,
        "summarize": SUMMARY_RESPONSE_MODEL,
    }[mode]
    fallback_model = {
        "ask": ASK_FALLBACK_MODEL,
        "quiz": QUIZ_FALLBACK_MODEL,
        "summarize": SUMMARY_FALLBACK_MODEL,
    }[mode]
    config = types.GenerateContentConfig(
        system_instruction=controller_instruction(mode),
        temperature=0.2 if mode == "ask" else 0.35,
        max_output_tokens=int(os.getenv("RETRIEVAL_FINAL_MAX_OUTPUT_TOKENS", "4096")),
    )
    last_exc: Exception | None = None
    tried_models: set[str] = set()
    for model in [primary_model, fallback_model]:
        if not model:
            continue
        if model in tried_models:
            continue
        tried_models.add(model)
        try:
            client = get_gemini_client()
            response = client.models.generate_content(
                model=model,
                contents=answer_prompt,
                config=config,
            )
            return response.text or "", model
        except Exception as exc:
            last_exc = exc
            if not is_retryable_gemini_error(exc):
                raise
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("No Gemini response model is configured.")


def run_retrieval(request: RetrievalRequest, mode: Literal["ask", "quiz", "summarize"]) -> dict:
    stage = "load retrieval pipeline"
    try:
        pipeline = load_retrieval_pipeline_module()
        stage = "retrieve chunks"
        result = pipeline.retrieve(
            query=request.query,
            course_id=request.course_id,
            mode=pipeline.RetrievalMode(mode),
            top_k=request.top_k,
            verbose=False,
        )
        answer_prompt = result.to_answer_prompt()
        stage = "generate final response"
        final_response, response_model = (
            generate_final_response(answer_prompt=answer_prompt, mode=mode)
            if result.chunks
            else ("I could not find relevant course material for this query yet.", "")
        )
        payload = to_jsonable(result)
        if not request.include_chunks:
            payload.pop("chunks", None)
        if request.include_context:
            payload["context"] = result.to_context_string()
        if request.include_answer_prompt:
            payload["answer_prompt"] = answer_prompt
        payload["final_response"] = final_response
        payload["response_model"] = response_model
        payload["pinecone_index"] = pipeline.get_course_index_name(request.course_id)
        return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed during {stage}: {exc}") from exc


def ask_controller(request: AskRetrievalRequest) -> dict:
    return run_retrieval(request, "ask")


def quiz_controller(request: QuizRetrievalRequest) -> dict:
    return run_retrieval(request, "quiz")


def summarize_controller(request: SummarizeRetrievalRequest) -> dict:
    return run_retrieval(request, "summarize")


@router.post("/ask")
async def ask(request: AskRetrievalRequest) -> dict:
    return await run_in_threadpool(ask_controller, request)


@router.post("/quiz")
async def quiz(request: QuizRetrievalRequest) -> dict:
    return await run_in_threadpool(quiz_controller, request)


@router.post("/summarize")
async def summarize(request: SummarizeRetrievalRequest) -> dict:
    return await run_in_threadpool(summarize_controller, request)
