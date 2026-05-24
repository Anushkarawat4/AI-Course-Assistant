from __future__ import annotations

from typing import Any

from apps.worker.services.document_chunking.run_pipeline import run_pipeline


def process_document_chunking_request(
    file_paths: list[str],
    namespace: str | None = None,
    **_legacy_kwargs,
) -> dict[str, Any]:
    return run_pipeline(
        file_paths=file_paths,
        namespace=namespace,
    )
