from __future__ import annotations

import os
from dataclasses import dataclass

from apps.worker.core.config import load_environment


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str
    index_name: str
    cloud: str
    region: str
    namespace: str


def get_pinecone_settings() -> PineconeSettings:
    load_environment()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured.")

    return PineconeSettings(
        api_key=api_key,
        index_name=os.getenv("PINECONE_INDEX_NAME", "rag-index"),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
        namespace=os.getenv("PINECONE_NAMESPACE", "documents"),
    )


def get_pinecone_client() -> Pinecone:
    from pinecone import Pinecone

    settings = get_pinecone_settings()
    return Pinecone(api_key=settings.api_key)


def ensure_index(
    index_name: str | None = None,
    dimension: int = 1536,
    metric: str = "cosine",
) -> None:
    from pinecone import ServerlessSpec

    settings = get_pinecone_settings()
    pc = get_pinecone_client()
    name = index_name or settings.index_name
    existing = [idx.name for idx in pc.list_indexes()]

    if name in existing:
        return

    pc.create_index(
        name=name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
    )


def get_index(index_name: str | None = None):
    settings = get_pinecone_settings()
    ensure_index(index_name=index_name)
    return get_pinecone_client().Index(index_name or settings.index_name)
