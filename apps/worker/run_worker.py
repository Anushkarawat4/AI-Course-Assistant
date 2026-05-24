from __future__ import annotations

import argparse

from apps.worker.core.config import load_environment
from apps.worker.services.document_chunking.worker import run_worker as run_document_chunking_worker
from apps.worker.services.image_chunking.worker import run_worker as run_image_chunking_worker


SERVICE_WORKERS = {
    "document-chunking": run_document_chunking_worker,
    "image-chunking": run_image_chunking_worker,
}


def main() -> None:
    load_environment("worker")

    parser = argparse.ArgumentParser(description="Run an RQ worker for a service.")
    parser.add_argument(
        "service",
        nargs="?",
        default="image-chunking",
        choices=sorted(SERVICE_WORKERS),
        help="Service queue to consume.",
    )
    args = parser.parse_args()
    SERVICE_WORKERS[args.service]()


if __name__ == "__main__":
    main()
