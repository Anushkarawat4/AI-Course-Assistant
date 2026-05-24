from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
API_ENV_PATH = PROJECT_ROOT / "apps" / "api" / ".env"
WORKER_ENV_PATH = PROJECT_ROOT / "apps" / "worker" / ".env"


def load_environment(service: str | None = None) -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    if service == "api":
        load_dotenv(API_ENV_PATH, override=False)
    elif service == "worker":
        load_dotenv(WORKER_ENV_PATH, override=False)
    else:
        load_dotenv(API_ENV_PATH, override=False)
        load_dotenv(WORKER_ENV_PATH, override=False)
