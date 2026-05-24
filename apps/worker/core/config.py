from __future__ import annotations

import os
from pathlib import Path
from io import StringIO

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[3]
API_ENV_PATH = PROJECT_ROOT / "apps" / "api" / ".env"
WORKER_ENV_PATH = PROJECT_ROOT / "apps" / "worker" / ".env"


def load_environment(service: str | None = None) -> None:
    load_env_file(PROJECT_ROOT / ".env")

    if service == "api":
        load_env_file(API_ENV_PATH)
    elif service == "worker":
        load_env_file(WORKER_ENV_PATH)
    else:
        load_env_file(API_ENV_PATH)
        load_env_file(WORKER_ENV_PATH)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    sanitized_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("//"):
            continue
        sanitized_lines.append(line)

    values = dotenv_values(stream=StringIO("\n".join(sanitized_lines)))
    for key, value in values.items():
        if key and value is not None and key not in os.environ:
            os.environ[key] = value
