"""Environment loading, kept free of langchain imports.

This module must be importable — and callable — before anything imports
langchain/langgraph, because the LangSmith SDK caches some env-var reads on first
access. Entry points call load_env_file() as their first statement.
"""

import os
from pathlib import Path


def load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
