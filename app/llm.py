"""The chat model, shared by the agent and the trustcall extractors.

Importing this module reads XAI_API_KEY, so callers must have loaded .env first.
`model` stays None without a key, which keeps the app importable (and the FastAPI docs
browsable) on a machine that has not been configured yet — the graph raises only when a
run is actually attempted.
"""

import os

from langchain_xai import ChatXAI
from pydantic import SecretStr

xai_api_key = os.getenv("XAI_API_KEY", "")
xai_model = os.getenv("XAI_MODEL", "grok-4")

model: ChatXAI | None = (
    ChatXAI(api_key=SecretStr(xai_api_key), model=xai_model, temperature=0.1)
    if xai_api_key
    else None
)


def require_model() -> ChatXAI:
    if model is None:
        raise RuntimeError(
            "XAI_API_KEY is not set. Please set the environment variable before running the graph."
        )
    return model
