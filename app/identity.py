"""Who a run belongs to.

Kept in its own module because both the graph and the tools need it: a tool that writes to
the store has to know whose store to write to, and importing the graph from a tool would be
circular.
"""

import os
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime

# Who memory is filed under when a run does not name a user. Studio does not send a
# user_id unless you set one, so overriding this is the quickest way to get a clean slate
# for testing: DEFAULT_USER_ID=test-2 uv run langgraph dev
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "anonymous")


# Where a caller's Cognito ID token travels. The `__` prefix is load-bearing, not a naming
# whim: langchain-core drops `configurable` keys starting with `__` from the metadata it
# sends to LangSmith (runnables/config.py), and LangGraph drops them from checkpoint metadata
# (checkpoint/base). A bearer token in a plainly-named key would be written into every trace
# and persisted with the thread.
ID_TOKEN_KEY = "__itl_id_token"


class AssistantContext(TypedDict, total=False):
    """Per-run settings. Declaring it makes `user_id` an editable field in Studio."""

    user_id: str
    # Set this in Studio to exercise the artist-only backend tools by hand. The API layer
    # uses ID_TOKEN_KEY instead, which is the channel that stays out of traces.
    id_token: str


def user_id_from_config(config: RunnableConfig) -> str:
    """Whose memory this run reads and writes.

    Checked on both channels because callers reach the graph differently: the FastAPI app
    passes `configurable`, while Studio and anything using a context schema pass `context`.
    """
    from_config = (config.get("configurable") or {}).get("user_id")
    if from_config:
        return from_config

    try:
        context = get_runtime(AssistantContext).context
    except Exception:  # no runtime outside a graph execution
        context = None
    if context:
        from_context = (
            context.get("user_id") if isinstance(context, dict)
            else getattr(context, "user_id", None)
        )
        if from_context:
            return from_context

    return DEFAULT_USER_ID


def id_token_from_config(config: RunnableConfig) -> str | None:
    """The caller's Cognito ID token for backend calls, or None if the run is anonymous.

    Read per call rather than held anywhere: these tokens expire in about an hour, so the UI
    refreshes them and posts the current one with each run. Checked on the same two channels
    as `user_id_from_config`, then falling back to `ITL_ID_TOKEN` in the environment so a
    local script or a Studio session can sign requests without an API layer to inject one.
    """
    from_config = (config.get("configurable") or {}).get(ID_TOKEN_KEY)
    if from_config:
        return from_config

    try:
        context = get_runtime(AssistantContext).context
    except Exception:  # no runtime outside a graph execution
        context = None
    if context:
        from_context = (
            context.get("id_token") if isinstance(context, dict)
            else getattr(context, "id_token", None)
        )
        if from_context:
            return from_context

    return os.getenv("ITL_ID_TOKEN") or None
