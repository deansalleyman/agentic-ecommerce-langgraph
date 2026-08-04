"""Who a run belongs to.

Kept in its own module because both the graph and the tools need it: a tool that writes to
the store has to know whose store to write to, and importing the graph from a tool would be
circular.
"""

import os
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import get_runtime

# Who memory is filed under when a run does not name a customer. Studio does not send a
# user_id unless you set one, so overriding this is the quickest way to get a clean slate
# for testing: DEFAULT_USER_ID=test-2 uv run langgraph dev
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "anonymous")


class AssistantContext(TypedDict, total=False):
    """Per-run settings. Declaring it makes `user_id` an editable field in Studio."""

    user_id: str


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
