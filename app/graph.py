"""The shopping-assistant graph.

    START -> agent ─ MemoryRouter(profile)    -> update_profile    ─┐
                   ├ MemoryRouter(trip)       -> update_trip       ─┤
                   ├ MemoryRouter(gear_needs) -> update_gear_needs ─┼-> agent
                   ├ catalog / cargo tool     -> tools -> weight_review
                   └ no tool call ------------------------------------> END

The agent decides when something is worth remembering by calling the MemoryRouter tool;
the conditional edge reads `update_type` off that call and sends the run to the matching
trustcall node. MemoryRouter is bound as a tool but never executed as one.
"""

from app.env import load_env_file

# Must run before the langchain/langgraph imports below: the LangSmith SDK caches some
# env-var reads on first access, so LANGSMITH_* has to be in os.environ by the time those
# packages are imported for tracing to pick it up.
load_env_file()

import os  # noqa: E402
from typing import Annotated, Any, Callable, Literal, Sequence, TypedDict  # noqa: E402

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import START, StateGraph, add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.store.base import BaseStore  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from langgraph.types import interrupt  # noqa: E402

from app import memory  # noqa: E402
from app.catalog import catalog_tools  # noqa: E402
from app.formatting import normalize_content  # noqa: E402
from app.llm import require_model  # noqa: E402
from app.schemas import MemoryRouter  # noqa: E402

# Maximum allowed pack weight before the graph pauses for human review
MAX_CARGO_WEIGHT_KG: float = float(os.getenv("MAX_CARGO_WEIGHT_KG", "100"))

DEFAULT_USER_ID = "anonymous"


class AgentState(TypedDict):
    # Appends new messages to the history instead of overwriting
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_topic: str
    summary: str
    original_query: str
    input: str
    output: str
    # Set by the memory nodes so a run can stream what was just written
    last_memory_update: dict[str, Any] | None


# -------------------------------------------------------------
# 1. Tools
# -------------------------------------------------------------
@tool
def calculate_cargo_weight(item_count: int, unit_weight: float) -> float:
    """Calculates the total weight of a number of identical items, in kg.

    Use for working out what a pack or a shipment will weigh.
    """
    print(f"Calculating cargo weight: item_count={item_count}, unit_weight={unit_weight}")
    return item_count * unit_weight


# Persistence — the checkpointer holds threads, the store holds per-user memory
checkpointer = InMemorySaver()
store = InMemoryStore()

tools = [calculate_cargo_weight, *catalog_tools]
tool_node = ToolNode(tools)

# MemoryRouter is bound so the model can request a memory write, but it is not in
# `tools` — no ToolNode executes it; the router edge interprets it instead.
bindable_tools = [*tools, MemoryRouter]


# -------------------------------------------------------------
# 2. Helpers
# -------------------------------------------------------------
def _user_id(config: RunnableConfig) -> str:
    return (config.get("configurable") or {}).get("user_id") or DEFAULT_USER_ID


def _router_call(message: Any) -> dict[str, Any] | None:
    """The MemoryRouter call on a message, if it made one."""
    for call in getattr(message, "tool_calls", None) or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name == "MemoryRouter":
            return call if isinstance(call, dict) else {"name": name, "args": {}, "id": None}
    return None


def _trailing_tool_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Tool results produced by the most recent step only.

    Scanning the whole history instead would let a stale over-limit weight from earlier in
    the conversation re-trigger the review on an unrelated turn.
    """
    trailing: list[BaseMessage] = []
    for message in reversed(messages):
        if getattr(message, "type", None) != "tool":
            break
        trailing.append(message)
    return list(reversed(trailing))


# -------------------------------------------------------------
# 3. Nodes
# -------------------------------------------------------------
def call_grok_agent(state: AgentState, config: RunnableConfig, store: BaseStore):
    """Passes the conversation, plus everything remembered about the customer, to Grok."""
    model_with_tools = require_model().bind_tools(bindable_tools, parallel_tool_calls=False)

    user_id = _user_id(config)
    system_prompt = SystemMessage(content=memory.render_memory_prompt(store, user_id))
    response = model_with_tools.invoke([system_prompt, *state["messages"]])

    original_query = state.get("original_query", "")
    existing_summary = state.get("summary", "")
    summary = existing_summary or f"Handled request: {original_query}"
    current_topic = state.get("current_topic") or original_query or "general"
    output = normalize_content(getattr(response, "content", ""))

    return {
        "messages": [response],
        "summary": summary,
        "original_query": original_query,
        "current_topic": current_topic,
        "output": output,
        "last_memory_update": None,
    }


def should_continue(
    state: AgentState,
) -> Literal["update_profile", "update_trip", "update_gear_needs", "tools", "__end__"]:
    """Routes on what the agent asked for: a memory write, a tool, or nothing."""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    router = _router_call(last_message)
    if router is not None:
        update_type = (router.get("args") or {}).get("update_type")
        if update_type in ("profile", "trip", "gear_needs"):
            return f"update_{update_type}"  # type: ignore[return-value]
        # Unrecognised update_type: fall through to the tool node, which answers the call
        # with an error the agent can recover from rather than stranding it.

    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "__end__"


def _memory_node(
    record: Literal["profile", "trip", "gear_needs"],
    update: Callable[[BaseStore, str, list[AnyMessage]], tuple[list[str], dict[str, Any]]],
):
    """Builds a node that patches one record and reports back what changed."""

    def node(state: AgentState, config: RunnableConfig, store: BaseStore):
        messages = list(state["messages"])
        router = _router_call(messages[-1]) if messages else None

        # Everything except the AI message carrying the router call: that message has a
        # tool call with no matching ToolMessage, which is a malformed history to extract from.
        history: list[AnyMessage] = messages[:-1]  # type: ignore[assignment]

        changes, saved = update(store, _user_id(config), history)

        if changes:
            content = f"Saved to {record}. Changes:\n" + "\n".join(f"- {c}" for c in changes)
        else:
            content = f"Nothing new to save to {record}; the record already matched."

        return {
            "messages": [
                ToolMessage(
                    content=content,
                    name="MemoryRouter",
                    tool_call_id=(router or {}).get("id") or "memory_router",
                )
            ],
            "last_memory_update": {
                "record": record,
                "changes": changes,
                "reason": (router or {}).get("args", {}).get("reason"),
                "saved": saved,
            },
        }

    return node


update_profile_node = _memory_node("profile", memory.update_profile)
update_trip_node = _memory_node("trip", memory.update_trip)
update_gear_needs_node = _memory_node("gear_needs", memory.update_gear_needs)


def weight_review_node(state: AgentState):
    """
    After a weight is calculated, check whether it exceeds MAX_CARGO_WEIGHT_KG. If it
    does, pause via interrupt() and ask for new item_count and unit_weight. The graph
    resumes when POST /threads/{id}/runs is called with
    {"command": {"resume": {"item_count": ..., "unit_weight": ...}}}.
    Unusable values re-prompt rather than resuming.
    """
    last_tool_msg = next(
        (
            m
            for m in reversed(_trailing_tool_messages(state.get("messages", [])))
            if getattr(m, "name", "") == "calculate_cargo_weight"
        ),
        None,
    )

    if last_tool_msg is None:
        return {}

    try:
        weight = float(normalize_content(getattr(last_tool_msg, "content", "0")))
    except (ValueError, TypeError):
        return {}

    if weight <= MAX_CARGO_WEIGHT_KG:
        # Weight is acceptable — continue to agent for the final answer
        return {}

    # Weight exceeds the limit — pause and hand control back to the user.
    # A malformed answer re-prompts instead of resuming: interrupt() is called again, so
    # the thread stays paused until usable numbers arrive.
    question = (
        f"Total weight {weight} kg exceeds the {MAX_CARGO_WEIGHT_KG} kg limit. "
        "Please provide adjusted item_count and unit_weight."
    )
    error: str | None = None

    while True:
        request: dict[str, Any] = {
            "question": question,
            "current_weight": weight,
            "limit_kg": MAX_CARGO_WEIGHT_KG,
        }
        if error:
            request["error"] = error

        user_input: Any = interrupt(request)

        try:
            new_item_count = int(user_input["item_count"])
            new_unit_weight = float(user_input["unit_weight"])
        except (KeyError, TypeError, ValueError):
            error = "item_count (integer) and unit_weight (number) are both required."
            continue

        if new_item_count <= 0 or new_unit_weight <= 0:
            error = "item_count and unit_weight must both be greater than zero."
            continue

        break

    # Inject a new human message so the agent recalculates with the updated values
    return {
        "messages": [
            HumanMessage(
                content=(
                    f"Recalculate cargo weight with {new_item_count} items "
                    f"at {new_unit_weight} kg each."
                )
            )
        ]
    }


# -------------------------------------------------------------
# 4. Build
# -------------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_grok_agent)
workflow.add_node("tools", tool_node)
workflow.add_node("weight_review", weight_review_node)
workflow.add_node("update_profile", update_profile_node)
workflow.add_node("update_trip", update_trip_node)
workflow.add_node("update_gear_needs", update_gear_needs_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "weight_review")
workflow.add_edge("weight_review", "agent")
workflow.add_edge("update_profile", "agent")
workflow.add_edge("update_trip", "agent")
workflow.add_edge("update_gear_needs", "agent")

# Two compilations of the same workflow, because the two runtimes persist differently.
#
# `graph` is what langgraph.json exposes to the LangGraph CLI and Studio. It must NOT carry
# a checkpointer or store: the platform supplies its own, and refuses to load a graph that
# brings its own (GraphLoadError).
#
# `api_graph` is what the FastAPI app runs, and it does need them — nothing else would keep
# threads or customer memory alive between requests.
graph = workflow.compile()
api_graph = workflow.compile(checkpointer=checkpointer, store=store)


def build_initial_state(text: str) -> AgentState:
    return {
        "input": text,
        "output": "",
        "messages": [HumanMessage(content=text)],
        "original_query": text,
        "summary": "",
        "current_topic": text,
        "last_memory_update": None,
    }
