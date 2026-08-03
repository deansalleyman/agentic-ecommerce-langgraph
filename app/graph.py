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

from langchain_core.messages import RemoveMessage, SystemMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
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
from app.schemas import GearItem, MemoryRouter  # noqa: E402

# Maximum allowed pack weight before the graph pauses for human review
MAX_CARGO_WEIGHT_LB: float = float(os.getenv("MAX_CARGO_WEIGHT_LB", "220"))

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
def calculate_gear_weight(items: list[GearItem]) -> float:
    """Calculates the total weight of a pack list, in lb.

    Use for working out what a camper can reasonably carry in their backpack. Pass one
    entry per distinct item, each with its own weight in lb and how many are carried.

    Args:
        items: The pack list, e.g.
            [{"name": "tent", "weight_lb": 3.79, "quantity": 1},
             {"name": "day of food", "weight_lb": 1.8, "quantity": 3}]

    Returns:
        The total weight of everything on the list, in lb.
    """
    total = sum(item.weight_lb * item.quantity for item in items)
    print(f"Calculating gear weight: {len(items)} lines, total={total} lb")
    return total


# Persistence — the checkpointer holds threads, the store holds per-user memory
checkpointer = InMemorySaver()
store = InMemoryStore()

tools = [calculate_gear_weight, *catalog_tools]
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


# -------------------------------------------------------------
# 3. Nodes
# -------------------------------------------------------------
def call_grok_agent(state: AgentState, config: RunnableConfig, store: BaseStore):
    """Passes the conversation, plus everything remembered about the customer, to Grok."""
    model_with_tools = require_model().bind_tools(bindable_tools, parallel_tool_calls=False)

    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + list(state["messages"])
    
    else:
        messages = state["messages"]

    user_id = _user_id(config)
    system_prompt = SystemMessage(
        content=memory.render_memory_prompt(store, user_id, MAX_CARGO_WEIGHT_LB)
    )



    response = model_with_tools.invoke([system_prompt, *messages])

    original_query = state.get("original_query", "")

   
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

def summarize_conversation_node(state: AgentState):

    model = require_model()
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = list(state["messages"]) + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def should_continue(
    state: AgentState,
) -> Literal["update_profile", "update_trip", "update_gear_needs", "tools", "summarize_conversation", "__end__"]:
    """Routes on what the agent asked for: a memory write, a tool, or nothing. also summarizes message history"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

     # If there are more than six messages, then we summarize the conversation
    if len(messages) > 2:
        return "summarize_conversation"

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


def weight_review_node(state: AgentState, config: RunnableConfig, store: BaseStore):
    """
    Check the customer's chosen kit against the weight limit for this trip, and pause if
    it is over.

    The total is computed from the store — the products they have actually chosen, weighed
    from the catalog — not read off a model's arithmetic. The limit is the trip's own
    max_carry_weight_lb when they have set one, falling back to MAX_CARGO_WEIGHT_LB.

    Resume via POST /threads/{id}/runs with exactly one of:
        {"command": {"resume": {"swap": {"need_id": "tent", "product_id": "..."}}}}
        {"command": {"resume": {"drop": "tent"}}}
        {"command": {"resume": {"max_carry_weight_lb": 35}}}
        {"command": {"resume": {"warning_overridden": true}}}
    Anything else re-prompts, so the thread stays paused until it gets a usable answer.
    """
    user_id = _user_id(config)
    applied: list[str] = []
    error: str | None = None

    while True:
        # Re-read every pass: an edit below changes the total, and the loop re-checks it.
        trip = memory.get_trip(store, user_id)
        needs = memory.get_gear_needs(store, user_id)
        totals = memory.committed_totals(needs)
        limit = memory.carry_limit(trip, MAX_CARGO_WEIGHT_LB)
        over_by = round(totals["weight_lb"] - limit, 2)

        if over_by <= 0:
            break

        request: dict[str, Any] = {
            "question": (
                f"The chosen kit is {totals['weight_lb']} lb, {over_by} lb over the "
                f"{limit} lb limit for this trip. Swap something lighter, drop an item, "
                "raise the limit, or carry on anyway."
            ),
            "committed_lb": totals["weight_lb"],
            "limit_lb": limit,
            "over_by_lb": over_by,
            "items": totals["items"],
            "lighter_options": memory.lighter_alternatives(needs),
        }
        if error:
            request["error"] = error

        user_input: Any = interrupt(request)

        # Resuming re-runs this node from the top, so any edit made here happens again on
        # replay. Each of these is idempotent — setting the same selection or dropping an
        # already-dropped need changes nothing the second time.
        if not isinstance(user_input, dict):
            error = "Send an object with one of: swap, drop, max_carry_weight_lb, warning_overridden."
            continue

        if user_input.get("warning_overridden"):
            applied.append(
                f"The customer chose to carry on at {totals['weight_lb']} lb, "
                f"{over_by} lb over their {limit} lb limit."
            )
            break

        swap = user_input.get("swap")
        if isinstance(swap, dict) and swap.get("need_id") and swap.get("product_id"):
            if memory.swap_selection(store, user_id, swap["need_id"], swap["product_id"]):
                applied.append(f"Swapped {swap['need_id']} to {swap['product_id']}.")
                error = None
                continue
            error = f"No such need or product: {swap['need_id']} / {swap['product_id']}."
            continue

        dropped = user_input.get("drop")
        if isinstance(dropped, str) and dropped:
            if memory.drop_selection(
                store, user_id, dropped, f"Pushed the pack over its {limit} lb limit."
            ):
                applied.append(f"Dropped the chosen {dropped}.")
                error = None
                continue
            error = f"No such gear need: {dropped}."
            continue

        new_limit = user_input.get("max_carry_weight_lb")
        if isinstance(new_limit, (int, float)) and not isinstance(new_limit, bool):
            if new_limit <= 0:
                error = "max_carry_weight_lb must be greater than zero."
                continue
            memory.set_carry_limit(store, user_id, float(new_limit))
            applied.append(f"Raised the trip's pack limit to {new_limit} lb.")
            error = None
            continue

        error = "Send an object with one of: swap, drop, max_carry_weight_lb, warning_overridden."

    if not applied:
        return {}

    # Tell the agent what the customer decided, so it can explain the new position.
    return {
        "messages": [
            HumanMessage(
                content=" ".join(applied)
                + " Tell me where that leaves my pack weight."
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
workflow.add_node("summarize_conversation", summarize_conversation_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)

# Tools return straight to the agent: they answer questions, they do not change what the
# customer has chosen. The weight review sits after the two nodes that can put the pack
# over its limit — deciding a product, or lowering the limit itself.
workflow.add_edge("tools", "agent")
workflow.add_edge("update_profile", "agent")
workflow.add_edge("update_trip", "weight_review")
workflow.add_edge("update_gear_needs", "weight_review")
workflow.add_edge("weight_review", "agent")

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
