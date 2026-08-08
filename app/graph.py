"""The label-assistant graph.

    START -> agent ─ MemoryRouter(profile)     -> update_profile     ────────┐
                   ├ MemoryRouter(artist)      -> update_artist -> publish   ┤
                   ├ MemoryRouter(found_music) -> update_found_music ────────┼-> agent
                   ├ catalog / weather / backend tool -> tools ──────────────┘
                   └ no tool call -----------------------------------------> END

The agent decides when something is worth remembering by calling the MemoryRouter tool;
the conditional edge reads `update_type` off that call and sends the run to the matching
trustcall node. MemoryRouter is bound as a tool but never executed as one.

`publish` is the one node that writes to the label's own backend, and it interrupts for a
yes before it does — see publish_review_node.
"""

from app.env import load_env_file

# Must run before the langchain/langgraph imports below: the LangSmith SDK caches some
# env-var reads on first access, so LANGSMITH_* has to be in os.environ by the time those
# packages are imported for tracing to pick it up.
load_env_file()

import os  # noqa: E402
from typing import Annotated, Any, Callable, Literal, Sequence, TypedDict  # noqa: E402

from langchain_core.messages import (  # noqa: E402
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph, add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.store.base import BaseStore  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from langgraph.types import interrupt  # noqa: E402

from app import memory  # noqa: E402
from app.catalog import catalog_tools  # noqa: E402
from app.identity import (  # noqa: E402,F401  (DEFAULT_USER_ID re-exported for callers)
    AssistantContext,
    DEFAULT_USER_ID,
    id_token_from_config,
    user_id_from_config,
)
from app.formatting import normalize_content  # noqa: E402
from app.graphql_backend_client import backend_tools, create_artist  # noqa: E402
from app.llm import require_model  # noqa: E402
from app.schemas import MemoryRouter  # noqa: E402
from app.weather_forcast import weather_forcast_tool  # noqa: E402

# Compaction. The whole transcript is re-sent on every agent turn, so a long shopping
# conversation gets expensive; past this many messages the tail is summarised and dropped.
SUMMARIZE_AFTER_MESSAGES: int = int(os.getenv("SUMMARIZE_AFTER_MESSAGES", "12"))
SUMMARY_KEEP_MESSAGES: int = int(os.getenv("SUMMARY_KEEP_MESSAGES", "4"))


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
# Persistence — the checkpointer holds threads, the store holds per-user memory
checkpointer = InMemorySaver()
store = InMemoryStore()

tools = [*catalog_tools, weather_forcast_tool, *backend_tools]
tool_node = ToolNode(tools)

# MemoryRouter is bound so the model can request a memory write, but it is not in
# `tools` — no ToolNode executes it; the router edge interprets it instead.
bindable_tools = [*tools, MemoryRouter]


# -------------------------------------------------------------
# 2. Helpers
# -------------------------------------------------------------
# Shared with the tools, which need it to know whose store to write to — see app/identity.py.
_user_id = user_id_from_config


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
    """Passes the conversation, plus everything remembered about the user, to Grok."""
    model_with_tools = require_model().bind_tools(bindable_tools, parallel_tool_calls=False)

    # Everything the model is told about this user — stored records and the summary of
    # any turns compaction has already dropped — is assembled in one place.
    summary = state.get("summary", "")
    system_prompt = SystemMessage(
        content=memory.render_memory_prompt(store, _user_id(config), summary=summary)
    )

    response = model_with_tools.invoke([system_prompt, *state["messages"]])

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

def _safe_cut(messages: Sequence[BaseMessage], keep: int) -> int:
    """Index of the first message to keep, moved forward to a safe boundary.

    Cutting at a fixed offset can land between an AI message requesting a tool and the
    ToolMessage answering it. Whichever half survives is then malformed — an orphan tool
    result, or a request nothing ever answered — and the next model call fails. Only a
    human turn or a plain AI reply is a safe place to start.
    """
    if len(messages) <= keep:
        return 0

    cut = len(messages) - keep
    while cut < len(messages):
        message = messages[cut]
        kind = getattr(message, "type", None)
        if kind == "human":
            break
        if kind == "ai" and not getattr(message, "tool_calls", None):
            break
        cut += 1
    return cut


def summarize_conversation_node(state: AgentState):
    """Fold older turns into `summary` and drop them from the transcript.

    Runs at the end of a turn, never in place of routing: a memory write or tool call must
    reach its node first, or the run stalls with an unanswered tool call.
    """
    messages = list(state["messages"])
    summary = state.get("summary", "")

    # Work out what would actually be dropped before paying for a summary. The safe
    # boundary can leave nothing to remove, and summarising to drop zero messages is a
    # model call bought for nothing.
    cut = _safe_cut(messages, SUMMARY_KEEP_MESSAGES)
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:cut] if m.id is not None]
    if not delete_messages:
        return {}

    model = require_model()

    if summary:
        instruction = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend it to take account of the newer messages above. Keep what the user "
            "asked for, what was chosen or ruled out and why."
        )
    else:
        instruction = (
            "Summarise the conversation above. Keep what the user asked for, what was "
            "chosen or ruled out and why."
        )

    response = model.invoke([*messages, HumanMessage(content=instruction)])

    return {
        "summary": normalize_content(getattr(response, "content", "")),
        "messages": delete_messages,
    }

def should_continue(
    state: AgentState,
) -> Literal[
    "update_profile",
    "update_artist",
    "update_found_music",
    "tools",
    "summarize_conversation",
    "__end__",
]:
    """Routes on what the agent asked for: a memory write, a tool, or nothing.

    Work in progress always wins. Compaction is checked only where the turn would
    otherwise end — intercepting a pending tool call to summarise instead would strand it
    with no ToolMessage, and the next model call would fail on the malformed history.
    """
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]
    router = _router_call(last_message)
    if router is not None:
        update_type = (router.get("args") or {}).get("update_type")
        if update_type in ("profile", "artist", "found_music"):
            return f"update_{update_type}"  # type: ignore[return-value]
        # Unrecognised update_type: fall through to the tool node, which answers the call
        # with an error the agent can recover from rather than stranding it.

    if getattr(last_message, "tool_calls", None):
        return "tools"

    # Turn is over: the user has their answer. Compact before the next one.
    if len(messages) > SUMMARIZE_AFTER_MESSAGES:
        return "summarize_conversation"
    return "__end__"


def _memory_node(
    record: Literal["profile", "artist", "found_music"],
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
update_artist_node = _memory_node("artist", memory.update_artist)
update_found_music_node = _memory_node("found_music", memory.update_found_music)


def publish_review_node(state: AgentState, config: RunnableConfig, store: BaseStore):
    """
    Show the user their artist profile and get a yes before it is sent to the label.

    Creating an artist is a public write the backend offers no way to undo, so it is the
    one thing in this graph that never happens on the model's say-so. The payload shown is
    built from the stored record, not from anything the model composed for this turn.

    Nothing to publish — no artist record, no name on it yet, or the label already has it —
    means this node does nothing at all and the run carries straight on.

    Resume via POST /threads/{id}/runs with exactly one of:
        {"command": {"resume": {"confirm": true}}}
        {"command": {"resume": {"amend": {"genre": "shoegaze", "bio": "..."}}}}
        {"command": {"resume": {"cancel": true}}}
    Anything else re-prompts, so the thread stays paused until it gets a usable answer.
    """
    user_id = _user_id(config)
    error: str | None = None

    while True:
        # Re-read every pass: an amendment below changes the payload, and the loop
        # re-shows it before asking again.
        artist = memory.get_artist(store, user_id)
        if not memory.awaiting_publication(artist):
            return {}

        payload = artist.to_create_input()  # not None: awaiting_publication checked it
        assert payload is not None

        request: dict[str, Any] = {
            "question": (
                f"Publish '{payload.name}' to the label? This creates a public artist "
                "profile, and the label's backend has no way to delete one."
            ),
            "artist": payload.model_dump(),
            # Not blockers — the backend only requires a name — but worth a last look
            # before something public goes out half-filled.
            "incomplete": [
                field
                for field in ("genre", "bio", "image_url")
                if not getattr(payload, field)
            ],
        }
        if error:
            request["error"] = error

        user_input: Any = interrupt(request)

        # Resuming re-runs this node from the top, so everything before the interrupt
        # happens again on replay. Only reads and idempotent amendments live above it;
        # the one irreversible act — the write itself — is below the loop, reached once.
        if not isinstance(user_input, dict):
            error = "Send an object with one of: confirm, amend, cancel."
            continue

        if user_input.get("cancel"):
            return {
                "messages": [
                    HumanMessage(
                        content="I don't want to publish my artist profile yet. "
                        "Leave it as a draft."
                    )
                ]
            }

        amend = user_input.get("amend")
        if isinstance(amend, dict) and amend:
            memory.set_artist_fields(store, user_id, **amend)
            error = None
            continue

        if user_input.get("confirm"):
            break

        error = "Send an object with one of: confirm, amend, cancel."

    # Confirmed. Read once more so an amendment made on the final pass is included.
    artist = memory.get_artist(store, user_id)
    payload = artist.to_create_input() if artist else None
    if payload is None:  # pragma: no cover - the loop cannot exit with this unset
        return {}

    result = create_artist(
        name=payload.name,
        genre=payload.genre,
        bio=payload.bio,
        image_url=payload.image_url,
        id_token=id_token_from_config(config),
    )

    if "error" in result:
        # The record keeps backend_artist_id unset, so the gate will offer again next
        # time rather than leaving the user thinking they are on the roster.
        outcome = (
            f"Publishing my artist profile failed: {result['error']} "
            "Tell me what happened and what I can do about it."
        )
    else:
        memory.mark_artist_published(
            store, user_id, result.get("id", ""), result.get("artistSlug")
        )
        outcome = (
            f"My artist profile is now published to the label as "
            f"'{result.get('name', payload.name)}'"
            + (f" (slug {result['artistSlug']})" if result.get("artistSlug") else "")
            + ". Tell me what that means and what happens next."
        )

    # Report back as a human turn, matching how the rest of this graph hands control back
    # to the agent after something happened outside its control.
    return {"messages": [HumanMessage(content=outcome)]}


# -------------------------------------------------------------
# 4. Build
# -------------------------------------------------------------
workflow = StateGraph(AgentState, context_schema=AssistantContext)
workflow.add_node("agent", call_grok_agent)
workflow.add_node("tools", tool_node)
workflow.add_node("publish_review", publish_review_node)
workflow.add_node("update_profile", update_profile_node)
workflow.add_node("update_artist", update_artist_node)
workflow.add_node("update_found_music", update_found_music_node)
workflow.add_node("summarize_conversation", summarize_conversation_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)

# Tools return straight to the agent: they answer questions, they do not change who the
# user is. The publish gate sits after the one node that can complete an artist profile,
# and is the only path to a write on the label's own backend.
workflow.add_edge("tools", "agent")
workflow.add_edge("update_profile", "agent")
workflow.add_edge("update_found_music", "agent")
workflow.add_edge("update_artist", "publish_review")
workflow.add_edge("publish_review", "agent")
# Compaction is the last thing that happens in a turn, so it ends the run rather than
# looping back — going back to the agent would produce a second reply to the same message.
workflow.add_edge("summarize_conversation", END)

# Two compilations of the same workflow, because the two runtimes persist differently.
#
# `graph` is what langgraph.json exposes to the LangGraph CLI and Studio. It must NOT carry
# a checkpointer or store: the platform supplies its own, and refuses to load a graph that
# brings its own (GraphLoadError).
#
# `api_graph` is what the FastAPI app runs, and it does need them — nothing else would keep
# threads or user memory alive between requests.
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
