import json
import os
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence, TypedDict


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


# Must run before the langchain/langgraph imports below: the LangSmith SDK caches
# some env-var reads on first access, so LANGSMITH_* has to be in os.environ by
# the time those packages are imported for tracing to pick it up.
load_env_file()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from langchain_core.messages import BaseMessage, HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_xai import ChatXAI  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import START, StateGraph, add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402
from pydantic import BaseModel, SecretStr  # noqa: E402

# Maximum allowed cargo weight before the graph pauses for human review
MAX_CARGO_WEIGHT_KG: float = float(os.getenv("MAX_CARGO_WEIGHT_KG", "100"))


class AgentState(TypedDict):
    # Appends new messages to the history instead of overwriting
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_topic: str
    summary: str
    original_query: str
    input: str
    output: str


# -------------------------------------------------------------
# 1. Define custom tools for Grok to use
# -------------------------------------------------------------
@tool
def calculate_cargo_weight(item_count: int, unit_weight: float) -> float:
    """Calculates total cargo weight for freight shipping logistics."""
    print(f"Calculating cargo weight: item_count={item_count}, unit_weight={unit_weight}")
    return item_count * unit_weight


# Persistence — one in-memory store shared across all threads
checkpointer = InMemorySaver()
store = InMemoryStore()

tools = [calculate_cargo_weight]
tool_node = ToolNode(tools)


# -------------------------------------------------------------
# 2. Initialize Grok
# -------------------------------------------------------------
xai_api_key = os.getenv("XAI_API_KEY", "")
xai_model = os.getenv("XAI_MODEL", "grok-4")
model: ChatXAI | None = None
model_with_tools: Any | None = None

if xai_api_key:
    model = ChatXAI(api_key=SecretStr(xai_api_key), model=xai_model, temperature=0.1)
    model_with_tools = model.bind_tools(tools)


# -------------------------------------------------------------
# 3. Helper utilities
# -------------------------------------------------------------
def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def build_react_output(state: dict[str, Any], thread_id: str) -> dict[str, Any]:
    messages = state.get("messages", [])
    steps: list[dict[str, Any]] = []

    for message in messages:
        msg_type = getattr(message, "type", None)
        if msg_type == "human":
            steps.append({
                "type": "human",
                "content": _normalize_content(getattr(message, "content", "")),
            })
        elif msg_type == "ai":
            tool_calls = [
                {
                    "name": getattr(tc, "name", "") if hasattr(tc, "name") else tc.get("name", ""),
                    "args": getattr(tc, "args", {}) if hasattr(tc, "args") else tc.get("args", {}),
                    "id": getattr(tc, "id", None) if hasattr(tc, "id") else tc.get("id"),
                }
                for tc in (getattr(message, "tool_calls", []) or [])
            ]
            steps.append({
                "type": "ai",
                "content": _normalize_content(getattr(message, "content", "")),
                "tool_calls": tool_calls,
            })
        elif msg_type == "tool":
            steps.append({
                "type": "tool",
                "name": getattr(message, "name", ""),
                "content": _normalize_content(getattr(message, "content", "")),
                "tool_call_id": getattr(message, "tool_call_id", None),
            })

    final_answer = ""
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and not getattr(message, "tool_calls", None):
            final_answer = _normalize_content(getattr(message, "content", ""))
            break

    return {"steps": steps, "final_answer": final_answer, "thread_id": thread_id}


# -------------------------------------------------------------
# 4. Graph nodes and routing
# -------------------------------------------------------------
def call_grok_agent(state: AgentState):
    """Passes the current conversation history to Grok to decide the next move."""
    if model_with_tools is None:
        raise RuntimeError("XAI_API_KEY is not set. Please set the environment variable before running the graph.")

    response = model_with_tools.invoke(state["messages"])
    original_query = state.get("original_query", "")
    existing_summary = state.get("summary", "")
    summary = existing_summary or f"Handled request: {original_query}"
    current_topic = state.get("current_topic") or original_query or "general"
    output = _normalize_content(getattr(response, "content", ""))

    return {
        "messages": [response],
        "summary": summary,
        "original_query": original_query,
        "current_topic": current_topic,
        "output": output,
    }


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Routes to the tools node if Grok requested a tool call, otherwise ends."""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "__end__"


def weight_review_node(state: AgentState):
    """
    After cargo weight is calculated, check if it exceeds MAX_CARGO_WEIGHT_KG.
    If it does, pause via interrupt() and ask the user to supply new item_count
    and unit_weight. The graph resumes when POST /threads/{id}/runs is called
    with {"command": {"resume": {"item_count": ..., "unit_weight": ...}}}.
    Unusable values re-prompt rather than resuming.
    """
    messages = state.get("messages", [])

    # Find the most recent calculate_cargo_weight tool result
    last_tool_msg = next(
        (
            m for m in reversed(messages)
            if getattr(m, "type", None) == "tool"
            and getattr(m, "name", "") == "calculate_cargo_weight"
        ),
        None,
    )

    if last_tool_msg is None:
        return {}

    try:
        weight = float(_normalize_content(getattr(last_tool_msg, "content", "0")))
    except (ValueError, TypeError):
        return {}

    if weight <= MAX_CARGO_WEIGHT_KG:
        # Weight is acceptable — continue to agent for the final answer
        return {}

    # Weight exceeds the limit — pause and hand control back to the user.
    # A malformed answer re-prompts instead of resuming: interrupt() is called
    # again, so the thread stays paused until usable numbers arrive.
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


# Build and compile the LangGraph workflow
# weight_review sits between tools and agent so it can intercept heavy loads
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_grok_agent)
workflow.add_node("tools", tool_node)
workflow.add_node("weight_review", weight_review_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "weight_review")
workflow.add_edge("weight_review", "agent")
graph = workflow.compile(checkpointer=checkpointer, store=store)


# -------------------------------------------------------------
# 5. FastAPI app
# -------------------------------------------------------------
app = FastAPI(title="Agentic E-commerce LangGraph API", version="1.0.0")


class RunCommand(BaseModel):
    """Mirrors LangGraph's Command — the value handed back to a paused interrupt()."""

    resume: dict[str, Any]


class RunPayload(BaseModel):
    # Exactly one of these: `input` starts or continues a conversation,
    # `command` resumes a thread paused at an interrupt.
    input: str | None = None
    command: RunCommand | None = None


# Threads live only as long as the process, same as the InMemorySaver backing them.
# Tracking ids explicitly is what lets an unknown thread 404 instead of silently
# behaving like a brand-new one.
_thread_ids: set[str] = set()


def _pending_interrupts(snapshot: Any) -> list[Any]:
    """Interrupt payloads the thread is currently waiting on, newest task first."""
    return [
        getattr(intr, "value", intr)
        for task in (snapshot.tasks or ())
        for intr in (getattr(task, "interrupts", None) or ())
    ]


def build_initial_state(text: str) -> AgentState:
    return {
        "input": text,
        "output": "",
        "messages": [HumanMessage(content=text)],
        "original_query": text,
        "summary": "",
        "current_topic": text,
    }


def _stream_graph(graph_input: Any, thread_id: str) -> StreamingResponse:
    """Stream graph execution (new run or Command resume) as Server-Sent Events."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        yield f"data: {json.dumps({'event': 'run_started', 'thread_id': thread_id})}\n\n"

        for chunk in graph.stream(graph_input, config=config, stream_mode="updates"):
            # Interrupt detected — pause and tell the client to post a resume command
            if "__interrupt__" in chunk:
                for intr in chunk["__interrupt__"]:
                    intr_value = getattr(intr, "value", intr)
                    yield (
                        f"data: {json.dumps({'event': 'interrupt', 'thread_id': thread_id, 'data': intr_value})}\n\n"
                    )
                return  # Stop streaming until the user resumes

            for node_name, node_output in chunk.items():
                # A node that returns {} or None (e.g. weight_review under the limit)
                # streams as {"node": None}; a node that ran more than once in a
                # superstep streams a list of updates.
                updates = node_output if isinstance(node_output, list) else [node_output]
                for update in updates:
                    if not isinstance(update, dict):
                        continue
                    for msg in update.get("messages") or []:
                        payload: dict[str, Any] = {
                            "event": "message",
                            "node": node_name,
                            "type": getattr(msg, "type", ""),
                            "content": _normalize_content(getattr(msg, "content", "")),
                        }
                        tool_calls = getattr(msg, "tool_calls", None)
                        if tool_calls:
                            payload["tool_calls"] = [
                                {
                                    "name": tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                                    "args": tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}),
                                }
                                for tc in tool_calls
                            ]
                        yield f"data: {json.dumps(payload)}\n\n"

        final_state = graph.get_state(config)
        react_output = build_react_output(dict(final_state.values), thread_id)
        yield f"data: {json.dumps({'event': 'done', 'thread_id': thread_id, 'react': react_output})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/")
def read_root():
    return {"message": "LangGraph FastAPI is running locally!"}


@app.post("/threads", status_code=201)
def create_thread():
    """Create a conversation thread. Runs are then posted against its id."""
    thread_id = str(uuid.uuid4())
    _thread_ids.add(thread_id)
    return {"thread_id": thread_id}


@app.post("/threads/{thread_id}/runs")
def create_run(thread_id: str, payload: RunPayload):
    """
    Execute the graph on a thread and stream the ReAct trace back as SSE.

    - `{"input": "..."}` starts the conversation, or adds a turn to an existing one.
    - `{"command": {"resume": {...}}}` resumes a thread paused at an interrupt.

    Which one is required depends on whether the thread is currently paused, so
    the client never has to guess — a mismatch comes back as 409 with the reason.
    """
    if thread_id not in _thread_ids:
        raise HTTPException(status_code=404, detail=f"Unknown thread: {thread_id}")

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    interrupts = _pending_interrupts(snapshot)

    if interrupts:
        if payload.command is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "Thread is paused at an interrupt; resume it with a command.",
                    "interrupts": interrupts,
                },
            )
        return _stream_graph(Command(resume=payload.command.resume), thread_id)

    if payload.command is not None:
        raise HTTPException(
            status_code=409,
            detail="Thread has no pending interrupt to resume; send an input instead.",
        )
    if not payload.input:
        raise HTTPException(status_code=422, detail="input must be a non-empty string.")

    # First run on the thread seeds the full state; later runs just add a turn.
    started = bool(snapshot.values.get("messages"))
    graph_input: Any = (
        {"messages": [HumanMessage(content=payload.input)]}
        if started
        else build_initial_state(payload.input)
    )
    return _stream_graph(graph_input, thread_id)


@app.get("/threads/{thread_id}/state")
def get_thread_state(thread_id: str):
    """Non-streaming snapshot of a thread — useful after a run, or from the browser."""
    if thread_id not in _thread_ids:
        raise HTTPException(status_code=404, detail=f"Unknown thread: {thread_id}")

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    values = dict(snapshot.values)
    interrupts = _pending_interrupts(snapshot)

    return {
        "thread_id": thread_id,
        "status": "interrupted" if interrupts else "idle",
        "interrupts": interrupts,
        "summary": values.get("summary", ""),
        "original_query": values.get("original_query", ""),
        "react": build_react_output(values, thread_id),
    }



