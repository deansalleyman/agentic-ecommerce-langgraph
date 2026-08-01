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

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from langchain_core.messages import BaseMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_xai import ChatXAI  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph, add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from langgraph.types import interrupt

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
    model = ChatXAI(api_key=xai_api_key, model=xai_model, temperature=0.1)
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
# 4. Define Graph Nodes and Routing Logic
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


def should_continue(state: AgentState) -> Literal["tools", END]:
    """Inspects the last message to see if Grok requested a tool call."""
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

def approval_node(state: AgentState):
    # Pause and ask for approval
    approved = interrupt("Do you approve this action?")

    # When you resume, Command(resume=...) returns that value here
    return {"approved": approved}

# Build and compile the LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_grok_agent)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
graph = workflow.compile(checkpointer=checkpointer, store=store)


# -------------------------------------------------------------
# 5. FastAPI app
# -------------------------------------------------------------
app = FastAPI(title="Agentic E-commerce LangGraph API", version="1.0.0")


class StartPayload(BaseModel):
    text: str


class ResumePayload(BaseModel):
    thread_id: str
    text: str


def build_initial_state(text: str) -> AgentState:
    return {
        "input": text,
        "output": "",
        "messages": [HumanMessage(content=text)],
        "original_query": text,
        "summary": "",
        "current_topic": text,
    }


def _stream_graph(input_state: Any, thread_id: str) -> StreamingResponse:
    """Stream graph execution as Server-Sent Events."""
    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        yield f"data: {json.dumps({'event': 'thread_created', 'thread_id': thread_id})}\n\n"

        for chunk in graph.stream(input_state, config=config, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                for msg in node_output.get("messages", []):
                    payload = {
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


@app.post("/run-graph")
def run_graph(payload: StartPayload):
    """Start a new graph thread and stream the ReAct execution back as SSE."""
    thread_id = str(uuid.uuid4())
    return _stream_graph(build_initial_state(payload.text), thread_id)


@app.post("/user-response")
def user_response(payload: ResumePayload):
    """Continue an existing thread with a new user message and stream the response."""
    return _stream_graph(
        {"messages": [HumanMessage(content=payload.text)]},
        payload.thread_id,
    )


@app.get("/test")
async def test_graph(text: str):
    """Quick GET endpoint for browser testing — non-streaming."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(build_initial_state(text), config=config)
    react_output = build_react_output(result, thread_id)
    return {
        "thread_id": thread_id,
        "summary": result.get("summary", ""),
        "original_query": result.get("original_query", text),
        "react": react_output,
    }

