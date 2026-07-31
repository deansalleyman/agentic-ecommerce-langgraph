import os
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence, TypedDict

from fastapi import FastAPI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_xai import ChatXAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel


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


load_env_file()


class AgentState(TypedDict):
    # Appends new messages to the history instead of overwriting
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_topic: str
    summary: str
    original_query: str
    decision: str
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


tools = [calculate_cargo_weight]
tool_node = ToolNode(tools)


# -------------------------------------------------------------
# 2. Initialize Grok
# -------------------------------------------------------------
# fetch the XAI API key from environment variables
xai_api_key = os.getenv("XAI_API_KEY") or ""
xai_model = os.getenv("XAI_MODEL", "grok-4")
model: ChatXAI | None = None
model_with_tools: Any | None = None

if xai_api_key:
    model = ChatXAI(api_key=xai_api_key, model=xai_model, temperature=0.1)
    model_with_tools = model.bind_tools(tools)


# -------------------------------------------------------------
# 3. Define Graph Nodes and Routing Logic
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


# 3. Build and compile the LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_grok_agent)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
graph = workflow.compile()


# 4. Initialize FastAPI app
app = FastAPI(title="Local LangGraph FastAPI", version="1.0.0")


class RequestPayload(BaseModel):
    text: str
    decision: str = ""


def build_initial_state(text: str, decision: str = "") -> AgentState:
    return {
        "input": text,
        "output": "",
        "messages": [HumanMessage(content=text)],
        "decision": decision,
        "original_query": text,
        "summary": "",
        "current_topic": text,
    }


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def build_react_output(state: dict[str, Any]) -> dict[str, Any]:
    messages = state.get("messages", [])
    steps: list[dict[str, Any]] = []

    for message in messages:
        if getattr(message, "type", None) == "human":
            steps.append({
                "type": "human",
                "content": _normalize_content(getattr(message, "content", "")),
            })
        elif getattr(message, "type", None) == "ai":
            tool_calls = []
            for tool_call in getattr(message, "tool_calls", []) or []:
                tool_calls.append({
                    "name": getattr(tool_call, "name", ""),
                    "args": getattr(tool_call, "args", {}),
                    "id": getattr(tool_call, "id", None),
                })
            steps.append({
                "type": "ai",
                "content": _normalize_content(getattr(message, "content", "")),
                "tool_calls": tool_calls,
            })
        elif getattr(message, "type", None) == "tool":
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

    return {
        "steps": steps,
        "final_answer": final_answer,
    }


@app.get("/")
def read_root():
    return {"message": "LangGraph FastAPI is running locally!"}


# 5. Expose graph execution via a POST route
@app.post("/run-graph")
def run_graph(payload: RequestPayload):
    initial_state = build_initial_state(payload.text, payload.decision)
    result = graph.invoke(initial_state)
    react_output = build_react_output(result)
    return {
        "result": {
            **result,
            "output": react_output["final_answer"],
            "react": react_output,
        },
        "summary": result.get("summary", ""),
        "original_query": result.get("original_query", payload.text),
        "react": react_output,
    }


@app.get("/test")
async def test_graph(text: str):
    initial_state = build_initial_state(text, "test")
    result = graph.invoke(initial_state)
    react_output = build_react_output(result)
    return {
        "result": {
            **result,
            "output": react_output["final_answer"],
            "react": react_output,
        },
        "summary": result.get("summary", ""),
        "original_query": result.get("original_query", text),
        "react": react_output,
    }
