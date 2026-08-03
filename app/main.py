"""FastAPI surface for the camping-store shopping assistant.

Resource-shaped, following LangGraph Platform: create a thread, then post runs to it.
Threads belong to a user, and memory is keyed by that user rather than the thread, so a
returning customer keeps their profile, trip and shortlists.
"""

from app.env import load_env_file

# Must run before the langchain/langgraph imports below — see app/graph.py for why.
load_env_file()

import json  # noqa: E402
import uuid  # noqa: E402
from typing import Any  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app import memory  # noqa: E402
from app.formatting import build_react_output, normalize_content  # noqa: E402
from app.graph import api_graph, build_initial_state, store  # noqa: E402

app = FastAPI(title="Camping Store Shopping Assistant", version="2.0.0")


class ThreadPayload(BaseModel):
    user_id: str


class RunCommand(BaseModel):
    """Mirrors LangGraph's Command — the value handed back to a paused interrupt()."""

    resume: dict[str, Any]


class RunPayload(BaseModel):
    # Exactly one of these: `input` starts or continues a conversation,
    # `command` resumes a thread paused at an interrupt.
    input: str | None = None
    command: RunCommand | None = None


# Threads live only as long as the process, same as the InMemorySaver backing them.
# Tracking them explicitly is what lets an unknown thread 404 instead of silently
# behaving like a brand-new one, and is where a thread's owner is recorded.
_threads: dict[str, str] = {}


def _pending_interrupts(snapshot: Any) -> list[Any]:
    """Interrupt payloads the thread is currently waiting on."""
    return [
        getattr(intr, "value", intr)
        for task in (snapshot.tasks or ())
        for intr in (getattr(task, "interrupts", None) or ())
    ]


def _thread_config(thread_id: str) -> RunnableConfig:
    return {
        "configurable": {"thread_id": thread_id, "user_id": _threads[thread_id]},
    }


def _require_thread(thread_id: str) -> str:
    if thread_id not in _threads:
        raise HTTPException(status_code=404, detail=f"Unknown thread: {thread_id}")
    return _threads[thread_id]


def _stream_graph(graph_input: Any, thread_id: str) -> StreamingResponse:
    """Stream graph execution (new run or Command resume) as Server-Sent Events."""
    config = _thread_config(thread_id)

    def event_generator():
        yield f"data: {json.dumps({'event': 'run_started', 'thread_id': thread_id})}\n\n"

        for chunk in api_graph.stream(graph_input, config=config, stream_mode="updates"):
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

                    # What a memory node just wrote, so a UI can show the record changing
                    memory_update = update.get("last_memory_update")
                    if memory_update:
                        yield f"data: {json.dumps({'event': 'memory', **memory_update})}\n\n"

                    messages = update.get("messages") or []

                    # Compaction returns one RemoveMessage per dropped turn. Those are
                    # bookkeeping, not conversation — a client rendering them would show a
                    # run of blank bubbles. Report the compaction once instead.
                    removals = [m for m in messages if getattr(m, "type", "") == "remove"]
                    if removals:
                        yield f"data: {json.dumps({'event': 'compacted', 'node': node_name, 'dropped': len(removals)})}\n\n"

                    for msg in messages:
                        if getattr(msg, "type", "") == "remove":
                            continue
                        payload: dict[str, Any] = {
                            "event": "message",
                            "node": node_name,
                            "type": getattr(msg, "type", ""),
                            "content": normalize_content(getattr(msg, "content", "")),
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

        final_state = api_graph.get_state(config)
        react_output = build_react_output(dict(final_state.values), thread_id)
        yield f"data: {json.dumps({'event': 'done', 'thread_id': thread_id, 'react': react_output})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/")
def read_root():
    return {"message": "Camping store shopping assistant is running locally!"}


@app.post("/threads", status_code=201)
def create_thread(payload: ThreadPayload):
    """Create a conversation thread for a customer. Runs are posted against its id."""
    thread_id = str(uuid.uuid4())
    _threads[thread_id] = payload.user_id
    return {"thread_id": thread_id, "user_id": payload.user_id}


@app.post("/threads/{thread_id}/runs")
def create_run(thread_id: str, payload: RunPayload):
    """
    Execute the graph on a thread and stream the ReAct trace back as SSE.

    - `{"input": "..."}` starts the conversation, or adds a turn to an existing one.
    - `{"command": {"resume": {...}}}` resumes a thread paused at an interrupt.

    Which one is required depends on whether the thread is currently paused, so the client
    never has to guess — a mismatch comes back as 409 with the reason.
    """
    _require_thread(thread_id)

    config = _thread_config(thread_id)
    snapshot = api_graph.get_state(config)
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
    user_id = _require_thread(thread_id)

    snapshot = api_graph.get_state(_thread_config(thread_id))
    values = dict(snapshot.values)
    interrupts = _pending_interrupts(snapshot)

    return {
        "thread_id": thread_id,
        "user_id": user_id,
        "status": "interrupted" if interrupts else "idle",
        "interrupts": interrupts,
        "summary": values.get("summary", ""),
        "original_query": values.get("original_query", ""),
        "react": build_react_output(values, thread_id),
    }


@app.get("/users/{user_id}/memory")
def get_user_memory(user_id: str):
    """Everything remembered about a customer: profile, trip, and gear needs."""
    return memory.read_all(store, user_id)


@app.delete("/users/{user_id}/memory", status_code=204)
def delete_user_memory(user_id: str) -> None:
    """Forget a customer entirely. Threads keep their transcripts."""
    memory.clear_all(store, user_id)
