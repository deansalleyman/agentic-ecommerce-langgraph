"""Turning graph messages into API-shaped output."""

from typing import Any


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def build_react_output(state: dict[str, Any], thread_id: str) -> dict[str, Any]:
    """The ReAct-style trace of a thread: what was said, called, and answered."""
    messages = state.get("messages", [])
    steps: list[dict[str, Any]] = []

    for message in messages:
        msg_type = getattr(message, "type", None)
        if msg_type == "human":
            steps.append({
                "type": "human",
                "content": normalize_content(getattr(message, "content", "")),
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
                "content": normalize_content(getattr(message, "content", "")),
                "tool_calls": tool_calls,
            })
        elif msg_type == "tool":
            steps.append({
                "type": "tool",
                "name": getattr(message, "name", ""),
                "content": normalize_content(getattr(message, "content", "")),
                "tool_call_id": getattr(message, "tool_call_id", None),
            })

    final_answer = ""
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and not getattr(message, "tool_calls", None):
            final_answer = normalize_content(getattr(message, "content", ""))
            break

    return {"steps": steps, "final_answer": final_answer, "thread_id": thread_id}
