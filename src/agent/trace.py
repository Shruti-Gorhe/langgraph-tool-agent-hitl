"""Helpers for displaying observable agent/tool traces without exposing CoT."""


def extract_trace(result) -> list[dict]:
    """Extract model tool calls and tool results from an agent GraphOutput."""
    value = getattr(result, "value", result)
    messages = value.get("messages", []) if isinstance(value, dict) else []
    trace = []

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            trace.append(
                {
                    "type": "tool_call",
                    "tool": call.get("name"),
                    "arguments": call.get("args", {}),
                }
            )

        name = getattr(message, "name", None)
        if name:
            content = getattr(message, "content", "")
            trace.append(
                {
                    "type": "tool_result",
                    "tool": name,
                    "result": content,
                }
            )

    return trace


def final_text(result) -> str:
    value = getattr(result, "value", result)
    messages = value.get("messages", []) if isinstance(value, dict) else []
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and getattr(message, "content", None):
            return message.content
    return ""
