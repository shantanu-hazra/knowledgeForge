"""
tool_call_handler.py

    Receives                          Calls              Returns
    {"name": "calculator",     -->    calculator.run()   -->   13
     "arguments": {"a": 5, "b": 8}}

Bridges the raw tool_call object the LLM API returns to the existing
ToolDispatcher (tools/registry.py). No new dispatch logic — this is
just the glue that: parses the tool_call, invokes dispatcher.call(),
and formats the result as a `tool` role message to append back onto
the conversation for the next LLM turn.
"""

import json
from dataclasses import dataclass

from tools.registry import ToolDispatcher, TOOLS
from llm.schemas import Message

dispatcher = ToolDispatcher()


@dataclass
class ToolCall:
    """Normalized shape of one tool call, regardless of API wire format."""
    id: str
    name: str
    arguments: dict

def _to_dict(obj) -> dict:
    """
    Coerce a tool_call (or its nested .function) to a plain dict,
    regardless of whether the SDK handed us a dict or a Pydantic
    model object (e.g. ChatCompletionMessageFunctionToolCall).
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):      # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):            # pydantic v1
        return obj.dict()
    return {k: getattr(obj, k) for k in ("id", "type", "function", "name", "arguments") if hasattr(obj, k)}


def parse_tool_call(raw: dict) -> ToolCall:
    """
    Normalize a raw tool_call block from the LLM response into a
    ToolCall. Handles the common OpenAI/Groq-style shape:

        {
          "id": "call_abc123",
          "type": "function",
          "function": {"name": "...", "arguments": "{...json string...}"}
        }

    Arguments may arrive as a JSON string (OpenAI) or already as a
    dict (some providers) — handle both.
    """
    raw = _to_dict(raw)
    fn = raw.get("function", raw)   # tolerate flat {"name", "arguments"} too
    fn = _to_dict(fn)
    args = fn["arguments"]

    if isinstance(args, str):
        args = json.loads(args)

    return ToolCall(
        id=raw.get("id", ""),
        name=fn["name"],
        arguments=args,
    )


def handle_tool_call(raw: dict) -> Message:
    """
    Full receive -> call -> return cycle for one tool call.

        receives {"name": "calculator", "arguments": {"a": 5, "b": 8}}
        calls    calculator.run()   (i.e. dispatcher.call("calculator", ...))
        returns  a `tool` role Message ready to append to context,
                 containing the JSON-encoded result (e.g. "13")
    """

    call = parse_tool_call(raw)

    # ToolDispatcher.call() already: validates args against the pydantic
    # model, invokes the underlying fn, and JSON-encodes the result
    # (or an {"error": ...} payload on failure) — see tools/registry.py
    result_json = dispatcher.call(call.name, call.arguments)

    print(f"[tool_call] {call.name}({call.arguments})")

    return Message(
        role="tool",
        tool_call_id=call.id,
        name=call.name,
        content=result_json,
    )


def handle_tool_calls(raw_tool_calls: list[dict]) -> list[Message]:
    """Convenience for the common case: an LLM turn requests several tools at once."""
    return [handle_tool_call(raw) for raw in raw_tool_calls]