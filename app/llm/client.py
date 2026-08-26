from openai import OpenAI
from openai import BadRequestError
from config import groq_key, model_name
from llm.schemas import Message

# Groq error code for the failure mode this module works around: a
# tool-trained model (the gpt-oss/Harmony family in particular) drops
# into its native tool-call completion channel even when the request
# effectively asked for plain/JSON text, and Groq rejects the
# generation server-side rather than returning it. Detected via
# BadRequestError.body["error"]["code"], not by string-matching the
# message text, since the message wording isn't a documented contract.
_TOOL_USE_LEAK_CODE = "tool_use_failed"
_JSON_VALIDATE_FAILED_CODE = "json_validate_failed"
_RETRYABLE_CODES = {_TOOL_USE_LEAK_CODE, _JSON_VALIDATE_FAILED_CODE}


class LLMResponse:
    def __init__(self, message):
        self._message = message
        self.content = message.content
        self.tool_calls = getattr(message, "tool_calls", None) or []

    def has_tool_call(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> dict:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": self.tool_calls,
        }


class LLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def _dump(self, m):
        return m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m

    @staticmethod
    def _is_retryable_generation_error(err: BadRequestError) -> bool:
        body = getattr(err, "body", None)
        if not isinstance(body, dict):
            return False
        error = body.get("error")
        return isinstance(error, dict) and error.get("code") in _RETRYABLE_CODES

    def chat(
        self,
        messages,
        tools=None,
        model_name=model_name,
        response_format=None,
        _retry_on_tool_leak: bool = True,
    ) -> LLMResponse:
        """
        Chat with the LLM.

        response_format: pass {"type": "json_object"} to force JSON-mode
        output (e.g. for planner.decide()). Some Groq models (the
        Harmony/gpt-oss family in particular) are heavily tool-trained
        and will drop into a synthetic tool-call channel when merely
        *asked* in the prompt to return JSON — even with tools=None —
        which the API then rejects as "Tool choice is none, but model
        called a tool". JSON mode constrains the output channel
        directly instead of relying on the model to follow prompt
        instructions, which substantially reduces that failure mode —
        but does NOT eliminate it; it's a per-generation quirk, not a
        deterministic request problem, so a second identical request
        commonly succeeds where the first didn't. This method retries
        once automatically when that specific error is detected
        (_is_tool_use_leak) before giving up and letting the error
        propagate — callers shouldn't need to know this retry happens,
        only that chat() either returns a real LLMResponse or raises.
        """
        dumped = [self._dump(m) for m in messages]
        kwargs = {
            "messages": dumped,
            "model": model_name,
        }
        # Only include "tools" when actually supplied — sending an
        # explicit `tools: null` is unnecessary and, per the failure
        # mode above, one less variable worth ruling out server-side.
        if tools:
            kwargs["tools"] = tools
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if _retry_on_tool_leak and self._is_retryable_generation_error(e):
                return self.chat(
                    messages,
                    tools=tools,
                    model_name=model_name,
                    response_format=response_format,
                    _retry_on_tool_leak=False,
                )
            raise

        if not response.choices:
            raise ValueError("No response choices returned from Groq API")
        return LLMResponse(response.choices[0].message)

    def complete(self, prompt: str, system: str = "") -> str:
        """
        Single-turn convenience wrapper over LLM.chat() for callers that
        just want plain text in, plain text out — e.g. summarizer.py,
        which doesn't need full conversation-message plumbing.
        """
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))

        return self.chat(messages).content   # <-- unwrap here