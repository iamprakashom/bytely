"""OpenAI-compatible transport (`/chat/completions`).

Works with any OpenAI-compatible endpoint: OpenAI, OpenRouter, Fireworks,
Groq, Together, DeepSeek, a LiteLLM proxy, OrcaRouter, or a local server;
the user picks it with the base URL. A forced tool is sent as an
object-form `tool_choice`.

Some servers reject parts of the request that others require, each with a
recognisable 400. Each is fixed once and retried, never guessed from a
model name:

- reasoning models refuse function tools while reasoning is on, and name
  the fix: `reasoning_effort: "none"` (DeepSeek words it "thinking mode");
- some local servers reject an object-form `tool_choice`; with a single
  tool, `"required"` means the same thing;
- reasoning models want `max_completion_tokens` instead of `max_tokens`;
- the same models fix temperature, so it is dropped.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bytely.ai.llm.http import post_json
from bytely.ai.llm.types import (
    ChatRequest,
    ChatResponse,
    LLMError,
    ToolCall,
    Usage,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _rejects(error: LLMError, pattern: str) -> bool:
    return error.status == 400 and bool(
        re.search(pattern, str(error), re.IGNORECASE)
    )


def _parse_args(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw) if isinstance(raw, str) and raw else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


class OpenAIChatModel:
    """A `ChatModel` over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        label: str | None = None,
    ) -> None:
        """Point at `base_url` (default: OpenAI) with `api_key`."""
        self.model = model
        self.url = (base_url or DEFAULT_BASE_URL).rstrip(
            "/"
        ) + "/chat/completions"
        self.headers = dict(headers or {})
        if api_key:
            self.headers["authorization"] = f"Bearer {api_key}"
        self.label = label or f"openai:{model}"

    def payload(self, request: ChatRequest) -> dict[str, Any]:
        """The request body for `request`."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages
            ],
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
        if request.force_tool:
            params["tool_choice"] = {
                "type": "function",
                "function": {"name": request.force_tool},
            }
        return params

    def create(self, request: ChatRequest) -> ChatResponse:
        """Send the request, adapting it to the server's known refusals."""
        params = self.payload(request)
        for _ in range(5):  # at most one retry per known refusal
            try:
                return self._response(post_json(self.url, params, self.headers))
            except LLMError as error:
                fixed = _adapt(params, error)
                if fixed is None:
                    raise
                params = fixed
        return self._response(post_json(self.url, params, self.headers))

    @staticmethod
    def _response(data: dict[str, Any]) -> ChatResponse:
        choices = data.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        calls = [
            ToolCall(
                str(call.get("function", {}).get("name", "")),
                _parse_args(call.get("function", {}).get("arguments")),
            )
            for call in message.get("tool_calls") or []
            if isinstance(call, dict)
        ]
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        )
        return ChatResponse(
            text=str(message.get("content") or ""),
            tool_calls=calls,
            stop_reason=choice.get("finish_reason"),
            usage=Usage(
                input=max(0, prompt - cached),
                output=int(usage.get("completion_tokens") or 0),
                cache_read=cached,
            ),
        )


def _adapt(params: dict[str, Any], error: LLMError) -> dict[str, Any] | None:
    """The request fixed for a known refusal, or None if it is not one."""
    # Before the tool_choice fallback: this refusal also names tool_choice,
    # and turning reasoning off keeps the caller's chosen tool.
    if (
        _rejects(
            error,
            r"function tools with reasoning_effort"
            r"|thinking mode does not support this tool_choice",
        )
        and "reasoning_effort" not in params
    ):
        return {**params, "reasoning_effort": "none"}
    if (
        _rejects(error, r"tool_choice")
        and isinstance(params.get("tool_choice"), dict)
        and len(params.get("tools") or []) == 1
    ):
        return {**params, "tool_choice": "required"}
    if (
        _rejects(error, r"max_tokens.*not supported.*max_completion_tokens")
        and "max_tokens" in params
    ):
        rest = {k: v for k, v in params.items() if k != "max_tokens"}
        return {**rest, "max_completion_tokens": params["max_tokens"]}
    if _rejects(error, r"temperature.*does not support") and (
        "temperature" in params
    ):
        return {k: v for k, v in params.items() if k != "temperature"}
    return None
