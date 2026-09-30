"""Native Anthropic transport (the Messages API).

Absorbs what differs from Chat Completions: `system` is a top-level
parameter, `max_tokens` is required, a forced tool is
`tool_choice: {type: "tool"}`, and temperature is never sent (current models
reject it).
"""

from __future__ import annotations

from typing import Any

from bytely.ai.llm.http import post_json
from bytely.ai.llm.types import ChatRequest, ChatResponse, ToolCall, Usage

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicChatModel:
    """A `ChatModel` over Anthropic's Messages API."""

    def __init__(
        self, model: str, api_key: str | None, base_url: str | None = None
    ) -> None:
        """Point at `base_url` (default: Anthropic) with `api_key`."""
        self.model = model
        base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.url = base + (
            "/messages" if base.endswith("/v1") else "/v1/messages"
        )
        self.headers = {"anthropic-version": API_VERSION}
        if api_key:
            self.headers["x-api-key"] = api_key
        self.label = f"anthropic:{model}"

    def payload(self, request: ChatRequest) -> dict[str, Any]:
        """The request body for `request`."""
        system = "\n\n".join(
            m.content for m in request.messages if m.role == "system"
        )
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
                if m.role != "system"
            ],
        }
        if system:
            params["system"] = system
        if request.tools:
            params["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in request.tools
            ]
        if request.force_tool:
            params["tool_choice"] = {"type": "tool", "name": request.force_tool}
        return params

    def create(self, request: ChatRequest) -> ChatResponse:
        """Send the request and return the reply."""
        data = post_json(self.url, self.payload(request), self.headers)
        text = ""
        calls = []
        for block in data.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text += str(block.get("text", ""))
            elif block.get("type") == "tool_use":
                args = block.get("input")
                calls.append(
                    ToolCall(
                        str(block.get("name", "")),
                        args if isinstance(args, dict) else {},
                    )
                )
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text,
            tool_calls=calls,
            stop_reason=data.get("stop_reason"),
            usage=Usage(
                input=int(usage.get("input_tokens") or 0),
                output=int(usage.get("output_tokens") or 0),
                cache_read=int(usage.get("cache_read_input_tokens") or 0),
                cache_create=int(usage.get("cache_creation_input_tokens") or 0),
            ),
        )
