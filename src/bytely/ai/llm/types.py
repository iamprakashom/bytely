"""The provider-neutral chat shape every LLM call goes through.

Nothing above this layer knows which provider is in play: adapters
translate `ChatRequest` to a provider's wire format and its reply back to
`ChatResponse`, so adding a provider is a new adapter, never a change to a
call site.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One chat message."""

    role: Role
    content: str


@dataclass(frozen=True)
class Tool:
    """A function the model may (or must) call; `parameters` is JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ChatRequest:
    """One completion request.

    `force_tool` names a tool the model must call: the one structured-output
    mechanism every provider shares.
    """

    messages: list[Message]
    tools: list[Tool] = field(default_factory=list)
    force_tool: str | None = None
    temperature: float | None = 0.0
    max_tokens: int = 4096


@dataclass(frozen=True)
class ToolCall:
    """A tool call in the reply, with its arguments parsed."""

    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    """Token counts; `input` is uncached input only, as for every provider."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_create: int = 0


@dataclass(frozen=True)
class ChatResponse:
    """The model's reply."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str | None
    usage: Usage = field(default_factory=Usage)


class ChatModel(Protocol):
    """Anything that answers a `ChatRequest`."""

    label: str

    def create(self, request: ChatRequest) -> ChatResponse:
        """Send one request and return the reply."""
        ...


class LLMError(RuntimeError):
    """A request failed; the message carries the provider's own error text."""

    def __init__(self, message: str, status: int | None = None) -> None:
        """Keep the HTTP status (None for a network failure)."""
        super().__init__(message)
        self.status = status


def transport_retries() -> int:
    """Retries for a failed request (429, 5xx, network): `BYTELY_LLM_RETRIES`.

    Enough to ride out a shared gateway's rate limit; a metered gateway may
    want fewer, a flaky local proxy more.
    """
    raw = os.environ.get("BYTELY_LLM_RETRIES", "")
    return int(raw) if raw.isdigit() else 4
