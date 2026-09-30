"""Which LLM to call, from CLI flags, then environment, then defaults.

Providers: `openai` (any OpenAI-compatible endpoint), `anthropic` (native
Messages API), `litellm` (a LiteLLM proxy), and `orcarouter` (the
OrcaRouter gateway). Settings come from `--provider/--model/--api-key/
--base-url`, else `BYTELY_PROVIDER/BYTELY_MODEL/BYTELY_API_KEY/
BYTELY_BASE_URL`. For convenience a provider's own conventional key
variable is also read (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`, `ORCAROUTER_API_KEY`); an OpenRouter key alone points
the `openai` provider at OpenRouter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bytely.ai.llm.anthropic import AnthropicChatModel
from bytely.ai.llm.openai import OpenAIChatModel

if TYPE_CHECKING:
    from bytely.ai.llm.types import ChatModel

PROVIDERS = ("openai", "anthropic", "litellm", "orcarouter")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"
LITELLM_BASE_URL = "http://localhost:4000"
# Gateways route by provider-prefixed ids; OpenAI itself does not.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "openai-gateway": "openai/gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
    "litellm": "openai/gpt-4o-mini",
    "orcarouter": "openai/gpt-4o-mini",
}


class ConfigError(ValueError):
    """The LLM settings are unusable (an unknown provider)."""


@dataclass(frozen=True)
class LLMConfig:
    """Resolved settings for the LLM pass."""

    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    headers: dict[str, str]
    key_source: str | None  # where the key came from, for messages


def _first(*pairs: tuple[str, str | None]) -> tuple[str | None, str | None]:
    for source, value in pairs:
        if value:
            return value, source
    return None, None


def resolve_config(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMConfig:
    """Merge explicit settings with the environment and defaults."""
    env = os.environ
    chosen = (provider or env.get("BYTELY_PROVIDER") or "openai").lower()
    if chosen not in PROVIDERS:
        raise ConfigError(
            f"Unknown provider {chosen!r}; "
            f"choose one of: {', '.join(PROVIDERS)}"
        )
    conventional = {
        "openai": ["OPENAI_API_KEY", "OPENROUTER_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "litellm": ["LITELLM_API_KEY"],
        "orcarouter": ["ORCAROUTER_API_KEY"],
    }[chosen]
    key, key_source = _first(
        ("--api-key", api_key),
        ("BYTELY_API_KEY", env.get("BYTELY_API_KEY")),
        *((name, env.get(name)) for name in conventional),
    )
    url = base_url or env.get("BYTELY_BASE_URL")
    if not url and chosen == "openai" and key_source == "OPENROUTER_API_KEY":
        url = OPENROUTER_BASE_URL
    if not url and chosen == "orcarouter":
        url = ORCAROUTER_BASE_URL
    if not url and chosen == "litellm":
        url = LITELLM_BASE_URL
    default = DEFAULT_MODELS[chosen]
    if chosen == "openai" and url and "api.openai.com" not in url:
        default = DEFAULT_MODELS["openai-gateway"]
    headers = {"x-title": "bytely"} if url and "openrouter.ai" in url else {}
    return LLMConfig(
        provider=chosen,
        model=model or env.get("BYTELY_MODEL") or default,
        api_key=key,
        base_url=url,
        headers=headers,
        key_source=key_source,
    )


def needs_key(config: LLMConfig) -> bool:
    """Whether calls will fail for want of a key (a local proxy may not)."""
    if config.api_key:
        return False
    local = config.base_url and any(
        host in config.base_url for host in ("localhost", "127.0.0.1", "[::1]")
    )
    return not local


def create_chat_model(config: LLMConfig) -> ChatModel:
    """The client for the configured provider."""
    if config.provider == "anthropic":
        return AnthropicChatModel(config.model, config.api_key, config.base_url)
    return OpenAIChatModel(
        config.model,
        config.api_key,
        config.base_url,
        config.headers,
        label=f"{config.provider}:{config.model}",
    )
