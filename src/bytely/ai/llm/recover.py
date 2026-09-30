"""Recover a forced tool call's arguments from plain reply text.

Some OpenAI-compatible gateways ignore a forced `tool_choice` and write the
payload as text instead: a bare object, fenced JSON, or an emulated
`[{name, parameters}]` array. Without this, a structured pass sees no tool
call and records nothing. Conservative: only JSON whose shape matches is
accepted, and callers still validate it like a real tool call.
"""

from __future__ import annotations

import json
from typing import Any


def _unfence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        if stripped[:4].lower() == "json":
            stripped = stripped[4:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _payload(
    value: object, tool_names: set[str], key: str
) -> dict[str, Any] | None:
    if isinstance(value, list):
        for item in value:
            hit = _payload(item, tool_names, key)
            if hit is not None:
                return hit
        return None
    if not isinstance(value, dict):
        return None
    wrapped = isinstance(value.get("name"), str) or any(
        field in value for field in ("parameters", "arguments", "args")
    )
    if wrapped:
        name = value.get("name")
        if isinstance(name, str) and name and name not in tool_names:
            return None
        params = value.get(
            "parameters", value.get("arguments", value.get("args"))
        )
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except ValueError:
                return None
        hit = _payload(params, tool_names, key)
        if hit is not None:
            return hit
    return value if isinstance(value.get(key), list) else None


def recover_tool_args(
    text: str | None, tool_names: list[str], key: str
) -> dict[str, Any] | None:
    """The tool payload in `text` whose `key` is a list, if there is one."""
    raw = (text or "").strip()
    if not raw:
        return None
    body = _unfence(raw)
    names = set(tool_names)
    candidates = [body]
    for open_, close in (("{", "}"), ("[", "]")):
        start, end = body.find(open_), body.rfind(close)
        if 0 <= start < end:
            candidates.append(body[start : end + 1])
    for candidate in candidates:
        try:
            hit = _payload(json.loads(candidate), names, key)
        except ValueError:
            continue
        if hit is not None:
            return hit
    return None
