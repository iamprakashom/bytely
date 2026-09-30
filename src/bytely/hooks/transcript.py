"""What the Stop hook reads from a host's session transcript (JSONL).

Only the tail is read: the turn that just ended is at the end, and a long
session's transcript can be large. None always means "not observed" (no
path, an unreadable file, nothing billable), never zero, so a host whose
hooks name no transcript reports no rate rather than a wrong one.
Subagent (sidechain) entries are skipped: their prose is not what the
user read, and their usage is billed in their own transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bytely.hooks.savings import Usage

TAIL_BYTES = 1024 * 1024
_TALLY = re.compile(
    r"bytely\s+saved\s*[~≈]?\s*[\d,.]+\s*[km]?\s*(?:tok|tokens)", re.IGNORECASE
)


def has_savings_tally(text: str) -> bool:
    """Whether a reply reported the turn's bytely savings."""
    return bool(_TALLY.search(text))


def _tail_entries(path: object) -> list[dict[str, Any]]:
    if not isinstance(path, str) or not path:
        return []
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            length = min(size, TAIL_BYTES)
            handle.seek(size - length)
            raw = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    if length < size:  # the first line is probably cut
        raw = raw[raw.find("\n") + 1 :]
    entries = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _message(entry: dict[str, Any]) -> dict[str, Any]:
    message = entry.get("message")
    return message if isinstance(message, dict) else {}


def _is_user_prompt(entry: dict[str, Any]) -> bool:
    """A prompt the user typed (not a tool result fed back to the model)."""
    if entry.get("type") != "user" or entry.get("isMeta"):
        return False
    content = _message(entry).get("content")
    if isinstance(content, str):
        return True
    return isinstance(content, list) and not any(
        isinstance(part, dict) and part.get("type") == "tool_result"
        for part in content
    )


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _current_turn(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The main-thread assistant entries since the last user prompt."""
    turn = []
    for entry in reversed(entries):
        if entry.get("isSidechain"):
            continue
        if _is_user_prompt(entry):
            break
        if entry.get("type") == "assistant":
            turn.append(entry)
    turn.reverse()
    return turn


@dataclass(frozen=True)
class Turn:
    """The assistant prose of the last turn."""

    uuid: str
    text: str


def last_assistant_turn(transcript: object) -> Turn | None:
    """The prose the user read at the end of the transcript."""
    parts = []
    uuid = None
    for entry in _current_turn(_tail_entries(transcript)):
        text = _text_of(_message(entry).get("content"))
        if not text.strip():
            continue
        if isinstance(entry.get("uuid"), str):
            uuid = entry["uuid"]
        parts.append(text)
    if uuid is None or not parts:
        return None
    return Turn(uuid, "\n".join(parts))


@dataclass(frozen=True)
class Billing:
    """What the last turn's input cost."""

    uuid: str
    cost_micros: int
    tokens: int


def _count(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def last_turn_billing(transcript: object) -> Billing | None:
    """The last turn's input cost, priced per API response.

    One response is written as several entries that repeat the same
    `usage`, so responses are counted once each, by message id. A model
    with no known price adds neither cost nor tokens: its tokens alone
    would drag the blended rate toward zero.
    """
    seen: set[str] = set()
    uuid = None
    cost = 0
    tokens = 0
    for entry in reversed(_current_turn(_tail_entries(transcript))):
        if uuid is None and isinstance(entry.get("uuid"), str):
            uuid = entry["uuid"]
        message = _message(entry)
        message_id = message.get("id")
        usage = message.get("usage")
        if not isinstance(message_id, str) or message_id in seen:
            continue
        if not isinstance(usage, dict):
            continue
        seen.add(message_id)
        turn = Usage(
            str(message.get("model", "")),
            _count(usage.get("input_tokens")),
            _count(usage.get("cache_creation_input_tokens")),
            _count(usage.get("cache_read_input_tokens")),
        )
        micros = turn.cost_micros()
        if micros is None:
            continue
        cost += micros
        tokens += turn.tokens
    if uuid is None or tokens == 0:
        return None
    return Billing(uuid, cost, tokens)
