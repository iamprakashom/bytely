"""Per-symbol summaries and crux spans, one LLM call per file.

The model sees the whole file with line numbers and a list of target
definitions, and records, for every target, a one-sentence purpose and the
line range of its most important few lines (0/0 when there is none). The
reply comes through a forced tool call; text that carries the same payload
is recovered when a gateway ignores the forced tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bytely.ai.llm.recover import recover_tool_args
from bytely.ai.llm.types import ChatRequest, Message, Tool

if TYPE_CHECKING:
    from bytely.ai.llm.types import ChatModel, ChatResponse

SYSTEM_PROMPT = """\
You explain code definitions for a code graph that helps engineers navigate a codebase.

You are given ONE source file with 1-based line numbers, and a list of TARGET definitions in it. Describe EVERY target via the record_symbols tool.

Rules:
- Return EXACTLY ONE entry for EVERY target id, using that id verbatim. The number of entries you return MUST equal the number of targets. Never omit a target: a reply missing any id is invalid and will be re-requested.
- A trivial symbol is NOT an exception. You still return it — with a one-sentence summary and crux 0/0 (see below). "Skip" means "give it no crux span", NEVER "leave it out".
- summary: ONE sentence — what the symbol is FOR at the business-logic level (the problem it solves or the rule it enforces), not a restatement of its signature.
- crux_start / crux_end: FILE line numbers (as shown), inside that symbol's own line range. Pick the SINGLE most important contiguous span — the core branch, formula, guard, or state change — at most ~8 lines, and NEVER the whole function. When there is no single focal span (a trivial getter, a plain data holder, a one-line delegation, or logic spread evenly), use crux_start: 0 and crux_end: 0. That 0/0 IS the answer — do not drop the entry."""  # noqa: E501

TOOL_NAME = "record_symbols"
TOOL = Tool(
    TOOL_NAME,
    "Record each target definition's purpose and crux line range.",
    {
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "summary": {"type": "string"},
                        "crux_start": {"type": "number"},
                        "crux_end": {"type": "number"},
                    },
                    "required": ["id", "summary", "crux_start", "crux_end"],
                },
            }
        },
        "required": ["symbols"],
    },
)
MAX_CODE_CHARS = 18_000


@dataclass(frozen=True)
class Target:
    """A definition to describe."""

    id: str
    kind: str
    start: int
    end: int
    signature: str | None = None


@dataclass(frozen=True)
class SymbolNote:
    """What the model recorded for one target."""

    id: str
    summary: str
    crux_start: int
    crux_end: int


def _numbered(source: str) -> str:
    if len(source) > MAX_CODE_CHARS:
        source = source[:MAX_CODE_CHARS] + "\n… (truncated)"
    return "\n".join(
        f"{number}\t{line}"
        for number, line in enumerate(source.split("\n"), start=1)
    )


def user_content(path: str, source: str, targets: list[Target]) -> str:
    """The user message: the numbered file, then the targets."""
    listed = "\n".join(
        f"- id={t.id} | {t.kind} | lines L{t.start}-L{t.end}"
        + (f" | {' '.join(t.signature.split())}" if t.signature else "")
        for t in targets
    )
    count = len(targets)
    return (
        f"FILE: {path}\n\n{_numbered(source)}\n\n"
        f"TARGETS ({count} — return all {count}, one entry per id):\n{listed}"
    )


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def parse_notes(args: dict[str, Any] | None) -> list[SymbolNote]:
    """The notes in a tool payload, skipping malformed entries."""
    if not args or not isinstance(args.get("symbols"), list):
        return []
    return [
        SymbolNote(
            entry["id"],
            str(entry.get("summary") or "").strip(),
            _int(entry.get("crux_start")),
            _int(entry.get("crux_end")),
        )
        for entry in args["symbols"]
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


def tool_args(
    response: ChatResponse, tool: str, key: str
) -> dict[str, Any] | None:
    """The forced tool's arguments, or the same payload recovered from text."""
    named = [c for c in response.tool_calls if c.name == tool]
    calls = named or response.tool_calls
    call = calls[0] if calls else None
    if call is not None and call.args:
        return call.args
    return recover_tool_args(response.text, [tool, "emit_json"], key)


def miss_reason(response: ChatResponse, notes: list[SymbolNote]) -> str | None:
    """Why a reply gave nothing usable (None if at least one summary)."""
    if any(note.summary for note in notes):
        return None
    stop = response.stop_reason or "null"
    if stop.lower() in ("length", "max_tokens"):
        kind = "truncated"
    elif notes:
        kind = "empty-parsed"
    elif not response.tool_calls and not response.text.strip():
        kind = "empty-reply"
    else:
        kind = "unparseable"
    return (
        f"model returned no usable symbol summaries [{kind}, "
        f"finish_reason={stop}]"
    )


class CruxSummarizer:
    """Describes a file's definitions with one forced-tool call."""

    def __init__(self, model: ChatModel) -> None:
        """Use `model` for every call."""
        self.model = model

    def describe(
        self, path: str, source: str, targets: list[Target]
    ) -> tuple[list[SymbolNote], str | None]:
        """Notes for the targets, and why the reply was unusable, if it was."""
        if not targets:
            return [], None
        response = self.model.create(
            ChatRequest(
                messages=[
                    Message("system", SYSTEM_PROMPT),
                    Message("user", user_content(path, source, targets)),
                ],
                tools=[TOOL],
                force_tool=TOOL_NAME,
                max_tokens=8192,
            )
        )
        notes = parse_notes(tool_args(response, TOOL_NAME, "symbols"))
        return notes, miss_reason(response, notes)
