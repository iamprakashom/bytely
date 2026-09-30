"""Per-session accounting: graph reads versus source reads, tokens saved.

A tool call is a bytely read when it is one of bytely's MCP tools, or a
shell command that runs `bytely`; it is a source read when it is the
host's own Read/Grep/Glob/Search. Savings come from the header lines in
the tool's output (see `savings.sum_savings`).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from bytely.hooks.savings import dollars_saved, format_dollars, sum_savings
from bytely.hooks.state import latest_session, update_session

if TYPE_CHECKING:
    from pathlib import Path

Kind = Literal["bytely", "source"]

MCP_TOOL_NAMES = frozenset(
    {
        "bytely_find_code",
        "bytely_find_all",
        "bytely_trace_calls",
        "bytely_file_api",
        "bytely_repo_map",
        "bytely_check_freshness",
    }
)
_SOURCE_TOOLS = frozenset({"read", "grep", "glob", "search"})
_SHELL_TOOLS = frozenset({"bash", "shell", "powershell"})
_INVOKES = re.compile(
    r"(^|[|&;(]\s*)((uv|uvx|pipx)\s+(run\s+)?)?"
    r"(python3?\s+-m\s+)?bytely(\.exe)?\b",
    re.IGNORECASE,
)


def is_mcp_tool_name(tool: str) -> bool:
    """Whether a host tool name is an MCP tool (any server)."""
    name = tool.lower()
    return name.startswith("mcp") or ":" in name or "__" in name


def is_bytely_mcp_tool(tool: str) -> bool:
    """Whether a tool name is one of bytely's, whatever the host prefix."""
    bare = re.split(r"[:./]|__", tool.lower())[-1]
    return bare in MCP_TOOL_NAMES


def command_invokes_bytely(command: str) -> bool:
    """Whether a shell command runs the bytely CLI."""
    return bool(_INVOKES.search(command.strip()))


def classify_tool_use(tool: object, command: object = None) -> Kind | None:
    """`bytely`, `source`, or None for a call that is neither."""
    name = str(tool or "").lower()
    if not name:
        return None
    if is_bytely_mcp_tool(name):
        return "bytely"
    if name in _SOURCE_TOOLS:
        return "source"
    if (
        name in _SHELL_TOOLS
        and isinstance(command, str)
        and command_invokes_bytely(command)
    ):
        return "bytely"
    return None


def score_tool_use(
    tool: object, command: object, output: object
) -> tuple[Kind | None, int]:
    """The call's kind, and the savings its output reports.

    A source read reports none by definition. Anything else whose output
    carries a savings line was a bytely read, whatever the tool was
    called (a wrapper script, an alias).
    """
    kind = classify_tool_use(tool, command)
    if kind == "source":
        return kind, 0
    text = output if isinstance(output, str) else _flatten(output)
    saved = sum_savings(text)
    if saved > 0:
        kind = "bytely"
    return kind, saved


def _flatten(value: object) -> str:
    import json

    try:
        # ensure_ascii=False keeps `≈` literal for the savings pattern.
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def record_tool_use(
    project: Path,
    session_id: str,
    kind: Kind | None,
    saved: int = 0,
    host: str | None = None,
) -> None:
    """Count one call in the session's totals."""
    if kind is None and saved <= 0:
        return
    with update_session(project, session_id) as session:
        if kind == "bytely":
            session["bytelyReads"] = int(session.get("bytelyReads", 0)) + 1
            session["turnUsedBytely"] = True
        elif kind == "source":
            session["sourceReads"] = int(session.get("sourceReads", 0)) + 1
        if saved > 0:
            session["savedTokens"] = int(session.get("savedTokens", 0)) + saved
        if host and not session.get("host"):
            session["host"] = host


def session_input_rate(project: Path) -> float | None:
    """The latest session's blended input price, in $ per million tokens."""
    session = latest_session(project)
    if not session:
        return None
    cost = session.get("inputCostMicros")
    tokens = session.get("inputTokensBilled")
    if not isinstance(cost, (int, float)) or not isinstance(
        tokens, (int, float)
    ):
        return None
    if cost <= 0 or tokens <= 0:
        return None
    # Micro-dollars per token and dollars per million tokens are equal.
    return cost / tokens


def format_session_stats(session: dict[str, Any] | None) -> str:
    """The `bytely stats` report."""
    if session is None:
        return (
            "bytely stats: no session recorded yet — use bytely in an agent "
            "session, then look again."
        )
    graph_reads = int(session.get("bytelyReads", 0))
    source_reads = int(session.get("sourceReads", 0))
    saved = int(session.get("savedTokens", 0))
    total = graph_reads + source_reads
    mix = (
        "no retrieval yet"
        if total == 0
        else f"{round(graph_reads / total * 100)}% bytely"
    )
    lines = [
        f"bytely stats — session {session.get('id', 'default')}",
        f"  bytely reads:  {graph_reads}",
        f"  source reads:  {source_reads}   (Read / Grep / Glob)",
        f"  mix:           {mix}",
        f"  tokens saved:  ~{saved:,}",
    ]
    usd = dollars_saved(
        saved, session.get("inputCostMicros"), session.get("inputTokensBilled")
    )
    if usd is not None:
        lines.append(f"  value saved:   ~{format_dollars(usd)}")
    turns = int(session.get("bytelyTurns", 0))
    if turns:
        reported = int(session.get("reportedTurns", 0))
        lines.append(f"  tally shown:   {reported}/{turns} turns using bytely")
    if session.get("lastQuery"):
        lines.append(f"  last query:    {session['lastQuery']}")
    return "\n".join(lines)
