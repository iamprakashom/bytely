"""The MCP tools: each one is a query, answered from the graph.

Every tool except `bytely_check_freshness` refreshes the graph first, so an
agent always sees the code as it is now; when that refresh rebuilt the
graph, the answer starts with a note saying so. `bytely_check_freshness`
reports on the saved graph as it is, which is its whole job. The queries
themselves live in `bytely.query.service`, shared with the CLI.

Arguments a tool does not know are ignored, so a client that sends an
extra field still gets an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bytely.graph.check import check_graph
from bytely.query.service import (
    QueryError,
    Workspace,
    ask_text,
    callers_text,
    grep_text,
    map_text,
    skeleton_text,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Re-exported: the server and its tests build a Workspace from here.
__all__ = ["TOOLS", "TOOLS_BY_NAME", "Tool", "ToolError", "Workspace"]


class ToolError(Exception):
    """A tool could not answer; the message is shown to the agent."""


@dataclass(frozen=True)
class Tool:
    """One MCP tool: its listing and the function that answers it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[Workspace, dict[str, Any]], str]
    refreshes: bool = True

    def listing(self) -> dict[str, Any]:
        """The tool as `tools/list` describes it."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def call(self, workspace: Workspace, arguments: dict[str, Any]) -> str:
        """Answer the call; a query error becomes a `ToolError`."""
        before = workspace.last
        try:
            text = self.handler(workspace, arguments)
        except QueryError as error:
            raise ToolError(str(error)) from error
        refreshed = workspace.last
        if (
            self.refreshes
            and refreshed is not None
            and refreshed is not before
            and refreshed.rebuilt
            and refreshed.build is not None
        ):
            changed = refreshed.build.cache_misses
            text = (
                f"(graph refreshed: {changed} "
                f"file{'' if changed == 1 else 's'} re-parsed)\n" + text
            )
        return text


def _text(arguments: dict[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    return value if isinstance(value, str) and value else None


def _flag(arguments: dict[str, Any], name: str, default: bool) -> bool:
    value = arguments.get(name, default)
    return value if isinstance(value, bool) else default


def _find_code(workspace: Workspace, arguments: dict[str, Any]) -> str:
    return ask_text(
        workspace,
        arguments.get("query"),
        limit=arguments.get("limit", 5),
        source=_flag(arguments, "source", True),
        full=_flag(arguments, "full", False),
        scope=_text(arguments, "in"),
    )


def _find_all(workspace: Workspace, arguments: dict[str, Any]) -> str:
    return grep_text(
        workspace,
        arguments.get("pattern"),
        fixed=_flag(arguments, "fixed", False),
        ignore_case=_flag(arguments, "ignore_case", False),
        scope=_text(arguments, "in"),
    )


def _trace_calls(workspace: Workspace, arguments: dict[str, Any]) -> str:
    return callers_text(
        workspace,
        arguments.get("symbol", arguments.get("file")),
        direction=arguments.get("direction", "in"),
        depth=arguments.get("depth", 1),
        scope=_text(arguments, "in"),
    )


def _file_api(workspace: Workspace, arguments: dict[str, Any]) -> str:
    return skeleton_text(workspace, arguments.get("file"))


def _repo_map(workspace: Workspace, arguments: dict[str, Any]) -> str:
    return map_text(workspace, arguments.get("max_dirs", 12))


def _check_freshness(workspace: Workspace, arguments: dict[str, Any]) -> str:
    fresh, message = check_graph(
        str(workspace.root),
        workspace.context_dir,
        use_cache=not _flag(arguments, "no_cache", False),
    )
    if fresh:
        return f"fresh: {message}\n"
    return (
        f"stale: {message}\nThe other bytely tools refresh the graph "
        "before answering, so no action is needed to query it.\n"
    )


_IN = {
    "type": "string",
    "description": "Only consider code at or under this repo-relative path "
    "prefix, e.g. `server/src`",
}

TOOLS: tuple[Tool, ...] = (
    Tool(
        "bytely_find_code",
        "Query the repository's code graph in plain words, e.g. \"how "
        'does auth work" or "where is rate limiting handled". Returns '
        "ranked definitions with exact file:line spans and each one's "
        "first lines of code (full=true for whole definitions) — usually "
        'the full answer, with no file reads needed. "Who calls X" and '
        '"what does X call" are answered from exact call edges.',
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you want to understand, in plain "
                    "words",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum results (default 5)",
                },
                "source": {
                    "type": "boolean",
                    "description": "Include each result's code (default true)",
                },
                "full": {
                    "type": "boolean",
                    "description": "Show whole definitions instead of the "
                    "first 8 lines",
                },
                "in": _IN,
            },
            "required": ["query"],
        },
        _find_code,
    ),
    Tool(
        "bytely_find_all",
        "Every occurrence of a pattern (regular expression, or a literal "
        "with fixed=true) in the indexed files, grouped by the innermost "
        "enclosing function or class and ordered by how often that code is "
        "referenced. Exhaustive, unlike bytely_find_code, which returns only "
        "the top matches.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "fixed": {
                    "type": "boolean",
                    "description": "Treat pattern as a literal string",
                },
                "ignore_case": {"type": "boolean"},
                "in": _IN,
            },
            "required": ["pattern"],
        },
        _find_all,
    ),
    Tool(
        "bytely_trace_calls",
        "Exact graph edges for a symbol: who calls, extends, or implements "
        'it (direction "in", the default), or what it calls ("out"). '
        'depth>1 walks further; "all" returns the whole connected set — '
        "the blast radius of a change. Run it before a multi-file refactor "
        "to find every affected file. The symbol may be bare (`save`), "
        "qualified (`Cache.save`), a node ID (`src/cache.py#Cache.save`), "
        "or a file path, which covers everything defined in the file.",
        {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "direction": {"type": "string", "enum": ["in", "out"]},
                "depth": {
                    "description": 'Levels to walk (default 1), or "all" '
                    "for the full closure",
                    "oneOf": [
                        {"type": "integer", "minimum": 1},
                        {"type": "string", "enum": ["all", "full"]},
                    ],
                },
                "in": _IN,
            },
            "required": ["symbol"],
        },
        _trace_calls,
    ),
    Tool(
        "bytely_file_api",
        "A file's definitions with their line spans and signatures: its "
        "whole API at a glance, far cheaper than reading the file.",
        {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Repo-relative path, or a unique suffix "
                    "such as `graph/build.py`",
                }
            },
            "required": ["file"],
        },
        _file_api,
    ),
    Tool(
        "bytely_repo_map",
        "A short orientation for an unfamiliar repository: top-level "
        "folders with their most-referenced symbols, and the "
        "most-referenced symbols overall. Use it before diving into files.",
        {
            "type": "object",
            "properties": {
                "max_dirs": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Folders to list (default 12)",
                }
            },
        },
        _repo_map,
    ),
    Tool(
        "bytely_check_freshness",
        "Whether the saved graph still matches the source tree (a drift "
        "check). It does not refresh the graph; the other tools do that "
        "themselves before answering. no_cache=true re-extracts every file "
        "instead of trusting the extraction cache.",
        {
            "type": "object",
            "properties": {"no_cache": {"type": "boolean"}},
        },
        _check_freshness,
        refreshes=False,
    ),
)

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}
