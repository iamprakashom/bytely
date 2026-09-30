"""One implementation of each query, shared by the CLI and the MCP server.

Each function checks its arguments before it touches the graph, so a bad
call fails fast instead of paying for a refresh, and both front ends give
the same answers and the same error messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from bytely.graph.map import render_map
from bytely.graph.refresh import RefreshResult, refresh_graph
from bytely.graph.root import find_bytely_root
from bytely.hooks.metrics import session_input_rate
from bytely.hooks.savings import baseline_for, paths_in, with_savings
from bytely.query.ask import render_ask
from bytely.query.callers import Direction, render_callers
from bytely.query.common import SourceReader, find_file, find_symbols
from bytely.query.grep import render_grep
from bytely.query.skeleton import render_skeleton

if TYPE_CHECKING:
    from bytely.graph.types import GraphV1, NodeV1


class QueryError(Exception):
    """A query could not be answered; the message is meant for the caller."""


def repo_root(directory: str | None) -> Path:
    """The repository a command works on.

    An explicit directory wins; otherwise the nearest ancestor of the current
    directory that has a graph, else the current directory itself.
    """
    if directory:
        return Path(directory).resolve()
    return (find_bytely_root(Path.cwd()) or Path.cwd()).resolve()


def parse_depth(value: object) -> int | None:
    """A walk depth: a whole number of levels, or `all`/`full` for no limit."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("all", "full"):
            return None
        if text.isdigit() and int(text) >= 1:
            return int(text)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if value >= 1 and float(value).is_integer():
            return int(value)
    raise QueryError('depth must be a positive whole number, or "all"')


def parse_direction(value: object) -> Direction:
    """`in` (who uses it) or `out` (what it uses)."""
    if value == "in":
        return "in"
    if value == "out":
        return "out"
    raise QueryError('direction must be "in" or "out"')


def parse_count(value: object, name: str) -> int:
    """A positive whole number (JSON clients may send `5.0`)."""
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 1
        and float(value).is_integer()
    ):
        return int(value)
    raise QueryError(f"{name} must be a positive whole number")


def require_text(value: object, name: str) -> str:
    """A non-empty string argument."""
    if not isinstance(value, str) or not value.strip():
        raise QueryError(f"{name} must be a non-empty string")
    return value


@dataclass
class Workspace:
    """The repository queries run against, and its current graph.

    `create=False` never indexes a directory that has no graph (the MCP
    server may be started from anywhere). The last graph is kept in memory
    and reused while the source tree is unchanged, so a long-lived server
    does not re-read the graph file for every call.
    """

    root: Path
    context_dir: str | None = None
    create: bool = True
    last: RefreshResult | None = field(default=None, repr=False)

    def refresh(self) -> RefreshResult:
        """Bring the graph up to date and return it."""
        known = (
            (self.last.fingerprint, self.last.graph)
            if self.last is not None and self.last.fingerprint
            else None
        )
        self.last = refresh_graph(
            str(self.root),
            self.context_dir,
            known=known,
            create=self.create,
        )
        return self.last

    def graph(self) -> GraphV1:
        """The current graph."""
        return self.refresh().graph


def _saved(
    workspace: Workspace, text: str, paths: list[str] | None = None
) -> str:
    """Open `text` with its savings estimate (the files it spares reading).

    `paths` defaults to the indexed files the output names.
    """
    graph = (
        workspace.graph() if workspace.last is None else workspace.last.graph
    )
    covered = paths if paths is not None else paths_in(graph, text)
    return with_savings(
        text, baseline_for(graph, covered), session_input_rate(workspace.root)
    )


def _in_scope(node: NodeV1, scope: str | None) -> bool:
    if not scope:
        return True
    prefix = scope.replace("\\", "/").strip("/")
    return node.path == prefix or node.path.startswith(prefix + "/")


def skeleton_text(workspace: Workspace, file: object) -> str:
    """`skeleton`: a file's definitions."""
    name = require_text(file, "file")
    graph = workspace.graph()
    node = find_file(graph, workspace.root, name)
    if node is None:
        raise QueryError(f"No indexed file matches {name!r}")
    return _saved(workspace, render_skeleton(graph, node), [node.path])


def callers_text(
    workspace: Workspace,
    symbol: object,
    *,
    direction: object = "in",
    depth: object = 1,
    scope: str | None = None,
) -> str:
    """`callers`: edges into or out of a symbol.

    A file path stands for every definition in the file.
    """
    name = require_text(symbol, "symbol")
    walk_direction = parse_direction(direction)
    walk_depth = parse_depth(depth)
    graph = workspace.graph()
    seeds = [
        node for node in find_symbols(graph, name) if _in_scope(node, scope)
    ]
    if not seeds:
        file_node = find_file(graph, workspace.root, name)
        if file_node is not None and _in_scope(file_node, scope):
            seeds = [file_node]
    if not seeds:
        where = f" under {scope}" if scope else ""
        raise QueryError(f"No symbol matches {name!r}{where}")
    return _saved(
        workspace,
        render_callers(
            graph, seeds, direction=walk_direction, depth=walk_depth
        ),
    )


def grep_text(
    workspace: Workspace,
    pattern: object,
    *,
    fixed: bool = False,
    ignore_case: bool = False,
    scope: str | None = None,
) -> str:
    """`grep`: every match, grouped by enclosing symbol."""
    text = require_text(pattern, "pattern")
    if not fixed:
        try:
            re.compile(text)
        except re.error as error:
            raise QueryError(f"Invalid pattern {text!r}: {error}") from error
    return _saved(
        workspace,
        render_grep(
            workspace.graph(),
            SourceReader(workspace.root),
            text,
            fixed=fixed,
            ignore_case=ignore_case,
            scope=scope,
        ),
    )


def ask_text(
    workspace: Workspace,
    question: object,
    *,
    limit: object = 8,
    source: bool = False,
    full: bool = False,
    scope: str | None = None,
) -> str:
    """`ask`: ranked definitions for a question."""
    text = require_text(question, "query")
    count = parse_count(limit, "limit")
    return _saved(
        workspace,
        render_ask(
            workspace.graph(),
            SourceReader(workspace.root),
            text,
            limit=count,
            source=source,
            full=full,
            scope=scope,
        ),
    )


def map_text(workspace: Workspace, max_dirs: object = 12) -> str:
    """`map`: folders, hubs, and hotspots."""
    count = parse_count(max_dirs, "max_dirs")
    graph = workspace.graph()
    files = [node.path for node in graph.nodes if node.kind == "file"]
    return _saved(workspace, render_map(graph, max_dirs=count), files)
