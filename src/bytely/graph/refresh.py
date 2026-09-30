"""Keep the graph current before answering a query.

Every query refreshes first, so answers always describe the code as it is
now, including uncommitted edits. A refresh hashes a cheap snapshot of the
source tree — each source file's path, size, and modification time, plus
the build settings and the extractor fingerprint — and compares it with the
snapshot saved by the last build. If they match, the saved graph is loaded
as is; otherwise the graph is rebuilt, which re-parses only changed files.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from bytely.graph.extract_cache import current_extractor_fingerprint
from bytely.graph.write import graph_path, read_graph
from bytely.util import lock

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from contextlib import AbstractContextManager

    from bytely.graph.build import GraphBuildResult
    from bytely.graph.types import GraphV1

TREE_FINGERPRINT_FILE = Path("cache") / "tree-fingerprint.json"
BUILD_LOCK_FILE = Path("cache") / ".build.lock"
# The builder refreshes the lock every quarter of this while it works.
BUILD_LOCK_STALE_SECONDS = 60.0
# How long a refresh waits for another process's rebuild before building
# anyway (unlocked) rather than failing the query.
BUILD_LOCK_WAIT_SECONDS = 600.0


class NoGraphError(RuntimeError):
    """Raised when a refresh may not create a graph and none exists yet."""


@dataclass
class RefreshResult:
    """The current graph, and whether getting it needed a rebuild."""

    graph: GraphV1
    rebuilt: bool
    build: GraphBuildResult | None = None
    # The tree snapshot this graph matches; pass it back as `known` to skip
    # re-reading an unchanged graph.
    fingerprint: str = ""


@dataclass(frozen=True)
class BuildState:
    """What the last build recorded: its snapshot and its file selection."""

    fingerprint: str | None = None
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()


def tree_fingerprint(
    root: Path,
    source_paths: Sequence[str],
    include_patterns: Iterable[str],
    exclude_patterns: Iterable[str],
) -> str:
    """Hash what a build's output depends on, without reading any file."""
    digest = hashlib.sha256()
    digest.update(current_extractor_fingerprint().encode())
    digest.update(
        json.dumps([list(include_patterns), list(exclude_patterns)]).encode()
    )
    # Walked paths are POSIX-style and start with the root; slicing avoids a
    # Path object per file.
    prefix = root.as_posix().rstrip("/") + "/"
    for path in source_paths:
        try:
            stat = os.stat(path)
        except OSError:
            continue
        relative = (
            path[len(prefix) :]
            if path.startswith(prefix)
            else Path(path).relative_to(root).as_posix()
        )
        digest.update(
            f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def save_tree_fingerprint(
    context_dir: str,
    fingerprint: str,
    include_patterns: Iterable[str] = (),
    exclude_patterns: Iterable[str] = (),
) -> None:
    """Record the snapshot a build was made from, and its file selection."""
    path = Path(context_dir) / TREE_FINGERPRINT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tree": fingerprint,
                "include": list(include_patterns),
                "exclude": list(exclude_patterns),
            }
        ),
        "utf-8",
    )


def load_build_state(context_dir: str) -> BuildState:
    """What the last build in `context_dir` recorded (empty if nothing)."""
    try:
        data = json.loads(
            (Path(context_dir) / TREE_FINGERPRINT_FILE).read_text("utf-8")
        )
    except (OSError, ValueError):
        return BuildState()
    if not isinstance(data, dict):
        return BuildState()

    def patterns(key: str) -> tuple[str, ...]:
        value = data.get(key)
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return tuple(value)
        return ()

    fingerprint = data.get("tree")
    return BuildState(
        fingerprint if isinstance(fingerprint, str) else None,
        patterns("include"),
        patterns("exclude"),
    )


def load_tree_fingerprint(context_dir: str) -> str | None:
    """The snapshot the saved graph was built from, if recorded."""
    return load_build_state(context_dir).fingerprint


def build_lock(context_dir: str | Path) -> AbstractContextManager[str | None]:
    """Hold the graph's build lock (for callers that build directly)."""
    return lock.held(
        Path(context_dir) / BUILD_LOCK_FILE,
        stale=BUILD_LOCK_STALE_SECONDS,
        wait=BUILD_LOCK_WAIT_SECONDS,
    )


def refresh_graph(
    root: str,
    context_dir: str | None = None,
    *,
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    known: tuple[str, GraphV1] | None = None,
    create: bool = True,
) -> RefreshResult:
    """Return the current graph, rebuilding only if the tree changed.

    Patterns left as `None` reuse the ones the last build recorded, so a
    graph built with `--include "src/**"` stays scoped. `known` is a
    `(fingerprint, graph)` pair from an earlier refresh: if the tree still
    matches it, that graph is returned without reading the disk. With
    `create=False`, a directory that has no graph yet raises `NoGraphError`
    instead of being indexed.
    """
    from bytely.graph.build import build_graph, list_source_files

    root_path = Path(root).resolve()
    ctx_dir = context_dir or str(root_path / "bytely")
    if not create and not graph_path(ctx_dir).is_file():
        raise NoGraphError(
            f"No graph in {ctx_dir}; run `bytely build` in {root_path} first"
        )
    state = load_build_state(ctx_dir)
    includes = list(
        state.include_patterns if include_patterns is None else include_patterns
    )
    excludes = list(
        state.exclude_patterns if exclude_patterns is None else exclude_patterns
    )
    current = tree_fingerprint(
        root_path,
        list_source_files(root_path, includes, excludes),
        includes,
        excludes,
    )
    if known is not None and known[0] == current:
        return RefreshResult(graph=known[1], rebuilt=False, fingerprint=current)
    if state.fingerprint == current:
        graph = read_graph(ctx_dir)
        if graph is not None:
            return RefreshResult(
                graph=graph, rebuilt=False, fingerprint=current
            )

    # One rebuild at a time per graph: a query, the MCP server, and the
    # background sync may all find the tree changed at once. Whoever waited
    # re-checks first, since the build it waited for may already be current.
    with build_lock(ctx_dir):
        if load_build_state(ctx_dir).fingerprint == current:
            graph = read_graph(ctx_dir)
            if graph is not None:
                return RefreshResult(
                    graph=graph, rebuilt=False, fingerprint=current
                )
        result = build_graph(
            str(root_path),
            ctx_dir,
            include_patterns=includes,
            exclude_patterns=excludes,
        )
    graph = read_graph(ctx_dir)
    if graph is None:
        raise RuntimeError(f"Build wrote no graph in {ctx_dir}")
    return RefreshResult(
        graph=graph,
        rebuilt=True,
        build=result,
        fingerprint=load_tree_fingerprint(ctx_dir) or current,
    )
