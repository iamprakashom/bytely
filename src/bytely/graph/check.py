"""Freshness checks for the saved structural graph."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from bytely.graph.build import build_graph
from bytely.graph.refresh import load_build_state
from bytely.graph.write import graph_path, read_graph

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bytely.graph.types import GraphV1


def check_graph(
    root: str,
    context_dir: str | None = None,
    *,
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = None,
    use_cache: bool = True,
) -> tuple[bool, str]:
    """Compare the saved graph with a deterministic source-tree rebuild.

    Patterns left as `None` reuse the ones the saved graph was built with,
    so a scoped graph is compared with a scoped rebuild. With
    `use_cache=False` every file is re-extracted, so a cache that no longer
    matches the extractor cannot make a stale graph look fresh.
    """
    root_path = Path(root).resolve()
    graph_dir = (
        Path(context_dir).resolve() if context_dir else root_path / "bytely"
    )
    saved_graph = read_graph(str(graph_dir))
    if saved_graph is None:
        return False, f"Graph is missing: {graph_path(graph_dir)}"
    state = load_build_state(str(graph_dir))
    if include_patterns is None:
        include_patterns = state.include_patterns
    if exclude_patterns is None:
        exclude_patterns = state.exclude_patterns

    with TemporaryDirectory(prefix="bytely-check-") as temporary_dir:
        build_graph(
            str(root_path),
            temporary_dir,
            cache_dir=str(graph_dir) if use_cache else temporary_dir,
            persist_cache=False,
            output_formats=(),
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        current_graph = read_graph(temporary_dir)

    if current_graph is None:
        return False, "Could not build a graph for comparison."
    if _structure(saved_graph.to_dict()) != _structure(current_graph.to_dict()):
        return False, "Graph is stale; run `bytely build` to refresh it."
    return _meaning_report(saved_graph)


_MEANING_KEYS = ("summary", "summary_state", "crux")
PENDING_SAMPLE = 8


def _structure(data: dict[str, Any]) -> dict[str, Any]:
    """The graph without its meaning layer, which a rebuild cannot know."""
    return {
        **data,
        "nodes": [
            {k: v for k, v in node.items() if k not in _MEANING_KEYS}
            for node in data.get("nodes", [])
        ],
    }


def _meaning_report(graph: GraphV1) -> tuple[bool, str]:
    """Fresh unless summaries went stale; note how many are still pending.

    Pending (never summarized) is not drift: a build without `--deep`
    leaves everything pending by design.
    """
    stale = sorted(n.id for n in graph.nodes if n.summary_state == "stale")
    pending = sorted(n.id for n in graph.nodes if n.summary_state == "pending")
    if stale:
        shown = "\n".join(f"  ! {node_id}" for node_id in stale[:20])
        more = f"\n  … +{len(stale) - 20} more" if len(stale) > 20 else ""
        return False, (
            f"Graph structure is up to date, but {len(stale)} summar"
            f"{'y is' if len(stale) == 1 else 'ies are'} stale (the code "
            f"changed after it was summarized):\n{shown}{more}\n"
            "Run `bytely build --deep` to refresh them."
        )
    message = "Graph is up to date."
    total = len(graph.nodes)
    if pending and len(pending) < total:
        percent = round((total - len(pending)) / total * 100)
        sample = ", ".join(pending[:PENDING_SAMPLE])
        extra = len(pending) - PENDING_SAMPLE
        more = f", … +{extra} more" if extra > 0 else ""
        message += (
            f" Meaning layer {percent}% complete — {len(pending)} of "
            f"{total} node(s) pending: {sample}{more}. Run `bytely build "
            "--deep` to summarize them; if a deep build already left them "
            "pending, its meaning pass failed for them — see that build's "
            "errors."
        )
    return True, message
