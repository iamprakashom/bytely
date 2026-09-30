"""`bytely map`: a short orientation to a repository, read from the graph.

The map lists the top-level folders (clusters) with their file and symbol
counts and their most-referenced symbols (hubs), then the most-referenced
symbols overall (hotspots). "Referenced" counts incoming edges other than
`contains` and `imports` — calls and inheritance — from other symbols.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bytely.graph.extract import language_label_of

if TYPE_CHECKING:
    from bytely.graph.types import GraphV1, NodeV1

STRUCTURAL_RELATIONS = frozenset({"contains", "imports"})


@dataclass
class _Cluster:
    name: str
    files: int = 0
    symbols: list[NodeV1] = field(default_factory=list)


def reference_counts(graph: GraphV1) -> Counter[str]:
    """Incoming non-structural edges per node ID, excluding self-edges."""
    ids = {node.id for node in graph.nodes}
    return Counter(
        edge.target
        for edge in graph.edges
        if edge.relation not in STRUCTURAL_RELATIONS
        and edge.target in ids
        and edge.target != edge.source
    )


def render_map(
    graph: GraphV1,
    *,
    max_dirs: int = 12,
    hubs_per_dir: int = 3,
    hotspots: int = 12,
) -> str:
    """Render the map as plain text."""
    counts = reference_counts(graph)
    clusters: dict[str, _Cluster] = {}
    languages: set[str] = set()
    for node in graph.nodes:
        head, _, rest = node.path.partition("/")
        name = f"{head}/" if rest else head
        cluster = clusters.setdefault(name, _Cluster(name))
        if node.kind == "file":
            cluster.files += 1
            label = language_label_of(node.path)
            if label:
                languages.add(label)
        else:
            cluster.symbols.append(node)

    symbol_total = sum(len(cluster.symbols) for cluster in clusters.values())
    file_total = sum(cluster.files for cluster in clusters.values())
    header = (
        f"repo map — {_count(file_total, 'file')} · "
        f"{_count(symbol_total, 'symbol')} · {_count(len(graph.edges), 'edge')}"
    )
    if languages:
        header += " · " + ", ".join(sorted(languages))
    lines = [header, ""]

    ordered = sorted(
        clusters.values(), key=lambda cluster: (-cluster.files, cluster.name)
    )
    width = max((len(c.name) for c in ordered[:max_dirs]), default=0) + 2
    for cluster in ordered[:max_dirs]:
        line = (
            f"{cluster.name:<{width}}{_count(cluster.files, 'file')} · "
            f"{_count(len(cluster.symbols), 'symbol')}"
        )
        top = _ranked(cluster.symbols, counts)[:hubs_per_dir]
        if top:
            line += "   hubs: " + ", ".join(
                f"{node.name} ({node.path.rsplit('/', 1)[-1]}, "
                f"{counts[node.id]}←)"
                for node in top
            )
        lines.append(line)
    if len(ordered) > max_dirs:
        lines.append(f"… {len(ordered) - max_dirs} more (--max-dirs to show)")

    symbols = [node for node in graph.nodes if node.kind != "file"]
    top_overall = _ranked(symbols, counts)[:hotspots]
    if top_overall:
        lines.append("")
        lines.append(
            "hotspots: "
            + "  ".join(
                f"{node.name} · {node.kind} · {node.path}:{node.span} · "
                f"{counts[node.id]}←"
                for node in top_overall
            )
        )
    return "\n".join(lines) + "\n"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" + ("" if number == 1 else "s")


def _ranked(nodes: list[NodeV1], counts: Counter[str]) -> list[NodeV1]:
    """Referenced nodes, most-referenced first; ties broken by ID."""
    return sorted(
        (node for node in nodes if counts[node.id]),
        key=lambda node: (-counts[node.id], node.id),
    )
