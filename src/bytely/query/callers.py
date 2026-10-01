"""`bytely callers`: who uses a symbol, or what it uses, from the graph edges.

Edges are exact (resolved during the build), not a text search. `contains`
and `imports` are structure, not use, so they are skipped. With a depth
above one, the walk continues from each newly reached symbol: the blast
radius of a change (`--direction in`) or everything a symbol depends on
(`--direction out`).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Literal

from bytely.query.common import STRUCTURAL_RELATIONS, describe

if TYPE_CHECKING:
    from bytely.graph.types import EdgeV1, GraphV1, NodeV1

Direction = Literal["in", "out"]
ARROW = {"in": "←", "out": "→"}
MAX_REPORTED = 400


def render_callers(
    graph: GraphV1,
    seeds: list[NodeV1],
    *,
    direction: Direction = "in",
    depth: int | None = 1,
    limit: int = MAX_REPORTED,
) -> str:
    """Each seed with the edges reached from it, level by level.

    `depth=None` walks the whole connected closure in that direction. At
    most `limit` edges are listed per seed; a note says when more exist.
    """
    nodes = {node.id: node for node in graph.nodes}
    adjacent: dict[str, list[EdgeV1]] = {}
    for edge in graph.edges:
        if edge.relation in STRUCTURAL_RELATIONS or edge.source == edge.target:
            continue
        key = edge.target if direction == "in" else edge.source
        adjacent.setdefault(key, []).append(edge)
    for edges in adjacent.values():
        edges.sort(key=lambda edge: (edge.relation, edge.source, edge.target))

    blocks = []
    for seed in seeds:
        lines = [describe(seed)]
        # A file stands for everything defined in it: callers target the
        # symbols, never the file node, so start from all of them.
        starts = [seed.id] + (
            sorted(
                node.id
                for node in graph.nodes
                if node.path == seed.path and node.kind != "file"
            )
            if seed.kind == "file"
            else []
        )
        seen = set(starts)
        queue: deque[tuple[str, int]] = deque((start, 1) for start in starts)
        reported = 0
        truncated = False
        while queue and not truncated:
            current, level = queue.popleft()
            if depth is not None and level > depth:
                continue
            for edge in adjacent.get(current, []):
                other = edge.source if direction == "in" else edge.target
                if other in seen:
                    continue
                if reported >= limit:
                    truncated = True
                    break
                seen.add(other)
                indent = "  " * level
                via = (
                    f"  (via {nodes[current].name})"
                    if current != seed.id and current in nodes
                    else ""
                )
                confidence = (
                    "" if edge.confidence == "extracted" else " [inferred]"
                )
                target = nodes.get(other)
                shown = (
                    describe(target)
                    if target is not None
                    else f"{other} (external)"
                )
                lines.append(
                    f"{indent}{edge.relation} {ARROW[direction]} "
                    f"{shown}{confidence}{via}"
                )
                reported += 1
                if target is not None:
                    queue.append((other, level + 1))
        if len(lines) == 1:
            lines.append(
                "  (no callers)" if direction == "in" else "  (no calls)"
            )
        elif truncated:
            lines.append(f"  … stopped after {limit} symbols; more exist")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
