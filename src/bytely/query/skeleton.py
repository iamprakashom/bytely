"""`bytely skeleton`: a file's definitions, with spans and signatures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bytely.query.common import span_lines

if TYPE_CHECKING:
    from bytely.graph.types import GraphV1, NodeV1


def render_skeleton(graph: GraphV1, file_node: NodeV1) -> str:
    """Every definition in the file, in source order, one per line."""
    definitions = sorted(
        (
            node
            for node in graph.nodes
            if node.path == file_node.path and node.kind != "file"
        ),
        key=lambda node: (span_lines(node.span), node.id),
    )
    lines = [f"skeleton — {file_node.path} ({len(definitions)} definitions)"]
    for node in definitions:
        line = f"- {node.span}  {node.kind} {node.name}"
        if node.owner:
            line += f" (in {node.owner})"
        if node.signature:
            line += f"  {' '.join(node.signature.split())}"
        lines.append(line)
    return "\n".join(lines) + "\n"
