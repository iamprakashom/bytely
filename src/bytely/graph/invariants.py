"""Structural invariants for serialized code graphs."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bytely.graph.types import GraphV1

SPAN_PATTERN = re.compile(r"^L([1-9]\d*)-L([1-9]\d*)$")

# As in the reference implementation, an import or base type that resolves to no
# node keeps its edge, pointing at the raw specifier or bare name (`typing`,
# `react`, `Exception`). Every other relation must target a node (P4).
NAME_TARGET_RELATIONS = frozenset({"imports", "extends", "implements"})


def validate_graph(graph: GraphV1) -> None:
    """Raise ValueError when graph paths, spans, or edge references fail."""
    errors: list[str] = []
    node_ids: set[str] = set()

    for node in graph.nodes:
        if node.id in node_ids:
            errors.append(f"{node.id}: duplicate node id")
        node_ids.add(node.id)
        # String checks equivalent to PurePosixPath's is_absolute()/parts;
        # constructing a path object per node dominated validation time.
        if (
            node.path.startswith("/")
            or ".." in node.path.split("/")
            or "\\" in node.path
        ):
            errors.append(
                f"{node.id}: path must be repository-relative and "
                "POSIX-normalized"
            )
        if node.id != node.path and not node.id.startswith(f"{node.path}#"):
            errors.append(f"{node.id}: id must be scoped to its source path")
        span_match = SPAN_PATTERN.fullmatch(node.span)
        if not span_match or int(span_match.group(1)) > int(
            span_match.group(2)
        ):
            errors.append(f"{node.id}: invalid source span {node.span!r}")

    for edge in graph.edges:
        if edge.source not in node_ids:
            errors.append(f"{edge.relation}: missing source node {edge.source}")
        if edge.target not in node_ids and (
            edge.relation not in NAME_TARGET_RELATIONS or not edge.target
        ):
            errors.append(f"{edge.relation}: missing target node {edge.target}")

    for scope in graph.scopes:
        if scope.prefix and (
            not scope.prefix.endswith("/") or "\\" in scope.prefix
        ):
            errors.append(
                f"{scope.label}: scope prefix must be POSIX-normalized "
                "and end with '/'"
            )

    if errors:
        details = "\n".join(f"- {error}" for error in errors[:20])
        remainder = len(errors) - 20
        if remainder > 0:
            details += f"\n- and {remainder} more issue(s)"
        raise ValueError(f"Graph invariant check failed:\n{details}")
