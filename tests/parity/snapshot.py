"""Normalize Bytely's and the reference implementation's graphs and diff them.

Both tools write a `.graph/wiring.json`. Each is reduced to the fields Bytely
promises to match (see `NODE_FIELDS`), so a baseline recorded once from the
reference implementation can be compared against every Bytely build. Every
remaining difference must be listed in a reviewed divergence manifest with
one of the `REASONS` below.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from bytely.graph.types import GraphV1

NODE_FIELDS = (
    "name",
    "kind",
    "path",
    "span",
    "body_hash",
    "owner",
    "signature",
    "exported",
    "origin",
    "chars",
)

# Why a difference from the reference implementation is accepted. Codes are
# referenced by the divergence manifests; descriptions point at the
# Migrationplan.md findings.
REASONS: dict[str, str] = {
    "TOP_LEVEL_BINDINGS": (
        "Bytely emits top-level constant/variable nodes (and their contains "
        "edges) for full-fidelity languages; the reference implementation "
        "only does so in its generic tier. Migrationplan decision #13."
    ),
    "PYTHON_IMPORT_RESOLUTION": (
        "Bytely resolves Python relative, absolute, and dotted-alias imports "
        "to files that the reference implementation leaves unresolved, "
        "including the calls bound through them. Finding P5."
    ),
    "COMMONJS_REQUIRE": (
        "Bytely reads CommonJS require() imports and bindings (an external "
        "`require('fs')` keeps a raw-specifier edge, P4); the reference "
        "implementation does not for JavaScript. Finding A3."
    ),
    "IMPORT_BOUND_CALLS": (
        "Bytely resolves a call, or a base class (`class W(Base)`), through "
        "its explicit import binding (`extracted`), where the reference "
        "implementation falls back to a unique-name match (`inferred`) or "
        "finds none (aliased imports). Finding P6."
    ),
    "NESTED_SCOPE": (
        "Bytely binds definitions nested in a function only from inside that "
        "function; the reference implementation binds them from anywhere in "
        "the file. Finding A11."
    ),
    "RUST_GENERIC_TIER": (
        "The reference implementation indexes Rust only in its generic tier: "
        "flat IDs (`lib.rs#get`), every fn a `function`, traits as "
        "`interface`, and no contains edges. Bytely scopes impl and trait "
        "methods under their type (`lib.rs#Cache.get`, kind `method`) and "
        "nested fns under their function. The generic tier also marks every "
        "node `origin: generic` and `exported: true`; Bytely reports `ast` "
        "and `pub` visibility. Findings A16, P10."
    ),
    "RUST_MODULE_IMPORTS": (
        "Bytely resolves `use crate::...` paths, including glob imports, to "
        "module files and binds calls through them; the reference "
        "implementation resolves fewer. Neither keeps an unresolved `use` as "
        "a raw-specifier edge (P4 excludes Rust to match the reference "
        "implementation's generic tier). Finding A16."
    ),
    "MEMBER_ASSIGNED_FUNCTIONS": (
        "Bytely mints function/method nodes for functions assigned to "
        "properties (`app.use = function () {}`, `Foo.prototype.bar = …`, "
        "`exports.x = …`), and resolves `this.` calls inside them; the "
        "reference implementation mints none. Decision #11, finding A5."
    ),
    "HERITAGE_NOT_PORTED": (
        "The reference implementation keeps an extends/implements edge to a "
        "base it cannot resolve (usually an external type) as an edge to the "
        "bare name; Bytely resolves heritage but omits unresolved bases until"
        " the P4 decision on non-node edge targets. Gap P7."
    ),
}


def normalize_reference(wiring: dict[str, Any]) -> dict[str, Any]:
    """Reduce a reference-implementation graph file to a snapshot."""
    return _snapshot(wiring["nodes"], wiring["edges"])


def normalize_bytely(graph: GraphV1) -> dict[str, Any]:
    """Reduce a Bytely graph to a comparable snapshot."""
    data = graph.to_dict()
    return _snapshot(data["nodes"], data["edges"])


def _snapshot(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "nodes": {
            node["id"]: {field: node.get(field) for field in NODE_FIELDS}
            for node in sorted(nodes, key=lambda n: n["id"])
        },
        "edges": sorted(
            [
                edge["source"],
                edge["target"],
                edge["relation"],
                edge.get("confidence") or "extracted",
            ]
            for edge in edges
        ),
    }


def diff(reference: dict[str, Any], bytely: dict[str, Any]) -> list[list[Any]]:
    """List every difference between two snapshots, in a stable order.

    Items are JSON-friendly lists so manifests can store them verbatim:
    `["node-missing", id]`, `["node-extra", id]`,
    `["node-field", id, field, reference_value, bytely_value]`,
    `["edge-missing", source, target, relation]`,
    `["edge-extra", source, target, relation]`, and
    `["edge-confidence", source, target, relation, reference, bytely]`.
    """
    items: list[list[Any]] = []
    ref_nodes, by_nodes = reference["nodes"], bytely["nodes"]
    items.extend(
        ["node-missing", node_id]
        for node_id in sorted(ref_nodes.keys() - by_nodes.keys())
    )
    items.extend(
        ["node-extra", node_id]
        for node_id in sorted(by_nodes.keys() - ref_nodes.keys())
    )
    for node_id in sorted(ref_nodes.keys() & by_nodes.keys()):
        for field in NODE_FIELDS:
            up_value = ref_nodes[node_id].get(field)
            by_value = by_nodes[node_id].get(field)
            if up_value != by_value:
                items.append(["node-field", node_id, field, up_value, by_value])

    ref_edges = _edge_confidences(reference["edges"])
    by_edges = _edge_confidences(bytely["edges"])
    items.extend(
        ["edge-missing", *key]
        for key in sorted(ref_edges.keys() - by_edges.keys())
    )
    items.extend(
        ["edge-extra", *key]
        for key in sorted(by_edges.keys() - ref_edges.keys())
    )
    items.extend(
        ["edge-confidence", *key, ref_edges[key], by_edges[key]]
        for key in sorted(ref_edges.keys() & by_edges.keys())
        if ref_edges[key] != by_edges[key]
    )
    return items


def _edge_confidences(
    edges: list[list[str]],
) -> dict[tuple[str, str, str], str]:
    return {
        (source, target, relation): conf
        for source, target, relation, conf in edges
    }


def load_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    """Write deterministic, diff-friendly UTF-8 JSON with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # `newline="\n"`: text mode on Windows would otherwise write CRLF, which
    # `.gitattributes` (eol=lf) then rewrites on every checkout and commit.
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
