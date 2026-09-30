"""``.graph/wiring.json`` — the code graph schema (v1).

One node per definition (file, class, function, method, interface, type, enum),
wired by edges (contains, imports, calls, …). Field names follow the LSP
vocabulary (``name``, ``kind``, …) rather than any one tool's conventions.

Two tiers of data live on a node:
  - Tier-1 (deterministic, $0): everything from the AST. Rebuilt on every run.
  - Tier-2 (one LLM call, cached on ``body_hash``): ``summary`` + ``crux``.
  M1 populates Tier-1 only; Tier-2 fields ship as ``pending``/null.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Enums (string literals mirroring the TS union types)
# ---------------------------------------------------------------------------

Kind = Literal[
    "file",
    "class",
    "function",
    "method",
    "interface",  # TS + Go
    "type",  # TS + Go (type alias / named type)
    "enum",  # TS + PHP + Java
    "struct",  # Go only
    "trait",  # PHP only
    # The generic (tags.scm) breadth tier also emits these — every tree-sitter
    # grammar's tags.scm uses the tree-sitter tags @definition.<X> vocabulary,
    # and module/constant/variable are common in the long tail (Ruby modules,
    # Rust consts, top-level lets, …). Kept distinct rather than coerced so the
    # breadth tier's kinds read truthfully in cards/skeleton.
    "module",
    "constant",
    "variable",
]

Confidence = Literal[
    "lsp_resolved",  # exact server-confirmed target
    "lsp_dispatch",  # interface/virtual candidate
    "extracted",  # hand-written AST resolver
    "inferred",  # name-only heuristic
]

SummaryState = Literal["pending", "ready", "stale"]

# "ast": a language-specific extractor; "generic": a grammar-agnostic tier
# driven by tree-sitter tags queries. Bytely emits only "ast" today.
Origin = Literal["ast", "generic"]

Relation = Literal[
    "contains",
    "imports",
    "calls",
    "extends",
    "implements",
    "type_reference",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Crux:
    """The LLM-chosen business-logic excerpt.

    ``code`` is the source of truth; ``span`` is a best-effort pointer that may
    drift and is never used to re-slice.
    """

    code: str
    span: str  # e.g. "L189-L196"


@dataclass(slots=True)
class NodeV1:
    """A single symbol in the code graph.

    Identity is ``id`` (path-scoped, e.g. ``src/cache.py#Cache.get``).
    """

    # identity
    id: str  # path-scoped: "src/cache.py#Cache.get"
    name: str  # the symbol's own name: "get"
    kind: Kind

    # location (Tier-1, deterministic)
    path: str  # repo-relative: "src/cache.py"
    span: str  # whole definition: "L165-L200"
    body_hash: str  # SHA-256 of the definition body (whitespace-collapsed)

    # optional identity
    owner: str | None = None  # method nodes: bare name of enclosing class

    # Tier-1 header facts (reference parity, P10)
    signature: str | None = None  # definition header; None for file nodes
    exported: bool = True  # visible outside its module/file
    origin: Origin = "ast"  # which extractor produced the node
    chars: int | None = None  # file nodes: source length in UTF-16 units

    # Tier-1 content
    body: str | None = None  # searchable body (whitespace-collapsed, capped)

    # Tier-2 (LLM meaning layer)
    summary: str | None = None
    summary_state: SummaryState = "pending"
    crux: Crux | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict matching the JSON schema."""
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "span": self.span,
            "body_hash": self.body_hash,
            "signature": self.signature,
            "exported": self.exported,
            "origin": self.origin,
        }
        if self.chars is not None:
            d["chars"] = self.chars
        if self.owner is not None:
            d["owner"] = self.owner
        if self.body is not None:
            d["body"] = self.body
        if self.summary is not None:
            d["summary"] = self.summary
        d["summary_state"] = self.summary_state
        if self.crux is not None:
            d["crux"] = {"code": self.crux.code, "span": self.crux.span}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeV1:
        """Deserialize from the JSON schema dict."""
        crux_data = d.get("crux")
        crux = (
            Crux(code=crux_data["code"], span=crux_data["span"])
            if crux_data
            else None
        )
        return cls(
            id=d["id"],
            name=d["name"],
            kind=d["kind"],
            path=d["path"],
            span=d["span"],
            body_hash=d["body_hash"],
            owner=d.get("owner"),
            signature=d.get("signature"),
            exported=d.get("exported", True),
            origin=d.get("origin", "ast"),
            chars=d.get("chars"),
            body=d.get("body"),
            summary=d.get("summary"),
            summary_state=d.get("summary_state", "pending"),
            crux=crux,
        )


@dataclass(slots=True)
class EdgeV1:
    """A resolved edge between two nodes."""

    source: str  # node id
    target: str  # node id
    relation: Relation
    confidence: Confidence = "extracted"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the edge to the graph JSON schema."""
        d: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
        }
        if self.confidence != "extracted":
            d["confidence"] = self.confidence
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EdgeV1:
        """Create an edge from its serialized representation."""
        return cls(
            source=d["source"],
            target=d["target"],
            relation=d["relation"],
            confidence=d.get("confidence", "extracted"),
        )


@dataclass(slots=True)
class ScopeV1:
    """A monorepo sub-scope represented by a directory prefix and label."""

    prefix: str  # e.g. "packages/core/" or "" for root
    label: str  # human-readable: "core" or ""
    markers: list[str] = field(default_factory=list)  # e.g. ["package.json"]


@dataclass(slots=True)
class GraphV1:
    """The full code graph — serialized as ``.graph/wiring.json``."""

    version: int = 1
    nodes: list[NodeV1] = field(default_factory=list)
    edges: list[EdgeV1] = field(default_factory=list)
    scopes: list[ScopeV1] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph and its nodes, edges, and scopes."""
        return {
            "version": self.version,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "scopes": [
                {"prefix": s.prefix, "label": s.label, "markers": s.markers}
                for s in self.scopes
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphV1:
        """Create a graph from its serialized representation."""
        return cls(
            version=d.get("version", 1),
            nodes=[NodeV1.from_dict(n) for n in d.get("nodes", [])],
            edges=[EdgeV1.from_dict(e) for e in d.get("edges", [])],
            scopes=[
                ScopeV1(
                    prefix=s["prefix"],
                    label=s["label"],
                    markers=s.get("markers", []),
                )
                for s in d.get("scopes", [])
            ],
        )
