"""Curated architecture nodes (systems, files, concepts) from file summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bytely.ai.crux import tool_args
from bytely.ai.llm.types import ChatRequest, Message, Tool

if TYPE_CHECKING:
    from bytely.ai.llm.types import ChatModel

SYSTEM_PROMPT = """\
You build an ARCHITECTURE graph of a codebase from per-file summaries. The reader is an AI agent that will read this graph before working on the code, so it must describe the system at the level a senior engineer would explain it — not file by file.

Produce a CURATED set of nodes of mixed granularity:
- "system" nodes: GROUP files that collaborate as one component (usually a directory or a cohesive set) into a SINGLE node. This should be the most common node type. Prefer one system node over several file nodes.
- "file" nodes: only for a substantial, standalone module that genuinely deserves its own node apart from its system.
- "concept" nodes: cross-cutting ideas, design decisions, or invariants that span multiple files (e.g. "local-first provider fallback", "staleness checking", "content-hash provenance"). Include several — they are the most valuable nodes for an agent.

Rules:
- Every summary must earn its tokens with NON-OBVIOUS information: invariants, ordering constraints, conventions, failure modes, "X must never happen after Y" facts, and the WHY behind a design. Never restate what a README says or what a directory listing already makes obvious ("src/api contains the API code" is worthless); an agent reading the node already sees the file paths. If all you can say about a group of files is what their names say, fold them into a larger node instead.
- Strongly prefer FEWER, larger, meaningful nodes. For a repo of N files, aim for well under N nodes. Do NOT emit one node per file, and never a node per incidental identifier (a local interface, helper, or third-party symbol).
- Merge duplicates and surface-form variants into one node.
- For each node give: a canonical human-readable name; a type ("system" | "file" | "concept"); a 1-3 sentence summary of its ROLE in the system; "sources" = the exact file paths (from the input) it is grounded in (a system lists all its files; a concept lists the files that motivate it); and "links" to other nodes you define, each with a relation and a short description of what concretely happens in the code.
- The relation MUST be one of exactly these verbs (each answers a question a code reviewer asks): "part_of" (where does this live?), "uses" (what breaks if the target changes?), "depends_on" (same, for non-call dependencies), "produces" (where does this output come from?), "configures" (what changes its behavior without a code change?), "validates" (what checks or judges this? tests, drift checks, scoring), "implements" (what contract must this honor?). Never invent vague relations like "influences", "supports", or "relates_to" — if none of the verbs fit, drop the link.
- Only link to nodes you actually define in this response.
Respond only via the record_graph tool / JSON schema."""  # noqa: E501

RELATIONS = (
    "part_of",
    "uses",
    "depends_on",
    "produces",
    "configures",
    "validates",
    "implements",
)
TOOL_NAME = "record_graph"
TOOL = Tool(
    TOOL_NAME,
    "Record the curated architecture-graph nodes and their links.",
    {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "summary": {"type": "string"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "links": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "to": {"type": "string"},
                                    "relation": {
                                        "type": "string",
                                        "enum": list(RELATIONS),
                                    },
                                    "description": {"type": "string"},
                                },
                                "required": ["to", "relation"],
                            },
                        },
                    },
                    "required": ["name", "type", "summary", "sources"],
                },
            }
        },
        "required": ["nodes"],
    },
)
MAX_INPUT_CHARS = 60_000


@dataclass(frozen=True)
class Link:
    """A relation from one synthesized node to another, by name."""

    to: str
    relation: str
    description: str | None = None


@dataclass(frozen=True)
class SynthNode:
    """One synthesized architecture node."""

    name: str
    type: str
    summary: str
    sources: list[str]
    links: list[Link] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON form, for the synthesis cache."""
        return {
            "name": self.name,
            "type": self.type,
            "summary": self.summary,
            "sources": self.sources,
            "links": [
                {"to": link.to, "relation": link.relation}
                | (
                    {"description": link.description}
                    if link.description
                    else {}
                )
                for link in self.links
            ],
        }


def clean_nodes(raw: object) -> list[SynthNode]:
    """Well-formed nodes from a payload; unknown relations are dropped."""
    if not isinstance(raw, list):
        return []
    nodes = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name, kind = item.get("name"), item.get("type")
        if not isinstance(name, str) or not name.strip():
            continue
        links = [
            Link(
                link["to"],
                link["relation"],
                link.get("description")
                if isinstance(link.get("description"), str)
                else None,
            )
            for link in item.get("links") or []
            if isinstance(link, dict)
            and isinstance(link.get("to"), str)
            and link.get("relation") in RELATIONS
        ]
        nodes.append(
            SynthNode(
                name.strip(),
                kind if isinstance(kind, str) and kind else "concept",
                str(item.get("summary") or ""),
                [s for s in item.get("sources") or [] if isinstance(s, str)],
                links,
            )
        )
    return nodes


class Synthesizer:
    """Turns a batch of file summaries into architecture nodes."""

    def __init__(self, model: ChatModel) -> None:
        """Use `model` for every call."""
        self.model = model

    def synthesize(self, files: list[tuple[str, str]]) -> list[SynthNode]:
        """Nodes for `(path, summary)` pairs (empty when nothing usable)."""
        if not files:
            return []
        body = "\n\n".join(f"## {path}\n\n{summary}" for path, summary in files)
        if len(body) > MAX_INPUT_CHARS:
            body = body[:MAX_INPUT_CHARS] + "\n… (truncated)"
        response = self.model.create(
            ChatRequest(
                messages=[
                    Message("system", SYSTEM_PROMPT),
                    Message("user", body),
                ],
                tools=[TOOL],
                force_tool=TOOL_NAME,
                max_tokens=8192,
            )
        )
        args = tool_args(response, TOOL_NAME, "nodes")
        return clean_nodes((args or {}).get("nodes"))
