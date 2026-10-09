"""Text the hooks inject and the status line prints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from bytely.hooks.savings import (
    PREFIX,
    baseline_for,
    dollars_saved,
    format_dollars,
    to_tokens,
)
from bytely.query.ask import rank, terms
from bytely.query.common import qualified_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from bytely.graph.types import GraphV1, NodeV1


def _color(code: str) -> Callable[[str], str]:
    return lambda text: f"\x1b[{code}m{text}\x1b[0m"


TEAL = _color("38;2;45;190;170")
AMBER = _color("38;2;224;165;68")
MUTED = _color("38;5;244")
TEXT = _color("38;5;251")
SEP = MUTED(" · ")


def freshness(stats: dict[str, Any]) -> str:
    """`✓ synced`, `⚠ N stale`, or `syncing…`."""
    if stats.get("syncing"):
        return AMBER("syncing…")
    stale = int(stats.get("staleCount") or 0)
    if stats.get("dirty"):
        return AMBER(f"⚠ {stale} stale" if stale else "⚠ stale")
    return TEAL("✓ synced")


def render_statusline(
    stats: dict[str, Any] | None,
    session: dict[str, Any] | None,
    context_percent: int | None = None,
) -> list[str]:
    """One or two status lines: graph, freshness, savings; context, file."""
    if not stats:
        return [MUTED("◆ bytely · not built · run ") + TEXT("bytely build")]
    top = [
        MUTED("◆ ") + TEAL("bytely"),
        TEXT(
            f"{stats.get('nodeCount', 0):,} nodes / "
            f"{stats.get('edgeCount', 0):,} edges"
        ),
        freshness(stats),
    ]
    saved = int((session or {}).get("savedTokens") or 0)
    if saved > 0:
        usd = dollars_saved(
            saved,
            (session or {}).get("inputCostMicros"),
            (session or {}).get("inputTokensBilled"),
        )
        money = "" if usd is None else f" · ~{format_dollars(usd)}"
        top.append(TEAL(f"~{saved:,} tok saved{money}"))
    bottom = []
    if context_percent is not None:
        bottom.append(TEXT(f"ctx {context_percent}%"))
    if stats.get("lastFile"):
        bottom.append(MUTED("last: ") + TEXT(str(stats["lastFile"])))
    lines = [SEP.join(top)]
    if bottom:
        lines.append(MUTED("▸ ") + SEP.join(bottom))
    return lines


def render_subagent(agent: str, session: dict[str, Any] | None) -> str:
    """A subagent's status line: its name and the query it started from."""
    query = ((session or {}).get("perAgentQuery") or {}).get(agent)
    tail = SEP + MUTED("bytely: ") + TEXT(query) if query else ""
    return MUTED("◆ ") + TEAL(agent) + tail


def _file_matches(node_path: str, edited: str) -> bool:
    return edited == node_path or edited.endswith("/" + node_path)


def blast_radius(graph: GraphV1, edited: str, cap: int = 8) -> str | None:
    """Who depends on the definitions in an edited file."""
    edited = edited.replace("\\", "/")
    ids = {node.id for node in graph.nodes if _file_matches(node.path, edited)}
    if not ids:
        return None
    edges = [
        edge
        for edge in graph.edges
        if edge.target in ids
        and edge.source not in ids
        and edge.relation != "contains"
    ]
    if not edges:
        return None
    by_id = {node.id: node for node in graph.nodes}
    items = []
    for edge in edges[:cap]:
        source = by_id.get(edge.source)
        label = (
            f"{qualified_name(source)} ({source.path})"
            if source
            else edge.source
        )
        items.append(f" • {edge.relation} ← {label}")
    if len(edges) > cap:
        items.append(
            f" • +{len(edges) - cap} more (bytely callers "
            f"{PurePath(edited).name} --depth 2)"
        )
    return (
        f"[bytely] blast radius for {PurePath(edited).name}, who depends on "
        "it:\n" + "\n".join(items)
    )


DIRECTIVE = """\
[bytely] This repo is indexed by bytely. To find, understand, or change code, reach for bytely first: it answers from a prebuilt graph with exact file:line, faster than grep/read. Call its MCP tools directly (the CLI in parentheses is only for when they are unavailable; never wrap either in shell pipes). Pick the ONE tool that fits and act on its answer; most tasks need a single call. If one isn't enough, switch to the tool that fits the next need rather than re-asking a reworded question:
  • bytely_find_code (CLI: bytely ask "<task>" --source): locate + understand. Ranked definitions with the code inlined at each file:line (the first 8 lines). The default for "how does X work" / "where is Y".
  • bytely_find_all (CLI: bytely grep "<literal>"): exhaustive find. Every occurrence, grouped by enclosing symbol; use it when you need them ALL (find_code is ranked top-N).
  • bytely_file_api (CLI: bytely skeleton <file>): a file's whole API, every signature + span, ~10x cheaper than reading the file.
  • bytely_trace_calls (CLI: bytely callers <sym>): exact edges. Who calls it (default), what it calls (direction "out"), or the blast radius (depth 2, or "all"). Run it before you change a symbol.
  • bytely_repo_map (CLI: bytely map): orientation for an unfamiliar repo: folders, hubs, hotspots. The map alone is the answer.
  Scope find_code/find_all/trace_calls to one sub-project with in="<path>/".
  Already know the file or symbol? Go straight to it: bytely_find_all on the symbol, read the span, edit.
  Refactor, rename, or multi-file change? Run bytely_trace_calls with depth "all" FIRST to map every connected file.
Each tool opens its output with a "[bytely] tokens saved ≈ N" line, sometimes with its dollar value; when you used bytely this turn, close your reply with a one-line tally of the total saved, dollars included when given (e.g. ⚡ bytely saved ~12k tokens (~$0.04) this turn, 3 calls). Never price tokens yourself; never pipe bytely through head/tail — output is already capped, and clipping drops that line.
"""  # noqa: E501


def orientation(
    index_md: str, budget: int = 1500, banner: str | None = None
) -> str:
    """The session-start context: directive, then the start of INDEX.md."""
    head = f"{banner}\n\n" if banner else ""
    return (
        f"{head}{DIRECTIVE}\nrepo map (bytely/INDEX.md):\n{index_md[:budget]}"
    )


# A probe below both floors found nothing the prompt is really about.
STRONG_FLOOR = 0.1
BROAD_FLOOR = 0.5
NUDGE_CAP = 2
INJECTED_POINTERS_CAP = 40


@dataclass(frozen=True)
class Probe:
    """The prompt's top hits, and how well the best of them matched."""

    hits: list[NodeV1]
    strong: float  # share of query terms in a hit's name
    broad: float  # share of query terms anywhere in a hit


def probe(
    graph: GraphV1, prompt: str, limit: int = 3, scope: str | None = None
) -> Probe:
    """Rank definitions for a prompt, measuring how well they match."""
    query = list(dict.fromkeys(terms(prompt)))
    hits = [hit.node for hit in rank(graph, prompt, scope=scope)[:limit]]
    if not query or not hits:
        return Probe(hits, 0.0, 0.0)
    strong = broad = 0.0
    for node in hits:
        name = set(terms(node.name))
        text = " ".join(
            (
                node.path,
                qualified_name(node),
                node.signature or "",
                node.body or "",
            )
        ).lower()
        strong = max(strong, sum(t in name for t in query) / len(query))
        broad = max(broad, sum(t in text for t in query) / len(query))
    return Probe(hits, strong, broad)


def pointer(node: NodeV1) -> str:
    """`path:L1-L9`, the key a pointer is remembered by."""
    return f"{node.path}:{node.span}"


def _pack(hits: list[NodeV1]) -> str:
    blocks = []
    for index, node in enumerate(hits, start=1):
        block = f" {index}. {qualified_name(node)}: {pointer(node)}"
        detail = " ".join((node.summary or node.signature or "").split())
        if detail:
            block += f"\n    {detail[:140]}"
        blocks.append(block)
    return (
        "[bytely] starting points for this task: pull the code inline with "
        "the bytely_find_code tool, trace impact with bytely_trace_calls, or "
        "search with bytely_find_all:"
        "\n" + "\n".join(blocks)
    )


def retrieval(
    graph: GraphV1, found: Probe, session: dict[str, Any], cap: int = 3
) -> str | None:
    """The prompt's pointers, or None when they would only be noise.

    Two gates: relevance (a weak probe gets at most `NUDGE_CAP` nudges a
    session, then silence) and novelty (a pointer is injected into a
    session once). `session` is updated in place.
    """
    if not found.hits:
        return None
    if found.strong < STRONG_FLOOR and found.broad < BROAD_FLOOR:
        spent = int(session.get("nudges") or 0)
        if spent >= NUDGE_CAP:
            return None
        session["nudges"] = spent + 1
        return (
            "[bytely] no strong match for this prompt (name match "
            f"{found.strong:.2f}) — the graph has more than this probe "
            "found. Call the bytely_find_code tool before grepping."
        )
    seen = set(session.get("injectedPointers") or [])
    fresh = [node for node in found.hits if pointer(node) not in seen][:cap]
    if not fresh:
        return None
    body = _pack(fresh)
    session["injectedPointers"] = [
        *(session.get("injectedPointers") or []),
        *(pointer(node) for node in fresh),
    ][-INJECTED_POINTERS_CAP:]
    baseline = baseline_for(graph, (node.path for node in fresh))
    if baseline is None:
        return body
    pack = to_tokens(len(body))
    base = to_tokens(baseline.chars)
    if base <= pack:
        return body
    saved = base - pack
    return (
        f"{body}\n{PREFIX} {saved:,} ({round(saved / base * 100)}%); this "
        f"pack ≈ {pack:,} tok vs reading the {baseline.files} file(s) whole "
        f"≈ {base:,} tok (estimate)."
    )
