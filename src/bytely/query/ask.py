"""`bytely ask`: rank the graph's definitions for a natural-language question.

No LLM and no embeddings. A question is split into terms (camelCase and
snake_case are split, words lightly stemmed, stop words dropped), and each
definition is scored on where the terms occur — its name counts most, then
its qualified name and path, its signature, then its body — weighted by how
rare each term is (IDF) with saturating term frequency (BM25-style). Body
matches count for less in long bodies, so a large class whose text holds
every word of the question does not outrank the short function about it. A
personalized PageRank over the call and inheritance edges, seeded by those
scores, then lifts code that is central to the matches. Structural
questions ("who calls X", "what does X call") go straight to `callers`.

A term matches in three tiers, strongest first: an exact word of the text
(`echo` in `echo` or `style_echo`), a word that starts with the term, or
that the term starts with in a name (`opt` for `option`, a prefix of
at least three letters), and any other fragment of a longer word (`echo`
inside `echoed`). The tiers count in full, `PREFIX_WEIGHT`, and
`FRAGMENT_WEIGHT`. A cheap scan over every definition picks the strongest
`RESCORE_LIMIT` candidates, and only those are scored with the tiers.
"""

from __future__ import annotations

import functools
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bytely.query.callers import Direction, render_callers
from bytely.query.common import (
    STRUCTURAL_RELATIONS,
    SourceReader,
    describe,
    find_symbols,
    numbered,
    qualified_name,
    span_lines,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bytely.graph.types import GraphV1, NodeV1

CRUX_LINES = 8
# Field weights: a term in the name says far more than one in the body.
NAME_WEIGHT = 4.0
QUALIFIED_WEIGHT = 2.0
SIGNATURE_WEIGHT = 1.5
# Body matches are discounted by the body's length (below), so they can
# weigh more than a flat count would let them without big bodies winning.
BODY_WEIGHT = 2.2
SATURATION = 1.2
# How much a body's length counts against its matches (BM25's b): a term
# found once in a short function says more than once in a class hundreds of
# lines long. 0 ignores length; 1 scales fully with length over the average.
LENGTH_NORM = 0.75
# How much a weaker kind of match counts, as a share of an exact word. An
# abbreviation or longer form of the term (`opt` / `option`) is a good sign;
# a fragment buried inside an unrelated longer word is a weak one.
PREFIX_WEIGHT = 0.65
FRAGMENT_WEIGHT = 0.3
# Shortest word that may stand for a longer one (`opt`, not `o`).
MIN_PREFIX = 3
# Definitions that get the full tiered scoring: the strongest by the cheap
# first pass. Far more than any answer shows, and a few tens of
# milliseconds of tokenizing whatever the repository's size.
RESCORE_LIMIT = 600
# How much of the final score comes from graph centrality.
PAGERANK_SHARE = 0.25
# At most this many hits: the question's words barely occur in the code, so
# another wording is unlikely to do better than a different tool.
FEW_HITS = 3
# The follow-up tools an answer suggests, as each surface spells them.
CLI_TOOLS = {
    "find_all": 'bytely grep "<literal>"',
    "file_api": "bytely skeleton {file}",
    "calls_out": "bytely callers {symbol} --direction out",
}
MCP_TOOLS = {
    "find_all": 'bytely_find_all pattern="<literal>"',
    "file_api": 'bytely_file_api file="{file}"',
    "calls_out": 'bytely_trace_calls symbol="{symbol}" direction="out"',
}
DAMPING = 0.85
ITERATIONS = 20
# Only the strongest lexical matches seed the walk; the long tail of weak
# matches barely moves the result but dominates the cost.
PAGERANK_SEEDS = 200
NEIGHBOURHOOD_HOPS = 2
FILE_PENALTY = 0.5
# Test names repeat the words of the code they test, so tests would crowd
# out the implementation; they rank lower unless the question is about tests.
TEST_PENALTY = 0.4
_TEST_PATH = re.compile(
    r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*$|_test\.[a-z]+$"
    r"|\.(test|spec)\.[a-z]+$"
)

STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "work",
        "works",
        "working",
        "code",
        "file",
        "files",
        "function",
        "functions",
        "method",
        "methods",
        "class",
        "classes",
        "find",
        "show",
        "me",
        "get",
        "use",
        "used",
        "using",
        "into",
        "there",
        "that",
        "these",
        "those",
        "my",
        "our",
        "we",
        "you",
    }
)
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STRUCTURAL: list[tuple[re.Pattern[str], Direction]] = [
    (re.compile(r"\b(?:who|what)\s+calls\s+`?([\w.#/~-]+)", re.I), "in"),
    (re.compile(r"\bcallers?\s+of\s+`?([\w.#/~-]+)", re.I), "in"),
    (re.compile(r"\bwhat\s+uses\s+`?([\w.#/~-]+)", re.I), "in"),
    (re.compile(r"\bwhat\s+does\s+`?([\w.#/~-]+)`?\s+call", re.I), "out"),
    (re.compile(r"\bcallees?\s+of\s+`?([\w.#/~-]+)", re.I), "out"),
]


def terms(text: str) -> list[str]:
    """Search terms in `text`: split identifiers, lowercase, stem lightly."""
    return list(_cached_terms(text))


@functools.lru_cache(maxsize=65536)
def _cached_terms(text: str) -> tuple[str, ...]:
    # Names, paths, and signatures repeat across nodes and queries.
    return _tokenize(text)


def _tokenize(text: str) -> tuple[str, ...]:
    out = []
    for raw in re.findall(r"[A-Za-z0-9]+", text):
        for part in _CAMEL.findall(raw) or [raw]:
            word = _stem(part.lower())
            if len(word) > 1 and word not in STOP_WORDS:
                out.append(word)
    return tuple(out)


@functools.lru_cache(maxsize=4096)
def _field_tokens(text: str) -> tuple[Counter[str], str]:
    """A field's words, counted, and as one space-led string.

    The string lets `str.count(" " + term)` count the words that start with
    a term, without looping over them.
    """
    tokens = _tokenize(text)
    return Counter(tokens), " " + " ".join(tokens)


@functools.lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


@dataclass
class Hit:
    """One ranked answer."""

    node: NodeV1
    score: float


def rank(
    graph: GraphV1, question: str, *, scope: str | None = None
) -> list[Hit]:
    """Definitions ordered by relevance to `question`, best first.

    At most `RESCORE_LIMIT` definitions are returned: the strongest matches
    of a first pass over all of them.
    """
    query = list(dict.fromkeys(terms(question)))
    if not query:
        return []
    prefix = scope.replace("\\", "/").strip("/") + "/" if scope else ""
    candidates = [node for node in graph.nodes if node.path.startswith(prefix)]

    # First pass, over every definition. The name is split into words (short,
    # and where precision matters most), so its tiers are exact. Other fields
    # only need the query's terms, counted as substrings of the lowercased
    # text: that runs in C, and it finds terms inside identifiers
    # (`renderTable` contains `table`). These counts also give each term's
    # document frequency, so rarity does not depend on the tiers.
    fields = []
    lengths = {}
    document_frequency: Counter[str] = Counter()
    for node in candidates:
        name_terms = terms(node.name)
        name = {term: _name_count(name_terms, term) for term in query}
        qualified = _count_terms(f"{node.path} {qualified_name(node)}", query)
        signature = _count_terms(node.signature or "", query)
        text = f"{node.summary or ''} {node.body or ''}"
        body = _count_terms(text, query)
        fields.append((name, qualified, signature, body))
        lengths[node.id] = len(text)
        document_frequency.update(
            term
            for term in query
            if name[term] or qualified[term] or signature[term] or body[term]
        )

    total = max(len(candidates), 1)
    idf = {
        term: math.log(
            1
            + (total - document_frequency[term] + 0.5)
            / (document_frequency[term] + 0.5)
        )
        for term in query
    }
    raw_name = {word.lower() for word in re.findall(r"[\w.]+", question)}
    penalize_tests = not {"test", "tests", "testing", "spec"} & {
        word.lower() for word in re.findall(r"\w+", question)
    }
    top_idf = max(idf.values())
    # Body length in characters: proportional enough to words, and free.
    average_length = sum(lengths.values()) / total or 1.0

    def score_of(
        node: NodeV1,
        name: Mapping[str, float],
        qualified: Mapping[str, float],
        signature: Mapping[str, float],
        body: Mapping[str, float],
    ) -> float:
        score = 0.0
        matched = 0
        length = (
            1 - LENGTH_NORM + LENGTH_NORM * lengths[node.id] / average_length
        )
        for term in query:
            weighted = (
                NAME_WEIGHT * _saturate(name[term])
                + QUALIFIED_WEIGHT * _saturate(qualified[term])
                + SIGNATURE_WEIGHT * _saturate(signature[term])
                + BODY_WEIGHT * _saturate(body[term], length)
            )
            if weighted:
                matched += 1
                score += idf[term] * weighted
        if not score:
            return 0.0
        if node.name.lower() in raw_name or qualified_name(node).lower() in (
            raw_name
        ):
            score += NAME_WEIGHT * top_idf
        score *= (matched / len(query)) ** 0.5
        if node.kind == "file":
            score *= FILE_PENALTY
        if penalize_tests and _TEST_PATH.search(node.path):
            score *= TEST_PENALTY
        return float(score)

    # Cheap scores pick the shortlist; the tiered scores then decide the order.
    first_pass = [
        (score, node, counts)
        for node, counts in zip(candidates, fields, strict=True)
        if (score := score_of(node, *counts))
    ]
    first_pass.sort(key=lambda item: (-item[0], item[1].id))
    lexical: dict[str, float] = {}
    for _, node, (name, qualified, signature, body) in first_pass[
        :RESCORE_LIMIT
    ]:
        score = score_of(
            node,
            name,
            _tiered(f"{node.path} {qualified_name(node)}", qualified, query),
            _tiered(node.signature or "", signature, query),
            _tiered(f"{node.summary or ''} {node.body or ''}", body, query),
        )
        if score:
            lexical[node.id] = score
    if not lexical:
        return []

    seeds = dict(
        sorted(lexical.items(), key=lambda item: (-item[1], item[0]))[
            :PAGERANK_SEEDS
        ]
    )
    centrality = _personalized_pagerank(graph, seeds)
    top_lexical = max(lexical.values())
    top_centrality = max(centrality.values(), default=0.0) or 1.0
    nodes = {node.id: node for node in candidates}
    hits = [
        Hit(
            nodes[node_id],
            (1 - PAGERANK_SHARE) * score / top_lexical
            + PAGERANK_SHARE * centrality.get(node_id, 0.0) / top_centrality,
        )
        for node_id, score in lexical.items()
    ]
    return sorted(hits, key=lambda hit: (-hit.score, hit.node.id))


def _count_terms(text: str, query: list[str]) -> dict[str, int]:
    lowered = text.lower()
    return {term: lowered.count(term) for term in query}


def _name_count(words: list[str], term: str) -> float:
    """Tiered count of `term` among a name's words.

    An exact word counts in full. A word that starts with the term, or that
    the term starts with (`opt` for `option`), counts `PREFIX_WEIGHT`.
    """
    exact = words.count(term)
    if len(term) < MIN_PREFIX:
        return float(exact)
    related = sum(
        1
        for word in words
        if word != term
        and (
            word.startswith(term)
            or (len(word) >= MIN_PREFIX and term.startswith(word))
        )
    )
    return exact + PREFIX_WEIGHT * related


def _tiered(
    text: str, substrings: dict[str, int], query: list[str]
) -> dict[str, float]:
    """Tiered counts of the query's terms in a field's text.

    `substrings` are the first pass's counts of each term anywhere in the
    text. Of those, the exact words count in full, the words that merely
    start with the term count `PREFIX_WEIGHT`, and whatever is left (the
    term inside a longer word) counts `FRAGMENT_WEIGHT`.
    """
    words, spaced = _field_tokens(text)
    out = {}
    for term in query:
        exact = words.get(term, 0)
        longer = (
            spaced.count(" " + term) - exact if len(term) >= MIN_PREFIX else 0
        )
        fragment = max(0, substrings[term] - exact - longer)
        out[term] = exact + PREFIX_WEIGHT * longer + FRAGMENT_WEIGHT * fragment
    return out


def _saturate(frequency: float, length: float = 1.0) -> float:
    """Diminishing credit for repeats, in [0, 1).

    `length` is the field's length relative to the average (scaled by
    `LENGTH_NORM`): above 1, more repeats are needed for the same credit.
    """
    return frequency / (frequency + SATURATION * length) if frequency else 0.0


def _personalized_pagerank(
    graph: GraphV1, seeds: dict[str, float]
) -> dict[str, float]:
    """PageRank over use edges (both directions), restarting at the seeds."""
    neighbours: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.relation in STRUCTURAL_RELATIONS or edge.source == edge.target:
            continue
        neighbours.setdefault(edge.source, []).append(edge.target)
        neighbours.setdefault(edge.target, []).append(edge.source)
    # Walk only the seeds' neighbourhood: code more than two edges from any
    # match should not be lifted by it, and the walk stays cheap on large
    # graphs.
    local = set(seeds)
    frontier = set(seeds)
    for _ in range(NEIGHBOURHOOD_HOPS):
        frontier = {
            other
            for node_id in frontier
            for other in neighbours.get(node_id, ())
            if other not in local
        }
        local |= frontier
    neighbours = {
        node_id: [
            other for other in neighbours.get(node_id, ()) if other in local
        ]
        for node_id in local
    }
    total = sum(seeds.values())
    restart = {node_id: score / total for node_id, score in seeds.items()}
    rank_of = dict(restart)
    for _ in range(ITERATIONS):
        following: dict[str, float] = {}
        dangling = 0.0
        for node_id, value in rank_of.items():
            links = neighbours.get(node_id)
            if not links:
                dangling += value
                continue
            spread = DAMPING * value / len(links)
            for other in links:
                following[other] = following.get(other, 0.0) + spread
        # Restart mass, plus dangling mass, returns to the seeds only.
        back = (1 - DAMPING) + DAMPING * dangling
        for node_id, share in restart.items():
            following[node_id] = following.get(node_id, 0.0) + back * share
        rank_of = following
    return rank_of


def render_ask(
    graph: GraphV1,
    reader: SourceReader,
    question: str,
    *,
    limit: int = 8,
    source: bool = False,
    full: bool = False,
    scope: str | None = None,
    tools: Mapping[str, str] = CLI_TOOLS,
) -> str:
    """The top answers, or the callers view for a structural question.

    `tools` names the follow-up tools the answer suggests, as the caller's
    surface spells them (`CLI_TOOLS` or `MCP_TOOLS`).
    """
    for pattern, direction in _STRUCTURAL:
        match = pattern.search(question)
        if match:
            symbol = match.group(1).strip("`'\"?.,")
            seeds = find_symbols(graph, symbol)
            if seeds:
                return (
                    f"ask — {question!r} (structural: "
                    f"{'callers' if direction == 'in' else 'callees'} "
                    f"of {symbol})\n\n"
                    + render_callers(graph, seeds, direction=direction)
                )

    ranked = rank(graph, question, scope=scope)
    hits = ranked[:limit]
    header = f"ask — {question!r} (lexical, {len(hits)} results)"
    if not hits:
        return (
            f"{header}\n\nNo matching definitions.\n{_next_step(ranked, tools)}"
        )
    out = [header]
    for index, hit in enumerate(hits, start=1):
        node = hit.node
        out.append("")
        out.append(f"{index}. {describe(node)}  [{hit.score:.2f}]")
        if node.signature:
            out.append(f"   {' '.join(node.signature.split())}")
        if (
            node.summary
            and node.summary.strip()
            and (node.summary_state == "ready")
        ):
            out.append(f"   {node.summary.strip().splitlines()[0]}")
        if (
            (source and not full)
            and node.crux
            and node.summary_state == "ready"
        ):
            # The span the meaning pass picked as the definition's core,
            # cut when the body was last summarized (ready = unchanged).
            crux_start, _ = span_lines(node.crux.span)
            out.append(
                numbered(node.crux.code.split("\n"), crux_start, indent="     ")
            )
            out.append(f"     … crux of {node.span} (--full for all of it)")
        elif source or full:
            start, end = span_lines(node.span)
            lines = reader.span(node, None if full else CRUX_LINES)
            if lines:
                out.append(numbered(lines, start, indent="     "))
                hidden = (end - start + 1) - len(lines)
                if hidden > 0:
                    out.append(f"     … +{hidden} more lines (--full)")
    out.append("")
    out.append(_next_step(ranked, tools).rstrip("\n"))
    return "\n".join(out) + "\n"


def _next_step(ranked: list[Hit], tools: Mapping[str, str]) -> str:
    """The line after an answer: the tool to switch to, not a rewording.

    Agents whose first wording misses tend to reword it again and again;
    an exact search, a file's API, or the top hit's calls usually settles
    it in one call. When few definitions match at all (`ranked` is every
    match, not just those shown), the words barely occur in the code, so
    the advice is firmer.
    """
    top = ranked[0].node if ranked else None
    file_api = tools["file_api"].format(file=top.path if top else "<file>")
    calls = tools["calls_out"].format(symbol=top.id if top else "<symbol>")
    count = len(ranked)
    if count <= FEW_HITS:
        found = (
            "no hits"
            if not count
            else f"only {count} hit{'' if count == 1 else 's'}"
        )
        return (
            f"[bytely] {found}; don't re-ask with new wording, switch tool: "
            f"{tools['find_all']} for every occurrence · {file_api} for a "
            f"file's API · {calls} for what it calls.\n"
        )
    return (
        "[bytely] Not here? Don't reword the question; switch tool: "
        f"{tools['find_all']} finds every occurrence · {file_api} lists "
        f"the file · {calls} shows what the top hit calls.\n"
    )
