"""Real repositories from the benchmark corpus: known facts and live edits.

For each pinned repository with `[*.expect]` facts in bench/corpus.toml
that has been fetched (`python bench/fetch.py NAME`), on a fresh copy:

1. Build; every expected call and import edge exists, every `absent` one
   does not; a second build reuses every extraction.
2. Through one live MCP session, with no manual rebuild: add a callee and
   its caller, rename the callee, add a file, delete it; after each edit
   the session answers with the new relationships, and its refreshed graph
   matches an uncached rebuild.
3. The facts still hold after the edits.

Marked `integration` (deselected by default; run with `-m integration`).
"""

from __future__ import annotations

import re
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from bytely.graph.build import build_graph
from bytely.graph.write import read_graph
from bytely.mcp.client import McpError, McpSession

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = tomllib.loads((REPO_ROOT / "bench" / "corpus.toml").read_text("utf-8"))
CACHE = REPO_ROOT / ".bench-cache"
OVERLOAD = re.compile(r"~\d+$")
CALLER_LINE = re.compile(r"calls ← (\S+) · \w+ · ([^:]+):L")

# Per language: a callee/caller pair to add, and the callee's new name.
PAIRS = {
    "python": (
        "\n\ndef bench_callee():\n    return 1\n\n\n"
        "def bench_caller():\n    return bench_callee()\n",
        ("bench_callee", "bench_caller", "bench_renamed"),
        ".py",
    ),
    "rust": (
        "\nfn bench_callee() -> u32 {\n    1\n}\n\n"
        "fn bench_caller() -> u32 {\n    bench_callee()\n}\n",
        ("bench_callee", "bench_caller", "bench_renamed"),
        ".rs",
    ),
    "javascript": (
        "\nfunction benchCallee() {\n  return 1;\n}\n\n"
        "function benchCaller() {\n  return benchCallee();\n}\n",
        ("benchCallee", "benchCaller", "benchRenamed"),
        ".js",
    ),
}

REPOS = sorted(name for name, repo in CORPUS.items() if repo.get("expect"))


def _tree(name: str) -> Path:
    repo = CORPUS[name]
    checkout = CACHE / name
    return checkout / repo["subdir"] if "subdir" in repo else checkout


def _key(node_id: str) -> str:
    """A node id without its overload suffix: `core.py#main~3` -> `#main`."""
    return OVERLOAD.sub("", node_id)


def _edges(root: Path) -> dict[str, set[tuple[str, str]]]:
    graph = read_graph(str(root / "bytely"))
    assert graph is not None
    edges: dict[str, set[tuple[str, str]]] = {}
    for edge in graph.edges:
        edges.setdefault(edge.relation, set()).add(
            (_key(edge.source), _key(edge.target))
        )
    return edges


def _pair(fact: str) -> tuple[str, str]:
    source, target = (part.strip() for part in fact.split("->"))
    return source, target


def _assert_facts(root: Path, expect: dict[str, Any]) -> None:
    edges = _edges(root)
    missing = [
        fact
        for relation in ("calls", "imports")
        for fact in expect.get(relation, [])
        if _pair(fact) not in edges.get(relation, set())
    ]
    assert missing == [], f"expected edges missing: {missing}"
    present = [
        fact
        for fact in expect.get("absent", [])
        if _pair(fact) in edges.get("calls", set())
    ]
    assert present == [], f"edges that must not exist: {present}"


def _callers(mcp: McpSession, symbol: str) -> set[tuple[str, str]]:
    answer = mcp.call("bytely_trace_calls", {"symbol": symbol})
    return set(CALLER_LINE.findall(answer))


def _assert_matches_uncached_rebuild(mcp: McpSession) -> None:
    verdict = mcp.call("bytely_check_freshness", {"no_cache": True})
    assert verdict.startswith("fresh:"), verdict


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


@pytest.mark.parametrize("name", REPOS)
def test_pinned_repository_facts_and_live_edits(
    tmp_path: Path, name: str
) -> None:
    if not _tree(name).is_dir():
        pytest.skip(f"{name} not fetched; run python bench/fetch.py {name}")
    repo = CORPUS[name]
    expect = repo["expect"]
    work = tmp_path / name
    shutil.copytree(
        _tree(name), work, ignore=shutil.ignore_patterns(".git", "bytely")
    )

    first = build_graph(str(work))
    _assert_facts(work, expect)
    second = build_graph(str(work))
    assert (second.cache_hits, second.cache_misses) == (first.files, 0)

    snippet, (callee, caller, renamed), extension = PAIRS[repo["language"]]
    # Edit a file the facts are about: the caller side of the first fact.
    target = _pair(expect["calls"][0])[0].split("#")[0]
    added = Path(target).with_name(f"bench_added{extension}").as_posix()

    with McpSession([sys.executable, "-m", "bytely", "mcp"], work) as mcp:
        fact_callee = _pair(expect["calls"][0])[1].split("#")[1]
        fact_caller = _pair(expect["calls"][0])[0].split("#")[1]
        assert fact_caller.split(".")[-1] in {
            who.split(".")[-1] for who, _ in _callers(mcp, fact_callee)
        }

        # A new callee and caller in an existing file.
        _append(work / target, snippet)
        assert (caller, target) in _callers(mcp, callee)
        _assert_matches_uncached_rebuild(mcp)

        # Rename the callee where it is defined and called.
        text = (work / target).read_text("utf-8")
        (work / target).write_text(
            text.replace(callee, renamed), encoding="utf-8", newline="\n"
        )
        with pytest.raises(McpError, match="No symbol"):
            mcp.call("bytely_trace_calls", {"symbol": callee})
        assert (caller, target) in _callers(mcp, renamed)
        _assert_matches_uncached_rebuild(mcp)

        # A new file, then its deletion.
        (work / added).write_text(
            snippet.lstrip().replace(callee, "bench_added_callee"),
            encoding="utf-8",
            newline="\n",
        )
        assert (caller, added) in _callers(mcp, "bench_added_callee")
        _assert_matches_uncached_rebuild(mcp)
        (work / added).unlink()
        with pytest.raises(McpError, match="No symbol"):
            mcp.call("bytely_trace_calls", {"symbol": "bench_added_callee"})
        _assert_matches_uncached_rebuild(mcp)

    _assert_facts(work, expect)
