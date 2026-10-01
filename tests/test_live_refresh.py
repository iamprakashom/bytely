"""A live MCP session keeps answering correctly while the code changes.

Each language's small project is edited step by step under one running
`bytely mcp` process, with no manual rebuild. After every edit the test
asks MCP again, asserts the exact call edges the edit should produce or
remove, and has the server compare its refreshed graph with an uncached
rebuild (`bytely_check_freshness`, `no_cache`). Known edges and rebuild
equivalence are both checked: two builds can reproduce the same bug.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from bytely.graph.build import build_graph
from bytely.graph.refresh import build_lock
from bytely.graph.write import read_graph
from bytely.mcp.client import McpError, McpSession

CALLER_LINE = re.compile(r"calls ← (\S+) · \w+ · ([^:]+):L")


@dataclass(frozen=True)
class Project:
    """A tiny two-module project and the edits made to it, per language."""

    files: dict[str, str]
    util: str  # where `helper` is defined
    app: str  # where `run` calls it
    tools: tuple[str, str]  # a second module that defines `assist`
    import_from: tuple[str, str]  # app's import: util module -> tools module
    extra: tuple[str, str]  # an added file whose `also` calls tools' assist
    registry: tuple[str, str] | None = None  # (file, line) to declare a module


PROJECTS = {
    "python": Project(
        files={
            "util.py": "def helper():\n    return 1\n",
            "app.py": "from util import helper\n\n\n"
            "def run():\n    return helper()\n",
        },
        util="util.py",
        app="app.py",
        tools=("tools.py", "def assist():\n    return 3\n"),
        import_from=("from util import", "from tools import"),
        extra=(
            "extra.py",
            "from tools import assist\n\n\ndef also():\n    return assist()\n",
        ),
    ),
    "rust": Project(
        files={
            "Cargo.toml": '[package]\nname = "live"\nversion = "0.1.0"\n',
            "src/lib.rs": "mod app;\nmod util;\n",
            "src/util.rs": "pub fn helper() -> u32 {\n    1\n}\n",
            "src/app.rs": "use crate::util::helper;\n\n"
            "pub fn run() -> u32 {\n    helper()\n}\n",
        },
        util="src/util.rs",
        app="src/app.rs",
        tools=("src/tools.rs", "pub fn assist() -> u32 {\n    3\n}\n"),
        import_from=("crate::util::", "crate::tools::"),
        extra=(
            "src/extra.rs",
            (
                "use crate::tools::assist;\n\n"
                "pub fn also() -> u32 {\n    assist()\n}\n"
            ),
        ),
        registry=("src/lib.rs", "mod {name};\n"),
    ),
    "javascript": Project(
        files={
            "util.mjs": "export function helper() {\n  return 1;\n}\n",
            "app.mjs": "import { helper } from './util.mjs';\n\n"
            "export function run() {\n  return helper();\n}\n",
        },
        util="util.mjs",
        app="app.mjs",
        tools=("tools.mjs", "export function assist() {\n  return 3;\n}\n"),
        import_from=("'./util.mjs'", "'./tools.mjs'"),
        extra=(
            "extra.mjs",
            (
                "import { assist } from './tools.mjs';\n\n"
                "export function also() {\n  return assist();\n}\n"
            ),
        ),
    ),
}


def _write(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")


def _replace(repo: Path, path: str, old: str, new: str) -> None:
    text = (repo / path).read_text("utf-8")
    assert old in text, f"{old!r} not in {path}"
    _write(repo, path, text.replace(old, new))


def _declare(repo: Path, project: Project, path: str) -> None:
    """Declare a new module where the language needs it (Rust's `mod`)."""
    if project.registry is not None:
        registry, line = project.registry
        name = Path(path).stem
        text = (repo / registry).read_text("utf-8")
        _write(repo, registry, text + line.format(name=name))


def _undeclare(repo: Path, project: Project, path: str) -> None:
    if project.registry is not None:
        registry, line = project.registry
        _replace(repo, registry, line.format(name=Path(path).stem), "")


def _calls(repo: Path) -> set[tuple[str, str, str, str]]:
    """The graph's call edges as (caller file, caller, callee file, callee)."""
    graph = read_graph(str(repo / "bytely"))
    assert graph is not None
    by_id = {node.id: node for node in graph.nodes}
    edges = set()
    for edge in graph.edges:
        if edge.relation != "calls":
            continue
        source, target = by_id.get(edge.source), by_id.get(edge.target)
        if source is not None and target is not None:
            edges.add((source.path, source.name, target.path, target.name))
    return edges


def _callers(mcp: McpSession, symbol: str) -> set[tuple[str, str]]:
    """Who calls `symbol`, as MCP answers it: {(caller, file)}."""
    answer = mcp.call("bytely_trace_calls", {"symbol": symbol})
    return set(CALLER_LINE.findall(answer))


def _assert_matches_uncached_rebuild(mcp: McpSession) -> None:
    verdict = mcp.call("bytely_check_freshness", {"no_cache": True})
    assert verdict.startswith("fresh:"), verdict


@pytest.mark.parametrize("language", sorted(PROJECTS))
def test_live_session_follows_every_edit(tmp_path: Path, language: str) -> None:
    project = PROJECTS[language]
    repo = tmp_path / "repo"
    for path, text in project.files.items():
        _write(repo, path, text)
    build_graph(str(repo))
    command = [sys.executable, "-m", "bytely", "mcp"]
    tools_path, tools_text = project.tools
    extra_path, extra_text = project.extra

    with McpSession(command, repo) as mcp:
        assert _callers(mcp, "helper") == {("run", project.app)}
        assert (project.app, "run", project.util, "helper") in _calls(repo)

        # 1. A body edit keeps every relationship.
        _replace(repo, project.util, "1", "2")
        assert _callers(mcp, "helper") == {("run", project.app)}
        _assert_matches_uncached_rebuild(mcp)

        # 2. Renaming a function and its callers moves the edge.
        for path in (project.util, project.app):
            _replace(repo, path, "helper", "assist")
        with pytest.raises(McpError, match="No symbol"):
            mcp.call("bytely_trace_calls", {"symbol": "helper"})
        assert _callers(mcp, "assist") == {("run", project.app)}
        calls = _calls(repo)
        assert (project.app, "run", project.util, "assist") in calls
        assert not any(callee == "helper" for *_, callee in calls)
        _assert_matches_uncached_rebuild(mcp)

        # 3. Changing an import retargets the call to the other module.
        _write(repo, tools_path, tools_text)
        _declare(repo, project, tools_path)
        _replace(repo, project.app, *project.import_from)
        assert ("run", project.app) in _callers(mcp, "assist")
        calls = _calls(repo)
        assert (project.app, "run", tools_path, "assist") in calls
        assert (project.app, "run", project.util, "assist") not in calls
        _assert_matches_uncached_rebuild(mcp)

        # 4. An added file's calls appear.
        _write(repo, extra_path, extra_text)
        _declare(repo, project, extra_path)
        assert ("also", extra_path) in _callers(mcp, "assist")
        assert (extra_path, "also", tools_path, "assist") in _calls(repo)
        _assert_matches_uncached_rebuild(mcp)

        # 5. A deleted file's definitions and calls disappear.
        (repo / extra_path).unlink()
        _undeclare(repo, project, extra_path)
        assert ("also", extra_path) not in _callers(mcp, "assist")
        assert not any(path == extra_path for path, *_ in _calls(repo))
        _assert_matches_uncached_rebuild(mcp)

        # 6. A burst of saves: only the final state matters.
        app = (repo / project.app).read_text("utf-8")
        for index in range(5):
            _write(repo, project.app, app + "\n" * (index + 1))
        assert ("run", project.app) in _callers(mcp, "assist")
        _assert_matches_uncached_rebuild(mcp)


def test_fresh_builds_of_the_same_tree_are_byte_identical(
    tmp_path: Path,
) -> None:
    trees = []
    for name in ("first", "second"):
        repo = tmp_path / name
        for language in sorted(PROJECTS):
            for path, text in PROJECTS[language].files.items():
                _write(repo / language, path, text)
        build_graph(str(repo))
        trees.append(repo / "bytely")

    def outputs(root: Path) -> dict[str, bytes]:
        # The cache holds stats and timestamps that differ by design.
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file() and "cache" not in path.relative_to(root).parts
        }

    first, second = (outputs(root) for root in trees)
    assert first.keys() == second.keys()
    assert [name for name in first if first[name] != second[name]] == []


def test_invalid_utf8_file_is_skipped_until_it_is_fixed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, "good.py", "def good():\n    return 1\n")
    (repo / "bad.py").write_bytes(b"def bad():\n    return '\xff'\n")

    build_graph(str(repo))
    graph = read_graph(str(repo / "bytely"))
    assert graph is not None
    paths = {node.path for node in graph.nodes}
    assert "good.py" in paths
    assert "bad.py" not in paths

    _write(repo, "bad.py", "def bad():\n    return 2\n")
    build_graph(str(repo))
    graph = read_graph(str(repo / "bytely"))
    assert graph is not None
    assert "bad" in {node.name for node in graph.nodes}


def test_live_query_waits_for_a_build_holding_the_lock(tmp_path: Path) -> None:
    project = PROJECTS["python"]
    repo = tmp_path / "repo"
    for path, text in project.files.items():
        _write(repo, path, text)
    build_graph(str(repo))

    with McpSession([sys.executable, "-m", "bytely", "mcp"], repo) as mcp:
        held, release = threading.Event(), threading.Event()

        def hold_lock() -> None:
            with build_lock(repo / "bytely"):
                held.set()
                release.wait(30)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        try:
            assert held.wait(30)
            for path in (project.util, project.app):
                _replace(repo, path, "helper", "assist")
            threading.Timer(1.0, release.set).start()
            started = time.monotonic()
            callers = _callers(mcp, "assist")
            waited = time.monotonic() - started
        finally:
            release.set()
            holder.join()
        assert callers == {("run", project.app)}
        assert waited >= 0.8  # it waited for the lock rather than racing it
        _assert_matches_uncached_rebuild(mcp)
