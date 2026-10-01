from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

import bytely.graph.extract as extract_module
import bytely.graph.extract_cache as extract_cache_module
from bytely.cli import main
from bytely.graph.build import build_graph
from bytely.graph.check import check_graph
from bytely.graph.extract import depth_extensions, extract_file
from bytely.graph.extract_cache import (
    EXTRACTOR_MODULES,
    GRAMMAR_MODULES,
    extractor_fingerprint,
)
from bytely.graph.invariants import validate_graph
from bytely.graph.resolve import resolve_edges
from bytely.graph.root import find_bytely_root
from bytely.graph.scopes import discover_scopes
from bytely.graph.types import EdgeV1, GraphV1, NodeV1
from bytely.graph.write import read_graph
from bytely.ingest.fs import walk_dir


def test_graph_invariants_reject_duplicate_node_ids() -> None:
    node = NodeV1(
        id="src/lib.rs#LIMIT",
        name="LIMIT",
        kind="constant",
        path="src/lib.rs",
        span="L1-L1",
        body_hash="0123456789abcdef",
    )

    with pytest.raises(ValueError, match="duplicate node id"):
        validate_graph(GraphV1(nodes=[node, node]))


@pytest.mark.parametrize(
    ("relation", "allowed"),
    [
        ("imports", True),
        ("extends", True),
        ("implements", True),
        ("calls", False),
        ("contains", False),
    ],
)
def test_only_imports_and_heritage_may_target_a_name(
    relation: str, allowed: bool
) -> None:
    # P4: an unresolved import or base keeps an edge to its raw name, as the
    # reference implementation; any other relation must still point at a node.
    node = NodeV1(
        id="app.py",
        name="app.py",
        kind="file",
        path="app.py",
        span="L1-L1",
        body_hash="0",
    )
    edge = EdgeV1(source="app.py", target="typing", relation=relation)  # type: ignore[arg-type]
    graph = GraphV1(nodes=[node], edges=[edge])

    if allowed:
        validate_graph(graph)
    else:
        with pytest.raises(ValueError, match="missing target node typing"):
            validate_graph(graph)
    orphan = EdgeV1(source="nope", target="app.py", relation="imports")
    with pytest.raises(ValueError, match="missing source node"):
        validate_graph(GraphV1(nodes=[node], edges=[orphan]))


@pytest.mark.parametrize(
    ("path", "source", "name", "kind"),
    [
        (
            "sample.ts",
            "export function greet() { return 'hi'; }",
            "greet",
            "function",
        ),
        ("sample.py", "def greet():\n    return 'hi'", "greet", "function"),
        (
            "sample.go",
            'package main\nfunc greet() string { return "hi" }',
            "greet",
            "function",
        ),
        (
            "Sample.java",
            'class Sample { String greet() { return "hi"; } }',
            "greet",
            "method",
        ),
        (
            "sample.php",
            "<?php function greet(): string { return 'hi'; }",
            "greet",
            "function",
        ),
        ("sample.kt", 'fun greet(): String = "hi"', "greet", "function"),
        (
            "sample.swift",
            'func greet() -> String { "hi" }',
            "greet",
            "function",
        ),
        (
            "sample.rs",
            'fn greet() -> &\'static str { "hi" }',
            "greet",
            "function",
        ),
        ("sample.c", "int greet(void) { return 1; }", "greet", "function"),
        ("sample.cpp", "int greet() { return 1; }", "greet", "function"),
        (
            "Sample.cs",
            'class Sample { string Greet() { return "hi"; } }',
            "Greet",
            "method",
        ),
        ("constants.py", "MAX_RETRIES = 3", "MAX_RETRIES", "constant"),
        ("state.py", "current_count = 0", "current_count", "variable"),
        (
            "constants.ts",
            "export const MAX_RETRIES = 3;",
            "MAX_RETRIES",
            "constant",
        ),
        ("state.ts", "let currentCount = 0;", "currentCount", "variable"),
        (
            "constants.go",
            "package main\nconst (\nMaxRetries = 3\n)",
            "MaxRetries",
            "constant",
        ),
        (
            "state.go",
            "package main\nvar currentCount = 0",
            "currentCount",
            "variable",
        ),
        (
            "constants.rs",
            "const MAX_RETRIES: usize = 3;",
            "MAX_RETRIES",
            "constant",
        ),
        (
            "state.rs",
            "static CURRENT_COUNT: usize = 0;",
            "CURRENT_COUNT",
            "variable",
        ),
        (
            "constants.kt",
            "const val MAX_RETRIES = 3",
            "MAX_RETRIES",
            "constant",
        ),
        ("state.kt", "var currentCount = 0", "currentCount", "variable"),
        ("state.c", "int current_count = 0;", "current_count", "variable"),
        (
            "constants.c",
            "const int MAX_RETRIES = 3;",
            "MAX_RETRIES",
            "constant",
        ),
        ("state.cpp", "int currentCount = 0;", "currentCount", "variable"),
        (
            "constants.cpp",
            "constexpr int MAX_RETRIES = 3;",
            "MAX_RETRIES",
            "constant",
        ),
        ("constants.swift", "let maxRetries = 3", "maxRetries", "constant"),
        ("state.swift", "var currentCount = 0", "currentCount", "variable"),
        (
            "constants.cs",
            "const int MAX_RETRIES = 3;",
            "MAX_RETRIES",
            "constant",
        ),
        ("state.cs", "int currentCount = 0;", "currentCount", "variable"),
    ],
)
def test_extracts_symbols_for_packaged_languages(
    path: str, source: str, name: str, kind: str
) -> None:
    result = extract_file(path, source)

    assert result.nodes[0].kind == "file"
    assert any(node.name == name and node.kind == kind for node in result.nodes)


def test_c_function_prototypes_are_not_extracted_as_variables() -> None:
    result = extract_file("api.c", "int greet(void);")

    assert not any(node.name == "greet" for node in result.nodes)


def test_extracts_each_c_global_declarator() -> None:
    result = extract_file("globals.c", "int first = 1, second = 2;")

    assert {node.name for node in result.nodes} >= {"first", "second"}


def test_extracts_c_global_function_pointer() -> None:
    result = extract_file("callbacks.c", "int (*handler)(void) = 0;")

    assert any(
        node.name == "handler" and node.kind == "variable"
        for node in result.nodes
    )


def test_extractor_fingerprint_tracks_every_source_and_grammar_version() -> (
    None
):
    sources = {"bytely.graph.extract": b"v1", "bytely.util.id": b"ids"}
    versions = {"tree_sitter": "0.26.0", "tree_sitter_python": "0.25.0"}
    base = extractor_fingerprint(sources, versions)

    assert extractor_fingerprint(dict(reversed(sources.items())), versions) == (
        base
    )
    assert (
        extractor_fingerprint(
            {**sources, "bytely.graph.extract": b"v2"}, versions
        )
        != base
    )
    assert (
        extractor_fingerprint(
            sources, {**versions, "tree_sitter_python": "0.25.1"}
        )
        != base
    )


def test_every_grammar_module_used_by_the_extractor_is_fingerprinted() -> None:
    # A17: a grammar imported outside GRAMMAR_LOADERS (as the lazily loaded
    # JavaScript grammar is) must still invalidate the cache on upgrade.
    extractor_source = Path(extract_module.__file__).read_text("utf-8")
    used = set(re.findall(r"\b(tree_sitter_[a-z_]+)\b", extractor_source))
    used.discard("tree_sitter_")
    assert used <= set(GRAMMAR_MODULES)
    assert set(EXTRACTOR_MODULES) >= {
        "bytely.graph.extract",
        "bytely.graph.generic",
        "bytely.graph.extract_cache",
    }


def test_changed_extractor_fingerprint_invalidates_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", "utf-8")
    build_graph(str(tmp_path))
    assert build_graph(str(tmp_path)).cache_hits == 1

    # Editing extract.py, or upgrading a grammar, changes the fingerprint;
    # no manual CACHE_VERSION bump is needed any more.
    monkeypatch.setattr(
        extract_cache_module,
        "current_extractor_fingerprint",
        lambda: "a-different-extractor",
    )
    rebuilt = build_graph(str(tmp_path))
    assert (rebuilt.cache_hits, rebuilt.cache_misses) == (0, 1)


def test_cached_edges_serialize_like_asdict() -> None:
    from dataclasses import asdict

    from bytely.graph.extract import RawEdge
    from bytely.graph.extract_cache import _edge_dict

    edge = RawEdge(
        source="a.py#f", relation="calls", file="a.py", kinds=["function"]
    )
    data = _edge_dict(edge)
    assert data == asdict(edge)
    assert data["kinds"] is not edge.kinds


def _write_aged(path: Path, text: str, mtime_ns: int | None = None) -> None:
    """Write a file whose mtime is old enough for a build to trust it."""
    path.write_text(text, "utf-8")
    old = mtime_ns if mtime_ns is not None else time.time_ns() - 60 * 10**9
    os.utime(path, ns=(old, old))


def test_build_of_an_unchanged_tree_reads_no_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_aged(tmp_path / "a.py", "def a():\n    return 1\n")
    first = build_graph(str(tmp_path))

    def no_reads(self: Path) -> bytes:
        raise AssertionError(f"read {self}")

    monkeypatch.setattr(Path, "read_bytes", no_reads)
    again = build_graph(str(tmp_path))
    assert (again.files, again.nodes, again.edges) == (
        first.files,
        first.nodes,
        first.edges,
    )


def test_build_reads_only_files_whose_stat_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_aged(tmp_path / "a.py", "def a():\n    return 1\n")
    _write_aged(tmp_path / "b.py", "def b():\n    return 2\n")
    build_graph(str(tmp_path))
    (tmp_path / "b.py").write_text("def b():\n    return 3\n", "utf-8")

    read: list[str] = []
    original = Path.read_bytes

    def tracking(self: Path) -> bytes:
        read.append(self.name)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", tracking)
    result = build_graph(str(tmp_path))
    assert "a.py" not in read
    assert "b.py" in read
    assert (result.cache_hits, result.cache_misses) == (1, 1)


def test_build_rewrites_a_deleted_index(
    tmp_path: Path,
) -> None:
    _write_aged(tmp_path / "a.py", "def a():\n    return 1\n")
    build_graph(str(tmp_path))
    index = tmp_path / "bytely" / "INDEX.md"
    index.unlink()
    build_graph(str(tmp_path))
    assert index.is_file()


def test_build_restores_a_card_deleted_or_edited_by_hand(
    tmp_path: Path,
) -> None:
    _write_aged(tmp_path / "a.py", "def a():\n    return 1\n")
    _write_aged(tmp_path / "b.py", "def b():\n    return 2\n")
    build_graph(str(tmp_path))
    card_a = tmp_path / "bytely" / "a.md"
    card_b = tmp_path / "bytely" / "b.md"
    original = card_b.read_text("utf-8")
    card_a.unlink()
    card_b.write_text("edited by hand\n", "utf-8")

    build_graph(str(tmp_path))
    assert card_a.is_file()
    assert card_b.read_text("utf-8") == original


def _count_full_builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count builds that do the work (the no-change shortcut writes none)."""
    import bytely.graph.build as build_module

    calls: list[int] = []
    original = build_module.write_outputs

    def counting(*args: object, **kwargs: object) -> None:
        calls.append(1)
        original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(build_module, "write_outputs", counting)
    return calls


def test_stale_card_that_is_the_index_on_this_filesystem_is_kept(
    tmp_path: Path,
) -> None:
    # A graph built before `index.js` got `index.js.md` listed `index.md`
    # as a card; on Windows and macOS that path *is* INDEX.md.
    _write_aged(tmp_path / "index.js", "function a() { return 1; }\n")
    build_graph(str(tmp_path))
    manifest = tmp_path / "bytely" / "cache" / "cards.json"
    names = json.loads(manifest.read_text("utf-8"))
    manifest.write_text(json.dumps([*names, "index.md"]), "utf-8")
    _write_aged(tmp_path / "index.js", "function b() { return 2; }\n")

    build_graph(str(tmp_path))
    assert (tmp_path / "bytely" / "INDEX.md").is_file()
    assert (tmp_path / "bytely" / "index.js.md").is_file()


def test_upgraded_build_code_redoes_an_unchanged_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bytely.graph.build as build_module

    _write_aged(tmp_path / "a.py", "def a():\n    return 1\n")
    build_graph(str(tmp_path))
    full = _count_full_builds(monkeypatch)
    build_graph(str(tmp_path))
    assert full == []  # unchanged: the shortcut

    monkeypatch.setattr(
        build_module, "pipeline_fingerprint", lambda: "a-newer-bytely"
    )
    build_graph(str(tmp_path))
    assert full == [1]


def test_new_project_manifest_updates_scopes_without_source_changes(
    tmp_path: Path,
) -> None:
    (tmp_path / "web").mkdir()
    _write_aged(tmp_path / "web" / "a.js", "function a() { return 1; }\n")
    build_graph(str(tmp_path))
    (tmp_path / "web" / "package.json").write_text("{}", "utf-8")

    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert "web/" in {scope.prefix for scope in graph.scopes}


def test_output_formats_may_be_a_one_shot_iterator(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", "utf-8")
    build_graph(str(tmp_path), output_formats=iter(["markdown"]))
    assert (tmp_path / "bytely" / "INDEX.md").is_file()
    assert (tmp_path / "bytely" / "a.md").is_file()


def test_unchanged_build_reports_the_meaning_a_full_build_would(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bytely.graph.build as build_module

    _write_aged(tmp_path / "a.py", "def a():\n    return 1\n")
    build_graph(str(tmp_path))
    full = _count_full_builds(monkeypatch)
    shortcut = build_graph(str(tmp_path))
    assert full == []

    monkeypatch.setattr(build_module, "pipeline_fingerprint", lambda: "x")
    rebuilt = build_graph(str(tmp_path))
    assert full == [1]
    assert shortcut.meaning is not None
    assert shortcut.meaning == rebuilt.meaning


def test_card_names_never_collide_case_insensitively() -> None:
    from bytely.graph.outputs import card_paths

    cards = card_paths(["index.js", "src/Util.py", "src/util.js", "a.py"])
    assert cards["index.js"] == "index.js.md"  # INDEX.md is the root index
    assert cards["src/Util.py"] == "src/Util.py.md"
    assert cards["src/util.js"] == "src/util.js.md"
    assert cards["a.py"] == "a.md"


def test_build_rereads_a_same_size_edit_made_right_after_a_build(
    tmp_path: Path,
) -> None:
    # Saved within its mtime's resolution of the build, a same-size edit
    # can keep the same stat; a fresh file is never trusted by stat.
    source = tmp_path / "a.py"
    source.write_text("def a():\n    return 1\n", "utf-8")
    stamp = source.stat().st_mtime_ns
    build_graph(str(tmp_path))
    source.write_text("def b():\n    return 1\n", "utf-8")
    os.utime(source, ns=(stamp, stamp))

    build_graph(str(tmp_path))
    names = {n.name for n in read_graph(str(tmp_path / "bytely")).nodes}
    assert "b" in names
    assert "a" not in names


def test_check_reads_every_file_even_when_its_stat_is_unchanged(
    tmp_path: Path,
) -> None:
    # A tool that restores mtimes can hide a same-size edit from the
    # stat shortcut; `check` compares against a rebuild that reads files.
    source = tmp_path / "a.py"
    _write_aged(source, "def a():\n    return 1\n")
    stamp = source.stat().st_mtime_ns
    build_graph(str(tmp_path))
    _write_aged(source, "def b():\n    return 1\n", stamp)

    fresh, message = check_graph(str(tmp_path))
    assert not fresh, message


def test_check_no_cache_catches_a_stale_cache_that_check_calls_fresh(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", "utf-8")
    runner = CliRunner()
    assert runner.invoke(main, ["build", str(tmp_path)]).exit_code == 0

    # Simulate the A17 incident: the graph and the cache agree with each
    # other but not with what the extractor produces today.
    graph_file = tmp_path / "bytely" / ".graph" / "wiring.json"
    cache_file = tmp_path / "bytely" / "cache" / "extractions-v1.json"
    for path in (graph_file, cache_file):
        text = path.read_text("utf-8")
        assert text.count('"name": "a"') + text.count('"name":"a"') >= 1
        path.write_text(
            text.replace('"name": "a"', '"name": "stale"').replace(
                '"name":"a"', '"name":"stale"'
            ),
            "utf-8",
        )

    cached = runner.invoke(main, ["check", str(tmp_path)])
    assert cached.exit_code == 0, cached.output
    cold = runner.invoke(main, ["check", "--no-cache", str(tmp_path)])
    assert cold.exit_code == 1
    assert "stale" in cold.output


def test_node_header_fields_follow_the_reference_rules() -> None:
    # P10: the reference implementation's `chars` is JavaScript `string.length`
    # (UTF-16 code units), so an astral character counts twice; parity fixtures
    # are ASCII and cannot show it.
    source = 'NAME = "\U0001f600"\n\n\nclass Widget(Base):\n    pass\n'
    nodes = {node.id: node for node in extract_file("u.py", source).nodes}

    assert nodes["u.py"].chars == len(source) + 1
    assert nodes["u.py"].signature is None
    # Header up to the body, one trailing `:`/`{`/`=`/`=>` stripped.
    assert nodes["u.py#Widget"].signature == "class Widget(Base)"
    assert nodes["u.py#Widget"].origin == "ast"
    assert nodes["u.py#Widget"].chars is None

    js = extract_file(
        "m.js",
        "export const run = (a) =>\n  a;\nfunction hidden() {}\n",
    ).nodes
    by_id = {node.id: node for node in js}
    assert by_id["m.js#run"].signature == "run = (a)"
    assert by_id["m.js#run"].exported is True
    assert by_id["m.js#hidden"].exported is False


def test_warm_build_skips_rewriting_an_unchanged_extraction_cache(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", "utf-8")
    (tmp_path / "b.py").write_text("def b():\n    return 2\n", "utf-8")
    cache_file = tmp_path / "bytely" / "cache" / "extractions-v1.json"

    build_graph(str(tmp_path))
    first = cache_file.stat().st_mtime_ns

    # All hits: nothing to write (A14 follow-up; the rewrite cost ~0.75 s
    # on a 576-file tree).
    warm = build_graph(str(tmp_path))
    assert (warm.cache_hits, warm.cache_misses) == (2, 0)
    assert cache_file.stat().st_mtime_ns == first

    # A changed file and a deleted file must both reach the cache on disk.
    (tmp_path / "a.py").write_text("def a():\n    return 3\n", "utf-8")
    (tmp_path / "b.py").unlink()
    changed = build_graph(str(tmp_path))
    assert (changed.cache_hits, changed.cache_misses) == (0, 1)
    rebuilt = build_graph(str(tmp_path))
    assert (rebuilt.cache_hits, rebuilt.cache_misses) == (1, 0)
    assert '"b.py"' not in cache_file.read_text("utf-8")


def test_build_hashes_crlf_sources_as_they_are_on_disk(tmp_path: Path) -> None:
    source = b"def greet():\r\n    return 1\r\n"
    (tmp_path / "app.py").write_bytes(source)

    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None
    nodes = {node.id: node for node in graph.nodes}

    # Hashes are the full SHA-256 of the exact bytes, as the reference
    # implementation's; universal-newline reading would silently hash an LF copy
    # instead.
    assert nodes["app.py"].body_hash == hashlib.sha256(source).hexdigest()
    assert (
        nodes["app.py#greet"].body_hash
        == hashlib.sha256(b"def greet():\r\n    return 1").hexdigest()
    )
    assert nodes["app.py"].span == "L1-L3"


def test_incremental_build_resolves_imports_and_detects_staleness(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    package = project / "pkg"
    package.mkdir(parents=True)
    (package / "main.py").write_text(
        "from .helper import greet\n\ndef run():\n    return greet()\n",
        encoding="utf-8",
    )
    helper = package / "helper.py"
    helper.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")

    first = build_graph(str(project))
    assert first.cache_misses == 2
    assert first.cache_hits == 0

    second = build_graph(str(project))
    assert second.cache_hits == 2
    assert second.cache_misses == 0
    graph = Path(first.graph_json).read_text(encoding="utf-8")
    assert '"relation": "imports"' in graph
    assert '"target": "pkg/helper.py"' in graph
    assert check_graph(str(project)) == (True, "Graph is up to date.")

    helper.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    fresh, message = check_graph(str(project))
    assert not fresh
    assert "stale" in message

    rebuilt = build_graph(str(project))
    assert rebuilt.cache_hits == 1
    assert rebuilt.cache_misses == 1
    assert check_graph(str(project))[0]


def test_build_includes_container_configuration_files(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile.dev").write_text(
        "FROM python:3.12\n", encoding="utf-8"
    )
    (tmp_path / "docker-compose.yaml").write_text(
        "services:\n  app:\n    build: .\n", encoding="utf-8"
    )

    result = build_graph(str(tmp_path))
    graph = read_graph(result.context_dir)
    assert graph is not None
    file_nodes = {
        node.path: node for node in graph.nodes if node.kind == "file"
    }

    assert set(file_nodes) == {"Dockerfile.dev", "docker-compose.yaml"}
    assert "FROM python:3.12" in file_nodes["Dockerfile.dev"].body
    assert "services:" in file_nodes["docker-compose.yaml"].body


def test_build_and_check_apply_explicit_source_filters(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "main.py").write_text(
        "def main():\n    return 1\n", encoding="utf-8"
    )
    (source_dir / "skip.py").write_text(
        "def skip():\n    return 0\n", encoding="utf-8"
    )
    (tmp_path / "other.py").write_text(
        "def other():\n    return 2\n", encoding="utf-8"
    )
    include_patterns = ["src/*.py", "other.py"]
    exclude_patterns = ["src/skip.py"]

    result = build_graph(
        str(tmp_path),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    graph = read_graph(result.context_dir)
    assert graph is not None
    file_paths = {node.path for node in graph.nodes if node.kind == "file"}

    assert file_paths == {"src/main.py", "other.py"}
    assert check_graph(
        str(tmp_path),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    ) == (True, "Graph is up to date.")


def test_build_command_accepts_source_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "main.py").write_text(
        "def main():\n    return 1\n", encoding="utf-8"
    )
    (tmp_path / "other.py").write_text(
        "def other():\n    return 2\n", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["build", "--include", "src/*.py"])

    assert result.exit_code == 0
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None
    assert {node.path for node in graph.nodes if node.kind == "file"} == {
        "src/main.py"
    }


@pytest.mark.parametrize(
    ("import_line", "call_name", "expected_name"),
    [
        ("from .helper import greet as say_hi", "say_hi", "greet"),
        ("import { greet as sayHi } from './helper';", "sayHi", "greet"),
    ],
)
def test_resolves_calls_through_named_import_aliases(
    import_line: str, call_name: str, expected_name: str
) -> None:
    extension = ".py" if import_line.startswith("from") else ".ts"
    source = (
        f"{import_line}\n\nfunction run() {{ return {call_name}(); }}"
        if extension == ".ts"
        else (f"def run():\n    return {call_name}()\n\n{import_line}\n")
    )
    importer = f"pkg/main{extension}"
    helper = "pkg/helper.py" if extension == ".py" else "pkg/helper.ts"
    extraction = extract_file(importer, source)
    target = extract_file(
        helper,
        f"export function {expected_name}() {{ return 1; }}"
        if extension == ".ts"
        else (f"def {expected_name}():\n    return 1\n"),
    )
    nodes = extraction.nodes + target.nodes
    edges = resolve_edges(nodes, extraction.raw_edges + target.raw_edges)
    run_node = next(node for node in extraction.nodes if node.name == "run")
    target_node = next(
        node for node in target.nodes if node.name == expected_name
    )

    assert any(
        edge.source == run_node.id
        and edge.target == target_node.id
        and edge.relation == "calls"
        for edge in edges
    )


@pytest.mark.parametrize(
    ("extension", "import_line", "call_expression"),
    [
        (".py", "import helper", "helper.greet"),
        (".py", "import helper as helpers", "helpers.greet"),
        (".py", "import pkg.helper", "pkg.helper.greet"),
        (".ts", "import * as helpers from './helper';", "helpers.greet"),
    ],
)
def test_resolves_calls_through_namespace_imports(
    extension: str, import_line: str, call_expression: str
) -> None:
    source = (
        f"def run():\n    return {call_expression}()\n\n{import_line}\n"
        if extension == ".py"
        else (
            f"{import_line}\n\nfunction run() {{ return {call_expression}(); }}"
        )
    )
    importer = f"pkg/main{extension}"
    helper = f"pkg/helper{extension}"
    extraction = extract_file(importer, source)
    target = extract_file(
        helper,
        "def greet():\n    return 1\n"
        if extension == ".py"
        else "export function greet() { return 1; }",
    )
    nodes = extraction.nodes + target.nodes
    edges = resolve_edges(nodes, extraction.raw_edges + target.raw_edges)
    run_node = next(node for node in extraction.nodes if node.name == "run")
    target_node = next(node for node in target.nodes if node.name == "greet")

    assert any(
        edge.source == run_node.id
        and edge.target == target_node.id
        and edge.relation == "calls"
        for edge in edges
    )


def test_scope_discovery_respects_ignored_directories(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='root'\n", encoding="utf-8"
    )
    nested = tmp_path / "packages" / "core"
    nested.mkdir(parents=True)
    (nested / "go.mod").write_text("module example/core\n", encoding="utf-8")
    ignored = tmp_path / "ignored" / "vendor"
    ignored.mkdir(parents=True)
    (ignored / "pyproject.toml").write_text(
        "[project]\nname='vendor'\n", encoding="utf-8"
    )

    scopes = discover_scopes(str(tmp_path))

    assert [(scope.prefix, scope.label) for scope in scopes] == [
        ("", tmp_path.name),
        ("packages/core/", "core"),
    ]


def test_file_walker_applies_nested_gitignore_rules_and_negations(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text(
        "root_ignored.py\nignored_dir/\n", encoding="utf-8"
    )
    source_dir = tmp_path / "src"
    nested_dir = source_dir / "deep"
    nested_dir.mkdir(parents=True)
    (source_dir / ".gitignore").write_text("*.py\n!keep.py\n", encoding="utf-8")
    (tmp_path / "root_ignored.py").write_text("", encoding="utf-8")
    (source_dir / "skip.py").write_text("", encoding="utf-8")
    (source_dir / "keep.py").write_text("", encoding="utf-8")
    (source_dir / "main.ts").write_text("", encoding="utf-8")
    (nested_dir / "skip.py").write_text("", encoding="utf-8")
    (nested_dir / "keep.py").write_text("", encoding="utf-8")
    ignored_dir = tmp_path / "ignored_dir"
    ignored_dir.mkdir()
    (ignored_dir / "hidden.ts").write_text("", encoding="utf-8")

    paths = walk_dir(str(tmp_path), [".py", ".ts"])

    assert {Path(path).relative_to(tmp_path).as_posix() for path in paths} == {
        "src/deep/keep.py",
        "src/keep.py",
        "src/main.ts",
    }


def test_finds_nearest_bytely_root_from_nested_directory(
    tmp_path: Path,
) -> None:
    outer_root = tmp_path / "outer"
    nested_root = outer_root / "nested"
    start = nested_root / "src"
    (outer_root / "bytely" / ".graph").mkdir(parents=True)
    (outer_root / "bytely" / ".graph" / "wiring.json").write_text(
        "{}", encoding="utf-8"
    )
    (nested_root / "bytely" / ".graph").mkdir(parents=True)
    (nested_root / "bytely" / ".graph" / "wiring.json").write_text(
        "{}", encoding="utf-8"
    )
    start.mkdir()

    assert find_bytely_root(start) == nested_root.resolve()


def test_find_bytely_root_returns_none_when_graph_is_missing(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "src"
    nested.mkdir()

    assert find_bytely_root(nested) is None


def test_check_command_uses_nearest_bytely_root_when_directory_is_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    start = project / "src"
    (project / "bytely" / ".graph").mkdir(parents=True)
    (project / "bytely" / ".graph" / "wiring.json").write_text(
        "{}", encoding="utf-8"
    )
    start.mkdir()
    checked_roots: list[str] = []

    def fake_check_graph(
        root: str, context_dir: str | None, **kwargs: object
    ) -> tuple[bool, str]:
        checked_roots.append(root)
        return True, "Graph is up to date."

    monkeypatch.chdir(start)
    monkeypatch.setattr("bytely.graph.check.check_graph", fake_check_graph)
    result = CliRunner().invoke(main, ["check"])

    assert result.exit_code == 0
    assert checked_roots == [str(project.resolve())]


def test_depth_extensions_only_include_configured_grammars() -> None:
    extensions = depth_extensions()

    assert ".rs" in extensions
    assert ".cs" in extensions
    assert ".r" not in extensions


def test_missing_optional_grammar_skips_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Grammars outside the core languages are optional extras; without one,
    # that language's files are skipped instead of failing the build.
    from bytely.graph import extract

    monkeypatch.setattr(
        extract, "grammar_installed", lambda grammar: grammar != "go"
    )
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", "utf-8")
    (tmp_path / "b.go").write_text("package main\nfunc main() {}\n", "utf-8")

    result = build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))

    assert ".go" not in extract.depth_extensions()
    assert result.files == 1
    assert graph is not None
    assert {node.path for node in graph.nodes} == {"a.py"}
