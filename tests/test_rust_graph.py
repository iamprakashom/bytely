from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from bytely.cli import main
from bytely.graph.build import build_graph
from bytely.graph.extract import extract_file
from bytely.graph.resolve import resolve_edges
from bytely.graph.write import read_graph

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("helper", ["src/helper.rs", "src/helper/mod.rs"])
def test_resolves_use_import_and_call_across_modules(helper: str) -> None:
    importer = "src/main.rs"
    main_result = extract_file(
        importer,
        "mod helper;\n"
        "use crate::helper::greet as say_hi;\n"
        "fn run() { say_hi(); }\n",
    )
    helper_result = extract_file(helper, "pub fn greet() {}\n")
    nodes = main_result.nodes + helper_result.nodes
    raw_edges = main_result.raw_edges + helper_result.raw_edges
    edges = resolve_edges(nodes, raw_edges)
    run_node = next(node for node in main_result.nodes if node.name == "run")
    greet_node = next(
        node for node in helper_result.nodes if node.name == "greet"
    )

    assert any(
        edge.source == importer
        and edge.target == helper
        and edge.relation == "imports"
        for edge in edges
    )
    assert any(
        edge.source == run_node.id
        and edge.target == greet_node.id
        and edge.relation == "calls"
        for edge in edges
    )


def test_resolves_grouped_use_imports_and_calls() -> None:
    importer = "src/main.rs"
    helper = "src/helper.rs"
    main_result = extract_file(
        importer,
        "mod helper;\n"
        "use crate::helper::{greet as say_hi, wave};\n"
        "fn run() { say_hi(); wave(); }\n",
    )
    helper_result = extract_file(
        helper,
        "pub fn greet() {}\npub fn wave() {}\n",
    )
    nodes = main_result.nodes + helper_result.nodes
    raw_edges = main_result.raw_edges + helper_result.raw_edges
    edges = resolve_edges(nodes, raw_edges)
    run_node = next(node for node in main_result.nodes if node.name == "run")
    targets = {
        node.name: node.id
        for node in helper_result.nodes
        if node.name in {"greet", "wave"}
    }

    assert any(
        edge.source == importer
        and edge.target == helper
        and edge.relation == "imports"
        for edge in edges
    )
    assert {
        edge.target
        for edge in edges
        if edge.source == run_node.id and edge.relation == "calls"
    } == set(targets.values())


def test_resolves_nested_use_trees_across_modules() -> None:
    importer = "src/main.rs"
    helper = "src/helper.rs"
    nested = "src/helper/nested.rs"
    main_result = extract_file(
        importer,
        "mod helper;\n"
        "use crate::helper::{greet as say_hi, nested::{wave as send}};\n"
        "fn run() { say_hi(); send(); }\n",
    )
    helper_result = extract_file(helper, "pub fn greet() {}\nmod nested;\n")
    nested_result = extract_file(nested, "pub fn wave() {}\n")
    nodes = main_result.nodes + helper_result.nodes + nested_result.nodes
    raw_edges = (
        main_result.raw_edges
        + helper_result.raw_edges
        + nested_result.raw_edges
    )
    edges = resolve_edges(nodes, raw_edges)
    run_node = next(node for node in main_result.nodes if node.name == "run")
    targets = {
        node.name: node.id
        for node in helper_result.nodes + nested_result.nodes
        if node.name in {"greet", "wave"}
    }

    assert {
        edge.target
        for edge in edges
        if edge.source == importer and edge.relation == "imports"
    } == {helper, nested}
    assert {
        edge.target
        for edge in edges
        if edge.source == run_node.id and edge.relation == "calls"
    } == set(targets.values())


def test_resolves_wildcard_import_calls_across_modules() -> None:
    importer = "src/main.rs"
    helper = "src/helper.rs"
    nested = "src/helper/nested.rs"
    main_result = extract_file(
        importer,
        "mod helper;\n"
        "use crate::helper::*;\n"
        "use crate::helper::nested::*;\n"
        "fn run() { greet(); wave(); }\n",
    )
    helper_result = extract_file(helper, "pub fn greet() {}\nmod nested;\n")
    nested_result = extract_file(nested, "pub fn wave() {}\n")
    nodes = main_result.nodes + helper_result.nodes + nested_result.nodes
    raw_edges = (
        main_result.raw_edges
        + helper_result.raw_edges
        + nested_result.raw_edges
    )
    edges = resolve_edges(nodes, raw_edges)
    run_node = next(node for node in main_result.nodes if node.name == "run")
    targets = {
        node.id
        for node in helper_result.nodes + nested_result.nodes
        if node.name in {"greet", "wave"}
    }

    assert {
        edge.target
        for edge in edges
        if edge.source == importer and edge.relation == "imports"
    } == {helper, nested}
    assert {
        edge.target
        for edge in edges
        if edge.source == run_node.id and edge.relation == "calls"
    } == targets


@pytest.mark.parametrize("import_path", ["crate::*", "super::*"])
def test_resolves_root_and_parent_wildcard_imports(import_path: str) -> None:
    root_path = "src/lib.rs"
    importer = "src/child.rs"
    root_result = extract_file(root_path, "pub fn greet() {}\n")
    importer_result = extract_file(
        importer,
        f"use {import_path}\nfn run() {{ greet(); }}\n",
    )
    nodes = root_result.nodes + importer_result.nodes
    raw_edges = root_result.raw_edges + importer_result.raw_edges
    edges = resolve_edges(nodes, raw_edges)
    run_node = next(
        node for node in importer_result.nodes if node.name == "run"
    )
    greet_node = next(
        node for node in root_result.nodes if node.name == "greet"
    )

    assert any(
        edge.source == importer
        and edge.target == root_path
        and edge.relation == "imports"
        for edge in edges
    )
    assert any(
        edge.source == run_node.id
        and edge.target == greet_node.id
        and edge.relation == "calls"
        for edge in edges
    )


def test_resolves_path_attribute_module_imports() -> None:
    importer = "src/lib.rs"
    helper = "src/support/custom.rs"
    importer_result = extract_file(
        importer,
        '#[path = "support/custom.rs"] pub mod custom;\n'
        "use crate::custom::greet as say_hi;\n"
        "pub fn run() { say_hi(); }\n",
    )
    helper_result = extract_file(helper, "pub fn greet() {}\n")
    nodes = importer_result.nodes + helper_result.nodes
    raw_edges = importer_result.raw_edges + helper_result.raw_edges
    edges = resolve_edges(nodes, raw_edges)
    run_node = next(
        node for node in importer_result.nodes if node.name == "run"
    )
    greet_node = next(
        node for node in helper_result.nodes if node.name == "greet"
    )

    assert any(
        edge.source == importer
        and edge.target == helper
        and edge.relation == "imports"
        for edge in edges
    )
    assert any(
        edge.source == run_node.id
        and edge.target == greet_node.id
        and edge.relation == "calls"
        for edge in edges
    )


def test_extracts_rust_types_and_owned_impl_methods() -> None:
    result = extract_file(
        "src/lib.rs",
        "pub struct Counter<T> { value: T }\n"
        "pub enum State { Ready, Busy }\n"
        "pub trait Reset { fn clear(&mut self); }\n"
        "impl<T> Counter<T> {\n"
        "    fn reset(&mut self) { self.increment(); }\n"
        "    fn increment(&mut self) {}\n"
        "}\n"
        "impl Reset for Counter { fn clear(&mut self) { self.reset(); } }\n",
    )
    nodes = {node.id: node for node in result.nodes}
    edges = resolve_edges(result.nodes, result.raw_edges)

    assert nodes["src/lib.rs#Counter"].kind == "struct"
    assert nodes["src/lib.rs#State"].kind == "enum"
    assert nodes["src/lib.rs#Reset"].kind == "trait"
    assert nodes["src/lib.rs#Reset.clear"].kind == "method"
    assert nodes["src/lib.rs#Reset.clear"].owner == "Reset"
    assert nodes["src/lib.rs#Counter.reset"].kind == "method"
    assert nodes["src/lib.rs#Counter.reset"].owner == "Counter"
    assert nodes["src/lib.rs#Counter.increment"].kind == "method"
    assert nodes["src/lib.rs#Counter.clear"].kind == "method"
    assert nodes["src/lib.rs#Counter.clear"].owner == "Counter"
    assert any(
        edge.source == "src/lib.rs#Counter"
        and edge.target == "src/lib.rs#Counter.reset"
        and edge.relation == "contains"
        for edge in edges
    )
    assert any(
        edge.source == "src/lib.rs#Counter.reset"
        and edge.target == "src/lib.rs#Counter.increment"
        and edge.relation == "calls"
        for edge in edges
    )
    assert any(
        edge.source == "src/lib.rs#Counter.clear"
        and edge.target == "src/lib.rs#Counter.reset"
        and edge.relation == "calls"
        for edge in edges
    )


def test_rust_matches_explicit_golden_for_symbols_and_unexpanded_macros() -> (
    None
):
    path = "src/lib.rs"
    source = (
        "use external_crate::Widget;\n"
        "pub type WidgetId = u64;\n"
        "pub const MAX_WIDGETS: usize = 8;\n"
        "pub static LIVE_WIDGETS: usize = 0;\n"
        "pub struct Pair<T>(pub T, pub T);\n"
        "pub enum State { Ready, Failed { code: u16 } }\n"
        "pub trait Render { fn render(&self); }\n"
        "impl<T> Pair<T> { fn first(&self) -> &T { &self.0 } }\n"
        "impl Render for State { fn render(&self) {} }\n"
        "macro_rules! make_generated { () => { fn generated() {} }; }\n"
        "pub fn invoke_macros() { make_generated!(); external_macro!(); }\n"
    )
    result = extract_file(path, source)
    nodes = {node.id: node for node in result.nodes}
    edges = resolve_edges(result.nodes, result.raw_edges)

    expected = {
        (path, "file", None, "L1-L12"),
        (f"{path}#WidgetId", "type", None, "L2-L2"),
        (f"{path}#MAX_WIDGETS", "constant", None, "L3-L3"),
        (f"{path}#LIVE_WIDGETS", "variable", None, "L4-L4"),
        (f"{path}#Pair", "struct", None, "L5-L5"),
        (f"{path}#State", "enum", None, "L6-L6"),
        (f"{path}#Render", "trait", None, "L7-L7"),
        (f"{path}#Render.render", "method", "Render", "L7-L7"),
        (f"{path}#Pair.first", "method", "Pair", "L8-L8"),
        (f"{path}#State.render", "method", "State", "L9-L9"),
        (f"{path}#invoke_macros", "function", None, "L11-L11"),
    }
    actual = {
        (node.id, node.kind, node.owner, node.span) for node in result.nodes
    }

    assert actual == expected
    assert len(nodes) == len(result.nodes)
    assert "generated" not in {node.name for node in result.nodes}
    assert "external_crate" in nodes[path].body
    assert "macro_rules! make_generated" in nodes[path].body
    # The external crate never resolves, and Rust keeps no raw-specifier edge
    # for it: the reference implementation's generic Rust tier emits no `use`
    # edges (P4).
    assert not any(
        edge.relation == "imports" and edge.source == path for edge in edges
    )
    invoke_node = nodes[f"{path}#invoke_macros"]
    assert not any(
        edge.relation == "calls"
        and edge.source == invoke_node.id
        and edge.name in {"make_generated", "external_macro"}
        for edge in edges
    )


def test_cfg_alternative_functions_receive_stable_unique_ids() -> None:
    result = extract_file(
        "src/lib.rs",
        '#[cfg(feature = "alpha")]\n'
        "pub fn decode() {}\n"
        '#[cfg(feature = "beta")]\n'
        "pub fn decode() {}\n",
    )
    decode_nodes = [node for node in result.nodes if node.name == "decode"]

    assert [node.id for node in decode_nodes] == [
        "src/lib.rs#decode",
        "src/lib.rs#decode~2",
    ]


def test_cfg_alternative_functions_with_complex_predicates_get_unique_ids() -> (
    None
):
    """`any`/`all`/`not` cfg predicates are indexed regardless of the
    active feature set; alternative definitions still get stable IDs."""
    result = extract_file(
        "src/lib.rs",
        '#[cfg(all(unix, not(target_os = "macos")))]\n'
        "pub fn platform() {}\n"
        '#[cfg(any(windows, target_os = "macos"))]\n'
        "pub fn platform() {}\n"
        "#[cfg(not(any(unix, windows)))]\n"
        "pub fn platform() {}\n",
    )
    platform_nodes = [node for node in result.nodes if node.name == "platform"]

    assert [node.id for node in platform_nodes] == [
        "src/lib.rs#platform",
        "src/lib.rs#platform~2",
        "src/lib.rs#platform~3",
    ]


def test_cfg_gated_impl_and_module_are_indexed_regardless_of_predicate() -> (
    None
):
    """cfg-gated impl blocks and modules are indexed unconditionally;
    Bytely does not evaluate cfg predicates."""
    importer = "src/lib.rs"
    helper = "src/unix_support.rs"
    importer_result = extract_file(
        importer,
        "pub struct Widget;\n"
        "#[cfg(unix)]\n"
        "impl Widget { pub fn describe(&self) {} }\n"
        "#[cfg(unix)]\n"
        "mod unix_support;\n"
        "use crate::unix_support::helper;\n"
        "pub fn run() { helper(); }\n",
    )
    helper_result = extract_file(helper, "pub fn helper() {}\n")
    nodes = importer_result.nodes + helper_result.nodes
    raw_edges = importer_result.raw_edges + helper_result.raw_edges
    edges = resolve_edges(nodes, raw_edges)
    node_ids = {node.id for node in importer_result.nodes}

    assert f"{importer}#Widget.describe" in node_ids
    assert any(
        node.id == f"{importer}#Widget.describe" and node.owner == "Widget"
        for node in importer_result.nodes
    )
    assert any(
        edge.source == importer and edge.target == helper
        for edge in edges
        if edge.relation == "imports"
    )


def test_path_attribute_is_found_alongside_other_attributes() -> None:
    """`#[path]` resolution tolerates preceding/interleaved attributes
    such as cfg and doc comments on the same module declaration."""
    importer = "src/lib.rs"
    helper = "src/support/custom.rs"
    importer_result = extract_file(
        importer,
        "/// doc comment\n"
        "#[allow(dead_code)]\n"
        '#[path = "support/custom.rs"]\n'
        "pub mod custom;\n"
        "use crate::custom::greet;\n"
        "pub fn run() { greet(); }\n",
    )
    helper_result = extract_file(helper, "pub fn greet() {}\n")
    nodes = importer_result.nodes + helper_result.nodes
    raw_edges = importer_result.raw_edges + helper_result.raw_edges
    edges = resolve_edges(nodes, raw_edges)

    assert any(
        edge.source == importer
        and edge.target == helper
        and edge.relation == "imports"
        for edge in edges
    )


def test_cfg_attr_path_is_not_resolved_as_an_active_override() -> None:
    """`#[cfg_attr(feature, path = "...")]` is conditional on an unknown
    Cargo feature set, so Bytely must not treat it as an active
    override; conventional module resolution should not apply either
    once a path-style attribute is present, since the file is legitimately
    ambiguous without evaluating the feature."""
    result = extract_file(
        "src/lib.rs",
        '#[cfg_attr(feature = "custom", path = "support/custom.rs")]\n'
        "mod maybe_custom;\n",
    )

    module_edges = [
        edge
        for edge in result.raw_edges
        if edge.rust_module_name == "maybe_custom"
    ]
    assert module_edges == []


def test_trait_default_methods_are_methods_that_resolve_self_calls() -> None:
    # Mirrors serde's Visitor: default bodies delegate via `self`.
    source = (
        "pub trait Visitor {\n"
        "    fn visit_i64(&self, v: i64) -> bool;\n"
        "    fn visit_i8(&self, v: i8) -> bool {\n"
        "        self.visit_i64(v as i64)\n"
        "    }\n"
        "}\n"
    )
    result = extract_file("src/lib.rs", source)
    by_id = {node.id: (node.kind, node.owner) for node in result.nodes}
    calls = {
        (edge.source, edge.target)
        for edge in resolve_edges(result.nodes, result.raw_edges)
        if edge.relation == "calls"
    }

    assert by_id["src/lib.rs#Visitor.visit_i8"] == ("method", "Visitor")
    assert by_id["src/lib.rs#Visitor.visit_i64"] == ("method", "Visitor")
    assert (
        "src/lib.rs#Visitor.visit_i8",
        "src/lib.rs#Visitor.visit_i64",
    ) in calls


def test_rust_spans_use_utf8_offsets_with_crlf_lines() -> None:
    result = extract_file(
        "src/lib.rs",
        "//! Unicode: λ\r\nmod before { pub fn run() {} }\r\nmod after;\r\n",
    )
    nodes = {node.id: node for node in result.nodes}

    # As in the reference implementation, the file span counts newline-separated
    # segments, so the trailing newline adds a line.
    assert nodes["src/lib.rs"].span == "L1-L4"
    assert nodes["src/lib.rs#before"].span == "L2-L2"
    assert nodes["src/lib.rs#before.run"].span == "L2-L2"
    assert nodes["src/lib.rs#after"].span == "L3-L3"


def test_rust_project_build_check_cache_and_staleness(tmp_path: Path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (tmp_path / "Cargo.toml").write_text(
        "[package]\nname = 'sample'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )
    (source_dir / "lib.rs").write_text(
        "pub mod helpers;\n"
        "use crate::helpers::greet;\n"
        "pub const MAX_ITEMS: usize = 8;\n"
        "pub static LIVE_ITEMS: usize = 0;\n"
        "pub fn run() { greet(); }\n",
        encoding="utf-8",
    )
    helper_path = source_dir / "helpers.rs"
    helper_path.write_text("pub fn greet() {}\n", encoding="utf-8")
    runner = CliRunner()

    build_result = runner.invoke(main, ["build", str(tmp_path)])
    assert build_result.exit_code == 0, build_result.output
    graph_dir = tmp_path / "bytely"
    graph = read_graph(str(graph_dir))
    assert graph is not None
    node_ids = [node.id for node in graph.nodes]
    assert len(node_ids) == len(set(node_ids))
    run_node = next(node for node in graph.nodes if node.id == "src/lib.rs#run")
    greet_node = next(
        node for node in graph.nodes if node.id == "src/helpers.rs#greet"
    )
    assert any(
        edge.source == "src/lib.rs"
        and edge.target == "src/helpers.rs"
        and edge.relation == "imports"
        for edge in graph.edges
    )
    assert any(
        edge.source == run_node.id
        and edge.target == greet_node.id
        and edge.relation == "calls"
        for edge in graph.edges
    )

    graph_bytes = (graph_dir / ".graph" / "wiring.json").read_bytes()
    warm_build = build_graph(str(tmp_path))
    assert warm_build.cache_hits == 2
    assert (graph_dir / ".graph" / "wiring.json").read_bytes() == graph_bytes
    assert runner.invoke(main, ["check", str(tmp_path)]).exit_code == 0

    helper_path.write_text(
        "pub fn greet() {}\npub fn wave() {}\n", encoding="utf-8"
    )
    stale_check = runner.invoke(main, ["check", str(tmp_path)])
    assert stale_check.exit_code == 1
    assert runner.invoke(main, ["build", str(tmp_path)]).exit_code == 0
    assert runner.invoke(main, ["check", str(tmp_path)]).exit_code == 0


@pytest.mark.parametrize(
    ("importer", "crate_root", "module_path", "helper"),
    [
        ("src/main.rs", None, "self::helper", "src/helper.rs"),
        ("src/sub.rs", "src/lib.rs", "super::helper", "src/helper.rs"),
        (
            "src/sub/nested.rs",
            "src/lib.rs",
            "super::helper",
            "src/sub/helper.rs",
        ),
        (
            "src/sub/nested/deep.rs",
            "src/lib.rs",
            "super::super::helper",
            "src/sub/helper.rs",
        ),
    ],
)
def test_resolves_self_and_super_module_imports(
    importer: str,
    crate_root: str | None,
    module_path: str,
    helper: str,
) -> None:
    root_nodes = (
        extract_file(crate_root, "mod sub;\n").nodes if crate_root else []
    )
    main_result = extract_file(
        importer,
        f"use {module_path}::greet as say_hi;\nfn run() {{ say_hi(); }}\n",
    )
    helper_result = extract_file(helper, "pub fn greet() {}\n")
    nodes = root_nodes + main_result.nodes + helper_result.nodes
    raw_edges = main_result.raw_edges + helper_result.raw_edges
    edges = resolve_edges(nodes, raw_edges)
    run_node = next(node for node in main_result.nodes if node.name == "run")
    greet_node = next(
        node for node in helper_result.nodes if node.name == "greet"
    )

    assert any(
        edge.source == importer
        and edge.target == helper
        and edge.relation == "imports"
        for edge in edges
    )
    assert any(
        edge.source == run_node.id
        and edge.target == greet_node.id
        and edge.relation == "calls"
        for edge in edges
    )


def test_rust_type_qualified_calls_resolve_to_the_types_method(
    tmp_path: Path,
) -> None:
    # `Cache::new()` and `Self::new()` name the type, so they bind to its
    # associated function; `Other::new()` and `String::from()` name types
    # with no such method here and get no edge (P13).
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text(
        "pub struct Cache;\n"
        "impl Cache {\n"
        "    pub fn new() -> Self { Cache }\n"
        "    pub fn fresh() -> Self { Self::new() }\n"
        "}\n"
        "pub struct Other;\n"
        "impl Other { pub fn make() -> Self { Other } }\n"
        "pub fn run() {\n"
        "    let _ = Cache::new();\n"
        "    let _ = Other::new();\n"
        '    let _ = String::from("x");\n'
        "}\n",
        encoding="utf-8",
    )
    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None
    calls = {
        (edge.source, edge.target, edge.confidence)
        for edge in graph.edges
        if edge.relation == "calls"
    }
    assert calls == {
        ("src/lib.rs#run", "src/lib.rs#Cache.new", "extracted"),
        ("src/lib.rs#Cache.fresh", "src/lib.rs#Cache.new", "extracted"),
    }


def test_bare_name_guesses_stay_inside_the_callers_crate(
    tmp_path: Path,
) -> None:
    # Mirrors serde (P12): an integration-test crate declares `struct Ok;`
    # to test macro hygiene, and `Ok(..)` everywhere else was guessed onto
    # it. Another crate's items are only visible through `use` or a path.
    files = {
        "core/src/lib.rs": (
            "mod value;\n"
            "pub fn run() -> Result<u8, ()> {\n"
            "    helper();\n"
            "    Ok(value::Wrapper::new())\n"
            "}\n"
        ),
        "core/src/value.rs": (
            "pub fn helper() {}\n"
            "pub struct Wrapper;\n"
            "impl Wrapper {\n"
            "    pub fn new() -> u8 { 1 }\n"
            "}\n"
        ),
        "core/tests/hygiene.rs": (
            "struct Ok;\n"
            "fn helper() {}\n"
            "fn check() -> u8 {\n"
            "    Wrapper::new()\n"
            "}\n"
        ),
    }
    for relative, text in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None
    calls = {
        (edge.source, edge.target)
        for edge in graph.edges
        if edge.relation == "calls"
    }

    # The test crate's `Ok` and `helper` are invisible to the library.
    assert ("core/src/lib.rs#run", "core/tests/hygiene.rs#Ok") not in calls
    assert (
        "core/src/lib.rs#run",
        "core/tests/hygiene.rs#helper",
    ) not in calls
    # Within one crate a unique cross-file name still resolves: the
    # test crate's `helper` no longer makes the library's call ambiguous.
    assert ("core/src/lib.rs#run", "core/src/value.rs#helper") in calls
    # An explicit type name may cross crates (`use` / re-exports).
    assert (
        "core/tests/hygiene.rs#check",
        "core/src/value.rs#Wrapper.new",
    ) in calls


def _build_rust_tree(root: Path, files: dict[str, str]) -> set[tuple[str, str]]:
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    build_graph(str(root))
    graph = read_graph(str(root / "bytely"))
    assert graph is not None
    return {
        (edge.source, edge.target)
        for edge in graph.edges
        if edge.relation == "calls"
    }


def test_flat_crate_layout_without_src_stays_one_crate(tmp_path: Path) -> None:
    # `[lib] path = "lib.rs"`: no `src/`, yet lib.rs and util.rs are one
    # crate, so a unique bare name between them still resolves.
    calls = _build_rust_tree(
        tmp_path,
        {
            "crates/foo/lib.rs": "mod util;\npub fn run() { helper(); }\n",
            "crates/foo/util.rs": "pub fn helper() {}\n",
        },
    )

    assert ("crates/foo/lib.rs#run", "crates/foo/util.rs#helper") in calls


def test_shared_test_module_joins_each_test_crate_that_declares_it(
    tmp_path: Path,
) -> None:
    # `tests/common/mod.rs` is compiled into every test crate declaring
    # `mod common;`, and into no other.
    calls = _build_rust_tree(
        tmp_path,
        {
            "pkg/src/lib.rs": "pub fn lib_fn() {}\n",
            "pkg/tests/common/mod.rs": "pub fn setup() {}\n",
            "pkg/tests/uses_common.rs": (
                "mod common;\nuse common::*;\nfn t() { setup(); }\n"
            ),
            "pkg/tests/no_common.rs": "fn t() { setup(); }\n",
        },
    )

    assert (
        "pkg/tests/uses_common.rs#t",
        "pkg/tests/common/mod.rs#setup",
    ) in calls
    assert (
        "pkg/tests/no_common.rs#t",
        "pkg/tests/common/mod.rs#setup",
    ) not in calls


def test_rust_exported_follows_pub_and_trait_impls() -> None:
    # P10 review: `impl Trait for Type` items take no `pub` yet are as
    # visible as the trait; only a bare `pub` exports (`pub(crate)`,
    # `pub(super)`, `pub(self)`, and `pub(in ...)` stay inside the crate).
    source = (
        "pub trait T { fn a(&self); }\n"
        "pub struct X;\n"
        "impl T for X {\n"
        "    fn a(&self) {}\n"
        "    const C: u32 = 1;\n"
        "}\n"
        "impl X {\n"
        "    pub fn p(&self) {}\n"
        "    pub(crate) fn pc(&self) {}\n"
        "    pub(self) fn ps(&self) {}\n"
        "    fn q(&self) {}\n"
        "}\n"
    )
    exported = {
        node.id: node.exported
        for node in extract_file("src/lib.rs", source).nodes
    }

    assert exported["src/lib.rs#T.a"] is True
    assert exported["src/lib.rs#X.a"] is True
    assert exported["src/lib.rs#X.C"] is True
    assert exported["src/lib.rs#X.p"] is True
    assert exported["src/lib.rs#X.pc"] is False
    assert exported["src/lib.rs#X.ps"] is False
    assert exported["src/lib.rs#X.q"] is False


def test_rust_inline_module_reexport_is_not_an_external_binding() -> None:
    # serde's `pub use self::content::{content_as_str}` re-exports an inline
    # module: unresolvable to a file, yet internal, so it must not block
    # the same-file call it names.
    source = (
        "mod content {\n"
        "    pub fn content_as_str() {}\n"
        "}\n"
        "pub use self::content::content_as_str;\n"
        "fn take() { content_as_str(); }\n"
    )
    result = extract_file("src/de.rs", source)
    calls = {
        (edge.source, edge.target)
        for edge in resolve_edges(result.nodes, result.raw_edges)
        if edge.relation == "calls"
    }

    assert ("src/de.rs#take", "src/de.rs#content.content_as_str") in calls
