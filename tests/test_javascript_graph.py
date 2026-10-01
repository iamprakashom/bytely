from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from bytely.cli import main
from bytely.graph.build import build_graph
from bytely.graph.extract import extract_file
from bytely.graph.resolve import resolve_edges
from bytely.graph.write import read_graph

if TYPE_CHECKING:
    from pathlib import Path

    from bytely.graph.types import GraphV1


def _write_files(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _build(root: Path, files: dict[str, str]) -> GraphV1:
    _write_files(root, files)
    build_graph(str(root))
    graph = read_graph(str(root / "bytely"))
    assert graph is not None
    return graph


def _edges(graph: GraphV1, relation: str) -> set[tuple[str, str]]:
    return {
        (edge.source, edge.target)
        for edge in graph.edges
        if edge.relation == relation
    }


def test_javascript_matches_explicit_golden_for_symbols_and_calls() -> None:
    path = "src/shapes.js"
    source = (
        "import { format } from 'external-package';\n"
        "\n"
        "export const LIMIT = 8;\n"
        "let counter = 0;\n"
        "\n"
        "export class Shape {\n"
        "  constructor(size) { this.size = size; }\n"
        "  area() { return this.scale(this.size); }\n"
        "  scale(n) { return n * 2; }\n"
        "  static unit() { return new Shape(1); }\n"
        "}\n"
        "\n"
        "export const describe = (shape) => format(shape.area());\n"
        "const legacy = function () { return describe(Shape.unit()); };\n"
        "function* ids() { yield counter++; }\n"
        "export default function main() {\n"
        "  const local = () => legacy();\n"
        "  return local();\n"
        "}\n"
    )
    result = extract_file(path, source)
    edges = resolve_edges(result.nodes, result.raw_edges)

    expected = {
        (path, "file", None, "L1-L20"),
        (f"{path}#LIMIT", "constant", None, "L3-L3"),
        (f"{path}#counter", "variable", None, "L4-L4"),
        (f"{path}#Shape", "class", None, "L6-L11"),
        (f"{path}#Shape.constructor", "method", "Shape", "L7-L7"),
        (f"{path}#Shape.area", "method", "Shape", "L8-L8"),
        (f"{path}#Shape.scale", "method", "Shape", "L9-L9"),
        (f"{path}#Shape.unit", "method", "Shape", "L10-L10"),
        (f"{path}#describe", "function", None, "L13-L13"),
        (f"{path}#legacy", "function", None, "L14-L14"),
        (f"{path}#ids", "function", None, "L15-L15"),
        (f"{path}#main", "function", None, "L16-L19"),
        (f"{path}#main.local", "function", None, "L17-L17"),
    }
    actual = {
        (node.id, node.kind, node.owner, node.span) for node in result.nodes
    }
    assert actual == expected

    call_edges = {
        (edge.source, edge.target) for edge in edges if edge.relation == "calls"
    }
    # `format` comes from an external package and `shape.area()` has an untyped
    # receiver, so neither resolves. `new Shape(1)` is a `new_expression`, which
    # (as in the reference implementation) is not a call site, and the static
    # `Shape.unit()` is not bound by receiver name.
    assert call_edges == {
        (f"{path}#Shape.area", f"{path}#Shape.scale"),
        (f"{path}#legacy", f"{path}#describe"),
        (f"{path}#main", f"{path}#main.local"),
        (f"{path}#main.local", f"{path}#legacy"),
    }
    contains_edges = {
        (edge.source, edge.target)
        for edge in edges
        if edge.relation == "contains"
    }
    assert (f"{path}#main", f"{path}#main.local") in contains_edges
    assert (f"{path}#Shape", f"{path}#Shape.unit") in contains_edges


def test_javascript_ids_and_hashes_are_stable_across_extractions() -> None:
    source = (
        "export function run() { return helper(); }\n"
        "function helper() { return 1; }\n"
        "function helper() { return 2; }\n"
    )
    first = extract_file("src/dup.mjs", source)
    second = extract_file("src/dup.mjs", source)

    assert [(n.id, n.body_hash) for n in first.nodes] == [
        (n.id, n.body_hash) for n in second.nodes
    ]
    helper_ids = [n.id for n in first.nodes if n.name == "helper"]
    assert helper_ids == ["src/dup.mjs#helper", "src/dup.mjs#helper~2"]
    bodies = {n.id: n.body for n in first.nodes}
    assert (
        bodies["src/dup.mjs#run"]
        == ("export function run() { return helper(); }")[len("export ") :]
    )


def test_javascript_named_aliased_and_namespace_imports_resolve(
    tmp_path: Path,
) -> None:
    graph = _build(
        tmp_path,
        {
            "src/math.js": (
                "export function add(a, b) { return a + b; }\n"
                "export const mul = (a, b) => a * b;\n"
            ),
            "src/main.mjs": (
                "import { add, mul as times } from './math.js';\n"
                "import * as math from './math';\n"
                "export function run() { add(1, 2); times(3, 4); }\n"
                "export function viaNamespace() { return math.mul(5, 6); }\n"
            ),
        },
    )

    assert ("src/main.mjs", "src/math.js") in _edges(graph, "imports")
    assert _edges(graph, "calls") >= {
        ("src/main.mjs#run", "src/math.js#add"),
        ("src/main.mjs#run", "src/math.js#mul"),
        ("src/main.mjs#viaNamespace", "src/math.js#mul"),
    }


def test_commonjs_require_bindings_resolve_across_files(
    tmp_path: Path,
) -> None:
    graph = _build(
        tmp_path,
        {
            "lib/math.cjs": (
                "function add(a, b) { return a + b; }\n"
                "const mul = function (a, b) { return a * b; };\n"
                "function sub(a, b) { return a - b; }\n"
                "module.exports = { add, mul, sub };\n"
            ),
            "lib/app.cjs": (
                "const { add, mul: product } = require('./math.cjs');\n"
                "const math = require('./math');\n"
                "const fs = require('fs');\n"
                "function total() { return add(1, 2) + product(3, 4); }\n"
                "function diff() { return math.sub(5, 1); }\n"
                "function read() { return fs.readFileSync('x'); }\n"
            ),
        },
    )

    assert ("lib/app.cjs", "lib/math.cjs") in _edges(graph, "imports")
    calls = _edges(graph, "calls")
    assert calls >= {
        ("lib/app.cjs#total", "lib/math.cjs#add"),
        ("lib/app.cjs#total", "lib/math.cjs#mul"),
        ("lib/app.cjs#diff", "lib/math.cjs#sub"),
    }
    # `fs` is a Node builtin with no local file: no edge is synthesized.
    assert not any(source == "lib/app.cjs#read" for source, _ in calls)


def test_jsx_components_resolve_calls_inside_handlers(tmp_path: Path) -> None:
    graph = _build(
        tmp_path,
        {
            "src/actions.js": "export function save() { return true; }\n",
            "src/Button.jsx": (
                "import { save } from './actions';\n"
                "export class Button {\n"
                "  render() { return <b onClick={() => save()}>ok</b>; }\n"
                "}\n"
                "export const Panel = () => <div onClick={() => save()} />;\n"
            ),
        },
    )
    kinds = {node.id: node.kind for node in graph.nodes}

    assert kinds["src/Button.jsx#Button"] == "class"
    assert kinds["src/Button.jsx#Button.render"] == "method"
    assert kinds["src/Button.jsx#Panel"] == "function"
    assert ("src/Button.jsx", "src/actions.js") in _edges(graph, "imports")
    assert _edges(graph, "calls") >= {
        ("src/Button.jsx#Button.render", "src/actions.js#save"),
        ("src/Button.jsx#Panel", "src/actions.js#save"),
    }


def test_default_imports_stay_unbound_like_the_reference(
    tmp_path: Path,
) -> None:
    # The reference implementation deliberately does not bind default imports
    # to a symbol; only the file import
    # resolves. The call then falls back to the reference implementation's
    # unique-name match, which is labelled `inferred`, never `extracted`.
    graph = _build(
        tmp_path,
        {
            "src/widget.js": "export default function build() { return 1; }\n",
            "src/other.js": "export function make() { return 3; }\n",
            "src/use.js": (
                "import make from './widget.js';\n"
                "export function go() { return make(); }\n"
            ),
        },
    )

    assert ("src/use.js", "src/widget.js") in _edges(graph, "imports")
    assert ("src/use.js#go", "src/widget.js#build") not in _edges(
        graph, "calls"
    )
    fallback = [
        edge
        for edge in graph.edges
        if edge.relation == "calls" and edge.source == "src/use.js#go"
    ]
    assert [(edge.target, edge.confidence) for edge in fallback] == [
        ("src/other.js#make", "inferred")
    ]


def test_bare_name_fallback_does_not_cross_language_families(
    tmp_path: Path,
) -> None:
    graph = _build(
        tmp_path,
        {
            "web/format.js": "export function render() { return 1; }\n",
            "web/types.ts": "export function parse() { return 2; }\n",
            "app/main.py": "def run():\n    return render()\n",
            "web/main.js": "export function go() { return parse(); }\n",
        },
    )
    calls = _edges(graph, "calls")

    # A Python call never resolves into JavaScript by name alone...
    assert not any(source == "app/main.py#run" for source, _ in calls)
    # ...but JavaScript and TypeScript share a family, as in the reference
    # implementation.
    assert ("web/main.js#go", "web/types.ts#parse") in calls


def test_nested_functions_bind_only_within_their_enclosing_scope(
    tmp_path: Path,
) -> None:
    graph = _build(
        tmp_path,
        {
            "src/m.js": (
                "function a() {\n"
                "  const next = () => 1;\n"
                "  function run() { return next(); }\n"
                "  return next() + run();\n"
                "}\n"
                "function b(req, next) { return next(); }\n"
            ),
            "src/other.js": (
                "import { next } from './m.js';\n"
                "export function c() { return next(); }\n"
            ),
        },
    )
    callers_of_next = {
        source
        for source, target in _edges(graph, "calls")
        if target == "src/m.js#a.next"
    }

    # Only `a` and its nested `run` can see the closure. `b`'s `next`
    # parameter and `other.js`'s import must not bind to it.
    assert callers_of_next == {"src/m.js#a", "src/m.js#a.run"}


def test_require_of_a_directory_resolves_its_javascript_index(
    tmp_path: Path,
) -> None:
    graph = _build(
        tmp_path,
        {
            "src/lib/index.js": "function foo() { return 1; }\n"
            "module.exports = { foo };\n",
            "src/elsewhere.js": "export function foo() { return 2; }\n",
            "src/esm/index.mjs": "export function bar() { return 3; }\n",
            "src/app.js": (
                "const { foo } = require('./lib');\n"
                "import { bar } from './esm';\n"
                "function run() { return foo() + bar(); }\n"
            ),
        },
    )

    assert _edges(graph, "imports") >= {
        ("src/app.js", "src/lib/index.js"),
        ("src/app.js", "src/esm/index.mjs"),
    }
    # A second `foo` exists, so this edge can only come from the binding.
    assert _edges(graph, "calls") >= {
        ("src/app.js#run", "src/lib/index.js#foo"),
        ("src/app.js#run", "src/esm/index.mjs#bar"),
    }


def test_parent_relative_requires_resolve_and_externals_keep_names(
    tmp_path: Path,
) -> None:
    # Express's examples use `require('../../')` and its tests
    # `require('..')` for the package root's index.js: `..` segments must
    # be normalized. Unresolvable specifiers keep raw-name edges (P4).
    graph = _build(
        tmp_path,
        {
            "index.js": "module.exports = function createApp() {};\n",
            "lib/router.js": (
                "function route() {}\nmodule.exports = { route };\n"
            ),
            "examples/auth/index.js": (
                "const express = require('../../');\n"
                "const { route } = require('../../lib/router');\n"
                "const escape = require('../../../outside');\n"
                "const fs = require('fs');\n"
                "function start() { route(); }\n"
            ),
            "test/app.js": "const express = require('..');\n",
        },
    )
    imports = _edges(graph, "imports")

    assert imports >= {
        ("examples/auth/index.js", "index.js"),
        ("examples/auth/index.js", "lib/router.js"),
        ("test/app.js", "index.js"),
        # Outside the tree, and a builtin: raw specifiers, never a file.
        ("examples/auth/index.js", "../../../outside"),
        ("examples/auth/index.js", "fs"),
    }
    assert ("examples/auth/index.js#start", "lib/router.js#route") in _edges(
        graph, "calls"
    )


def test_javascript_project_build_check_cache_and_staleness(
    tmp_path: Path,
) -> None:
    _write_files(
        tmp_path,
        {
            "src/util.js": "export function greet() { return 1; }\n",
            "src/esm.mjs": (
                "import { greet } from './util.js';\n"
                "export const run = () => greet();\n"
            ),
            "src/cjs.cjs": (
                "const { greet } = require('./util.js');\n"
                "function start() { return greet(); }\n"
                "module.exports = { start };\n"
            ),
            "src/View.jsx": (
                "import { run } from './esm.mjs';\n"
                "export function View() { return <p>{run()}</p>; }\n"
            ),
        },
    )
    runner = CliRunner()

    build_result = runner.invoke(main, ["build", str(tmp_path)])
    assert build_result.exit_code == 0, build_result.output
    graph_dir = tmp_path / "bytely"
    graph = read_graph(str(graph_dir))
    assert graph is not None
    node_ids = [node.id for node in graph.nodes]
    assert len(node_ids) == len(set(node_ids))
    assert {node.path for node in graph.nodes if node.kind == "file"} == {
        "src/util.js",
        "src/esm.mjs",
        "src/cjs.cjs",
        "src/View.jsx",
    }
    assert _edges(graph, "calls") >= {
        ("src/esm.mjs#run", "src/util.js#greet"),
        ("src/cjs.cjs#start", "src/util.js#greet"),
        ("src/View.jsx#View", "src/esm.mjs#run"),
    }

    graph_bytes = (graph_dir / ".graph" / "wiring.json").read_bytes()
    warm_build = build_graph(str(tmp_path))
    assert warm_build.cache_hits == 4
    assert warm_build.cache_misses == 0
    assert (graph_dir / ".graph" / "wiring.json").read_bytes() == graph_bytes
    assert runner.invoke(main, ["check", str(tmp_path)]).exit_code == 0

    util_path = tmp_path / "src" / "util.js"
    util_path.write_text(
        "export function greet() { return 1; }\n"
        "export function wave() { return 2; }\n",
        encoding="utf-8",
    )
    assert runner.invoke(main, ["check", str(tmp_path)]).exit_code == 1
    changed_build = build_graph(str(tmp_path))
    assert changed_build.cache_misses == 1
    assert runner.invoke(main, ["check", str(tmp_path)]).exit_code == 0

    # Reverting the edit reproduces the original graph byte-for-byte.
    util_path.write_text(
        "export function greet() { return 1; }\n", encoding="utf-8"
    )
    build_graph(str(tmp_path))
    assert (graph_dir / ".graph" / "wiring.json").read_bytes() == graph_bytes


def test_jsx_in_plain_js_file_is_extracted(tmp_path: Path) -> None:
    # The TypeScript grammar reads `<div>` in a `.js` file as a type
    # assertion and loses `Page`, mis-scopes `go`, and drops every call.
    graph = _build(
        tmp_path,
        {
            "App.js": (
                "export function Button({ label }) {\n"
                "  return <button onClick={() => go()}>{label}</button>;\n"
                "}\n"
                "export function Page() {\n"
                '  return <div><Button label="x" /></div>;\n'
                "}\n"
                "function go() { return 1 }\n"
            )
        },
    )
    spans = {node.id: node.span for node in graph.nodes}
    assert spans["App.js#Button"] == "L1-L3"
    assert spans["App.js#Page"] == "L4-L6"
    assert spans["App.js#go"] == "L7-L7"
    assert ("App.js#Button", "App.js#go") in _edges(graph, "calls")


def test_valid_javascript_the_typescript_grammar_misreads() -> None:
    # `interface` is an ordinary identifier in JavaScript, and
    # `f(a < b, c > (d))` is one call with two comparisons, not a generic
    # call `a<b, c>(d)` inside `f(...)`.
    result = extract_file(
        "m.js",
        "function f(x, y) { return x }\n"
        "function a() {}\n"
        "function run(b, c, d) {\n"
        "  const interface = 1;\n"
        "  return f(a < b, c > (d));\n"
        "}\n",
    )
    calls = {
        edge.name
        for edge in result.raw_edges
        if edge.relation == "calls" and edge.source == "m.js#run"
    }
    assert calls == {"f"}
    assert {node.id for node in result.nodes} >= {"m.js#run", "m.js#f"}


def test_flow_annotated_javascript_falls_back_to_tsx() -> None:
    # tree-sitter-javascript rejects Flow type annotations; TSX reads them.
    result = extract_file(
        "flow.js",
        "// @flow\n"
        "import type { A } from './a';\n"
        "export function f(x: number): string { return g(x); }\n"
        "function g(x: number): string { return String(x); }\n",
    )
    spans = {node.id: node.span for node in result.nodes}
    assert spans["flow.js#f"] == "L3-L3"
    assert spans["flow.js#g"] == "L4-L4"


def test_path_aliases_and_workspace_packages_are_not_external(
    tmp_path: Path,
) -> None:
    # Review of the external-import guard: `@/…` aliases and workspace
    # packages are internal, only unresolvable without tsconfig or
    # package.json, so their calls may still be guessed by unique name.
    # `react` and `node:fs` are genuinely external.
    graph = _build(
        tmp_path,
        {
            "src/utils/date.js": "export function formatDate() {}\n",
            "packages/shared/index.js": "export function share() {}\n",
            "vendor/react-shim.js": "export function useState() {}\n",
            "src/app.js": (
                "import { formatDate } from '@/utils/date';\n"
                "import { share } from '@myorg/shared';\n"
                "import { useState } from 'react';\n"
                "export function run() {\n"
                "  formatDate();\n"
                "  share();\n"
                "  useState();\n"
                "}\n"
            ),
        },
    )
    calls = _edges(graph, "calls")

    assert calls >= {
        ("src/app.js#run", "src/utils/date.js#formatDate"),
        ("src/app.js#run", "packages/shared/index.js#share"),
    }
    # `react` is external: its `useState` is not the shim's.
    assert ("src/app.js#run", "vendor/react-shim.js#useState") not in calls


def test_parse_error_recovery_does_not_mint_keyword_named_definitions() -> None:
    # Reduced from React's Flow-typed ReactFiberConfigDOM.js: neither
    # grammar parses it cleanly, and error recovery reads the `if`
    # statements as methods named `if`. Real keyword-named methods outside
    # an error (`delete() {}`) are valid JavaScript and are kept.
    result = extract_file(
        "config.js",
        "export type ChildSet = void;\n"
        "export type TimeoutHandle = TimeoutID;\n"
        "export type NoTimeout = -1;\n"
        "type SelectionInformation = {\n"
        "  if (style.display === 'inline-block') {\n"
        "    if (styleProp == null) {\n",
    )
    names = {node.name for node in result.nodes}
    assert not names & {"if", "return"}
    assert {"ChildSet", "TimeoutHandle", "NoTimeout"} <= names

    clean = extract_file("plain.js", "class Plain { delete() { return 2 } }\n")
    assert "plain.js#Plain.delete" in {node.id for node in clean.nodes}


def test_member_assigned_functions_become_nodes() -> None:
    # Decision #11: the dominant CommonJS style (Express `lib/`) defines
    # functions by assigning them to properties.
    result = extract_file(
        "app.js",
        "var app = exports = module.exports = {};\n"
        "app.init = function init() { this.set('x', 1); };\n"
        "app.set = function set(k, v) { return v; };\n"
        "Foo.prototype.bar = () => 1;\n"
        "module.exports.util = function () { return 1; };\n"
        "exports.other = function () {};\n"
        "this.skip = function () {};\n"
        "obj[key] = function () {};\n"
        "Foo.prototype = function () {};\n",
    )
    nodes = {
        node.id: (node.kind, node.owner, node.exported)
        for node in result.nodes
        if node.kind in ("function", "method")
    }
    assert nodes == {
        "app.js#app.init": ("method", "app", False),
        "app.js#app.set": ("method", "app", False),
        "app.js#Foo.bar": ("method", "Foo", False),
        "app.js#util": ("function", None, True),
        "app.js#other": ("function", None, True),
    }
    calls = {
        (edge.source, edge.target)
        for edge in resolve_edges(result.nodes, result.raw_edges)
        if edge.relation == "calls"
    }
    assert ("app.js#app.init", "app.js#app.set") in calls


def test_repeated_member_assignments_get_unique_ids() -> None:
    # Branch-dependent definitions and a prototype method that shadows a
    # class method each get their own stable `~N` ID.
    result = extract_file(
        "a.js",
        "class Foo { bar() {} }\n"
        "Foo.prototype.bar = function () {};\n"
        "if (a) { x.f = function () {}; } else { x.f = function () {}; }\n",
    )
    ids = [node.id for node in result.nodes]
    assert len(ids) == len(set(ids))
    assert {"a.js#Foo.bar", "a.js#Foo.bar~2", "a.js#x.f", "a.js#x.f~2"} <= set(
        ids
    )
