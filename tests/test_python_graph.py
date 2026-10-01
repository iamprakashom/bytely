from __future__ import annotations

import time
from typing import TYPE_CHECKING

from click.testing import CliRunner

from bytely.cli import main
from bytely.graph.build import build_graph
from bytely.graph.extract import RawEdge, extract_file
from bytely.graph.resolve import resolve_edges
from bytely.graph.types import NodeV1
from bytely.graph.write import read_graph

if TYPE_CHECKING:
    from pathlib import Path


def test_python_decorators_and_nested_classes_are_owned_correctly() -> None:
    source = (
        "import functools\n"
        "\n"
        "\n"
        "class Widget:\n"
        "    @staticmethod\n"
        "    def make():\n"
        "        return Widget()\n"
        "\n"
        "    @classmethod\n"
        "    def from_id(cls, id):\n"
        "        return cls()\n"
        "\n"
        "    @property\n"
        "    def name(self):\n"
        "        return self._name\n"
        "\n"
        "    async def fetch(self):\n"
        "        return await self._load()\n"
        "\n"
        "    def _load(self):\n"
        "        return None\n"
        "\n"
        "\n"
        "@functools.lru_cache\n"
        "def cached():\n"
        "    return 1\n"
        "\n"
        "\n"
        "class Outer:\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            pass\n"
    )
    result = extract_file("widget.py", source)
    by_id = {node.id: (node.kind, node.owner) for node in result.nodes}

    assert by_id["widget.py#Widget"] == ("class", None)
    assert by_id["widget.py#Widget.make"] == ("method", "Widget")
    assert by_id["widget.py#Widget.from_id"] == ("method", "Widget")
    assert by_id["widget.py#Widget.name"] == ("method", "Widget")
    assert by_id["widget.py#Widget.fetch"] == ("method", "Widget")
    assert by_id["widget.py#cached"] == ("function", None)
    assert by_id["widget.py#Outer"] == ("class", None)
    # As in the reference implementation, only methods carry an owner.
    assert by_id["widget.py#Outer.Inner"] == ("class", None)
    # Nested IDs are qualified by every enclosing definition, as in the
    # reference implementation.
    assert by_id["widget.py#Outer.Inner.method"] == ("method", "Inner")


def test_python_def_nested_in_a_method_is_a_scoped_local_function() -> None:
    source = (
        "class Widget:\n"
        "    def render(self):\n"
        "        def fmt():\n"
        "            return 1\n"
        "        return fmt()\n"
    )
    result = extract_file("w.py", source)
    by_id = {node.id: (node.kind, node.owner) for node in result.nodes}
    calls = {
        (edge.source, edge.target)
        for edge in resolve_edges(result.nodes, result.raw_edges)
        if edge.relation == "calls"
    }

    # The reference implementation promotes a def to a method only directly in a
    # class body.
    assert by_id["w.py#Widget.render.fmt"] == ("function", None)
    assert ("w.py#Widget.render", "w.py#Widget.render.fmt") in calls


def test_python_wildcard_import_resolves_bare_calls() -> None:
    module = extract_file("pkg/mod.py", "def helper():\n    return 1\n")
    user = extract_file(
        "pkg/user.py",
        "from pkg.mod import *\n\n\ndef run():\n    return helper()\n",
    )
    nodes = module.nodes + user.nodes
    edges = resolve_edges(nodes, module.raw_edges + user.raw_edges)

    assert any(
        edge.source == "pkg/user.py"
        and edge.target == "pkg/mod.py"
        and edge.relation == "imports"
        for edge in edges
    )
    assert any(
        edge.source == "pkg/user.py#run"
        and edge.target == "pkg/mod.py#helper"
        and edge.relation == "calls"
        for edge in edges
    )


def test_python_multilevel_relative_import_resolves_across_directories() -> (
    None
):
    helper = extract_file("pkg/sub/helper.py", "def greet():\n    return 1\n")
    user = extract_file(
        "pkg/sub/deep/user.py",
        "from ...sub.helper import greet\n\n\ndef run():\n    return greet()\n",
    )
    nodes = helper.nodes + user.nodes
    edges = resolve_edges(nodes, helper.raw_edges + user.raw_edges)

    assert any(
        edge.source == "pkg/sub/deep/user.py"
        and edge.target == "pkg/sub/helper.py"
        and edge.relation == "imports"
        for edge in edges
    )
    assert any(
        edge.source == "pkg/sub/deep/user.py#run"
        and edge.target == "pkg/sub/helper.py#greet"
        and edge.relation == "calls"
        for edge in edges
    )


def test_python_dotted_import_alias_resolves_attribute_calls() -> None:
    module = extract_file("a/b/c.py", "def func():\n    return 1\n")
    user = extract_file(
        "user.py",
        "import a.b.c as name\n\n\ndef run():\n    return name.func()\n",
    )
    nodes = module.nodes + user.nodes
    edges = resolve_edges(nodes, module.raw_edges + user.raw_edges)

    assert any(
        edge.source == "user.py"
        and edge.target == "a/b/c.py"
        and edge.relation == "imports"
        for edge in edges
    )
    assert any(
        edge.source == "user.py#run"
        and edge.target == "a/b/c.py#func"
        and edge.relation == "calls"
        for edge in edges
    )


def test_python_type_checking_guarded_import_still_resolves() -> None:
    helper = extract_file("tc/helper.py", "def do():\n    return 1\n")
    user = extract_file(
        "tc/user.py",
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    from tc.helper import do\n"
        "\n\n"
        "def run():\n"
        "    return do()\n",
    )
    nodes = helper.nodes + user.nodes
    edges = resolve_edges(nodes, helper.raw_edges + user.raw_edges)

    assert any(
        edge.source == "tc/user.py"
        and edge.target == "tc/helper.py"
        and edge.relation == "imports"
        for edge in edges
    )
    assert any(
        edge.source == "tc/user.py#run"
        and edge.target == "tc/helper.py#do"
        and edge.relation == "calls"
        for edge in edges
    )


def test_python_duplicate_function_names_get_collision_suffixed_ids() -> None:
    """`@overload` stubs and version-gated `if/else` branches legitimately
    define the same name multiple times at module scope; each definition
    must still get a stable, unique node id."""
    overload_source = (
        "from typing import overload\n"
        "\n"
        "\n"
        "@overload\n"
        "def process(x: int) -> int: ...\n"
        "@overload\n"
        "def process(x: str) -> str: ...\n"
        "def process(x):\n"
        "    return x\n"
    )
    result = extract_file("ov.py", overload_source)
    process_ids = [node.id for node in result.nodes if node.name == "process"]
    assert process_ids == [
        "ov.py#process",
        "ov.py#process~2",
        "ov.py#process~3",
    ]

    branch_source = (
        "import sys\n"
        "\n"
        "if sys.version_info >= (3, 8):\n"
        "    def compat():\n"
        "        return 1\n"
        "else:\n"
        "    def compat():\n"
        "        return 2\n"
    )
    branch_result = extract_file("branch.py", branch_source)
    compat_ids = [
        node.id for node in branch_result.nodes if node.name == "compat"
    ]
    assert compat_ids == ["branch.py#compat", "branch.py#compat~2"]


def test_python_nested_functions_bind_only_within_their_scope() -> None:
    source = (
        "def outer():\n"
        "    def helper():\n"
        "        return 1\n"
        "    return helper()\n"
        "\n"
        "\n"
        "def other(helper):\n"
        "    return helper()\n"
    )
    result = extract_file("scope.py", source)
    calls = {
        (edge.source, edge.target)
        for edge in resolve_edges(result.nodes, result.raw_edges)
        if edge.relation == "calls"
    }

    assert ("scope.py#outer", "scope.py#outer.helper") in calls
    assert ("scope.py#other", "scope.py#outer.helper") not in calls


def test_out_of_scope_local_blocks_cross_file_name_guess(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text(
        "def outer():\n"
        "    def helper():\n"
        "        return 1\n"
        "    return helper()\n"
        "\n"
        "\n"
        "def other(helper):\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    (tmp_path / "b.py").write_text(
        "def helper():\n    return 2\n", encoding="utf-8"
    )
    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None

    # `other`'s `helper` is a parameter; the file defines a `helper` it cannot
    # see, so no unique-name guess into `b.py` is made either.
    assert not any(
        edge.source == "a.py#other" and edge.relation == "calls"
        for edge in graph.edges
    )


def test_nested_definition_keeps_cross_file_name_guess_ambiguous(
    tmp_path: Path,
) -> None:
    # Mirrors Serde: `test_de.rs` has a top-level `test` and a nested one,
    # while `test_de_error.rs` calls a local variable named `test`.
    (tmp_path / "defs.py").write_text(
        "def check():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def wrapper():\n"
        "    def check():\n"
        "        return 2\n"
        "    return check()\n",
        encoding="utf-8",
    )
    (tmp_path / "use.py").write_text(
        "def run():\n    check = print\n    return check()\n",
        encoding="utf-8",
    )
    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None

    assert not any(
        edge.source == "use.py#run" and edge.relation == "calls"
        for edge in graph.edges
    )


def test_function_local_import_binds_only_inside_its_function(
    tmp_path: Path,
) -> None:
    # Mirrors Click's termui.py: `get_pager_file` wraps an import of the
    # same name, while `echo_via_pager` calls the module-level wrapper.
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "impl.py").write_text(
        "def get_pager_file():\n    return 1\n", encoding="utf-8"
    )
    (pkg / "termui.py").write_text(
        "def get_pager_file():\n"
        "    from .impl import get_pager_file\n"
        "    return get_pager_file()\n"
        "\n"
        "\n"
        "def echo_via_pager():\n"
        "    return get_pager_file()\n",
        encoding="utf-8",
    )
    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None
    calls = {
        (edge.source, edge.target)
        for edge in graph.edges
        if edge.relation == "calls"
    }

    assert calls >= {
        ("pkg/termui.py#get_pager_file", "pkg/impl.py#get_pager_file"),
        ("pkg/termui.py#echo_via_pager", "pkg/termui.py#get_pager_file"),
    }
    assert (
        "pkg/termui.py#echo_via_pager",
        "pkg/impl.py#get_pager_file",
    ) not in calls


def test_importing_one_symbol_twice_is_a_single_binding(
    tmp_path: Path,
) -> None:
    # Mirrors Click: a TYPE_CHECKING import plus a local import of the same
    # class must not make the binding ambiguous.
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "impl.py").write_text(
        "class ProgressBar:\n    pass\n", encoding="utf-8"
    )
    (pkg / "termui.py").write_text(
        "import typing as t\n"
        "\n"
        "if t.TYPE_CHECKING:\n"
        "    from .impl import ProgressBar\n"
        "\n"
        "\n"
        "def progressbar():\n"
        "    from .impl import ProgressBar\n"
        "    return ProgressBar()\n",
        encoding="utf-8",
    )
    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None

    assert any(
        edge.relation == "calls"
        and edge.source == "pkg/termui.py#progressbar"
        and edge.target == "pkg/impl.py#ProgressBar"
        for edge in graph.edges
    )


def test_absolute_import_falls_back_to_a_unique_path_suffix(
    tmp_path: Path,
) -> None:
    # `import tools.fmt` from a `src/` layout matches `src/tools/fmt.py` by
    # suffix; `util` is ambiguous (two files end in `/util.py`), and `os`
    # matches nothing, so neither gets an edge.
    for relative, text in {
        "src/tools/__init__.py": "",
        "src/tools/fmt.py": "def pretty():\n    return 1\n",
        "a/util.py": "",
        "b/util.py": "",
        "src/app.py": (
            "import os\nimport util\nfrom tools.fmt import pretty\n\n\n"
            "def main():\n    return pretty()\n"
        ),
    }.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None

    imports = {
        edge.target
        for edge in graph.edges
        if edge.relation == "imports" and edge.source == "src/app.py"
    }
    # `os` (unknown) and `util` (ambiguous) keep raw-specifier edges, as
    # the reference implementation (P4) — never a guessed file.
    assert imports == {"src/tools/fmt.py", "os", "util"}
    assert any(
        edge.relation == "calls"
        and edge.source == "src/app.py#main"
        and edge.target == "src/tools/fmt.py#pretty"
        for edge in graph.edges
    )


def test_import_resolution_scales_linearly_with_repository_size() -> None:
    # Regression guard for A14: every unresolvable absolute import (stdlib,
    # third-party) used to scan every file path. 1,500 files x 10 imports
    # took minutes that way; with the suffix index it takes milliseconds,
    # so this bound is far from flaky.
    files = [f"pkg{index % 30}/mod{index}.py" for index in range(1500)]
    nodes = [
        NodeV1(
            id=path,
            name=path.rsplit("/", 1)[-1],
            kind="file",
            path=path,
            span="L1-L1",
            body_hash="0",
        )
        for path in files
    ]
    raw_edges = [
        RawEdge(source=path, relation="imports", file=path, specifier=module)
        for path in files
        for module in (
            "os",
            "sys",
            "json",
            "re",
            "typing",
            "pathlib",
            "logging",
            "itertools",
            "functools",
            "collections",
        )
    ]

    start = time.perf_counter()
    edges = resolve_edges(nodes, raw_edges)
    elapsed = time.perf_counter() - start

    # Every import is unresolvable: each keeps a raw-specifier edge (P4),
    # and none is guessed onto a local file.
    import_targets = {
        edge.target for edge in edges if edge.relation == "imports"
    }
    assert import_targets == {
        "os",
        "sys",
        "json",
        "re",
        "typing",
        "pathlib",
        "logging",
        "itertools",
        "functools",
        "collections",
    }
    assert elapsed < 5, f"import resolution took {elapsed:.1f}s"


def test_bare_calls_never_bind_to_methods(tmp_path: Path) -> None:
    # Mirrors Click: `open()` is the builtin, not `_LazyFile.open`.
    (tmp_path / "lazy.py").write_text(
        "class LazyFile:\n"
        "    def __init__(self, name):\n"
        "        self.handle = open(name)\n"
        "\n"
        "    def open(self):\n"
        "        return self.handle\n",
        encoding="utf-8",
    )
    (tmp_path / "use.py").write_text(
        "import lazy\n\n\ndef read(name):\n"
        "    lazy.open()\n"
        "    return open(name)\n",
        encoding="utf-8",
    )
    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None

    assert not any(
        edge.relation == "calls" and edge.target == "lazy.py#LazyFile.open"
        for edge in graph.edges
    )


def test_python_matches_explicit_golden_for_symbols_and_calls() -> None:
    path = "src/app.py"
    source = (
        "import external_package\n"
        "\n"
        "MAX_WIDGETS: int = 8\n"
        "\n"
        "\n"
        "class Widget:\n"
        "    def render(self):\n"
        "        return self.describe()\n"
        "\n"
        "    def describe(self):\n"
        "        return external_package.format()\n"
        "\n"
        "\n"
        "def build_widget():\n"
        "    return Widget()\n"
    )
    result = extract_file(path, source)
    nodes = {node.id: node for node in result.nodes}
    edges = resolve_edges(result.nodes, result.raw_edges)

    expected = {
        (path, "file", None, "L1-L16"),
        (f"{path}#MAX_WIDGETS", "constant", None, "L3-L3"),
        (f"{path}#Widget", "class", None, "L6-L11"),
        (f"{path}#Widget.render", "method", "Widget", "L7-L8"),
        (f"{path}#Widget.describe", "method", "Widget", "L10-L11"),
        (f"{path}#build_widget", "function", None, "L14-L15"),
    }
    actual = {
        (node.id, node.kind, node.owner, node.span) for node in result.nodes
    }

    assert actual == expected
    assert len(nodes) == len(result.nodes)
    render_node = nodes[f"{path}#Widget.render"]
    describe_node = nodes[f"{path}#Widget.describe"]
    build_widget_node = nodes[f"{path}#build_widget"]
    widget_node = nodes[f"{path}#Widget"]
    call_edges = {
        (edge.source, edge.target) for edge in edges if edge.relation == "calls"
    }
    # `external_package.format()` is an unresolved external-module call:
    # the import to `external_package` is unresolvable (no local file),
    # so no `calls` edge is synthesized for it; only the two local calls
    # below (a same-class method call and a constructor call) resolve.
    assert call_edges == {
        (render_node.id, describe_node.id),
        (build_widget_node.id, widget_node.id),
    }


def test_python_project_build_check_cache_and_staleness(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "main.py").write_text(
        "from src.helpers import greet\n"
        "\n"
        "MAX_ITEMS = 8\n"
        "\n"
        "\n"
        "def run():\n"
        "    return greet()\n",
        encoding="utf-8",
    )
    helper_path = source_dir / "helpers.py"
    helper_path.write_text("def greet():\n    return 1\n", encoding="utf-8")
    runner = CliRunner()

    build_result = runner.invoke(main, ["build", str(tmp_path)])
    assert build_result.exit_code == 0, build_result.output
    graph_dir = tmp_path / "bytely"
    graph = read_graph(str(graph_dir))
    assert graph is not None
    node_ids = [node.id for node in graph.nodes]
    assert len(node_ids) == len(set(node_ids))
    run_node = next(
        node for node in graph.nodes if node.id == "src/main.py#run"
    )
    greet_node = next(
        node for node in graph.nodes if node.id == "src/helpers.py#greet"
    )
    assert any(
        edge.source == "src/main.py"
        and edge.target == "src/helpers.py"
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
    assert warm_build.cache_hits == 3
    assert (graph_dir / ".graph" / "wiring.json").read_bytes() == graph_bytes
    assert runner.invoke(main, ["check", str(tmp_path)]).exit_code == 0

    helper_path.write_text(
        "def greet():\n    return 1\n\n\ndef wave():\n    return 2\n",
        encoding="utf-8",
    )
    stale_check = runner.invoke(main, ["check", str(tmp_path)])
    assert stale_check.exit_code == 1
    assert runner.invoke(main, ["build", str(tmp_path)]).exit_code == 0
    assert runner.invoke(main, ["check", str(tmp_path)]).exit_code == 0


def _graph_edges(
    root: Path, files: dict[str, str]
) -> set[tuple[str, str, str, str]]:
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    build_graph(str(root))
    graph = read_graph(str(root / "bytely"))
    assert graph is not None
    return {
        (edge.source, edge.target, edge.relation, edge.confidence)
        for edge in graph.edges
    }


def test_external_import_blocks_guessing_a_base_class(tmp_path: Path) -> None:
    # Review of P7: `View` comes from an external package, so it must not
    # bind to a same-named local stub, nor let self-calls follow the stub.
    edges = _graph_edges(
        tmp_path,
        {
            "tests/stubs.py": (
                "class View:\n    def get(self):\n        return 1\n"
            ),
            "app/views.py": (
                "from django.views import View\n"
                "\n"
                "\n"
                "class MyView(View):\n"
                "    def run(self):\n"
                "        return self.get()\n"
            ),
        },
    )

    assert ("app/views.py#MyView", "View", "extends", "inferred") in edges
    # Nothing but the stub file's own `contains` edges points at the stub.
    stub_links = {
        (source, target)
        for source, target, relation, _ in edges
        if relation != "contains" and target.startswith("tests/stubs.py#")
    }
    assert not stub_links


def test_external_import_blocks_guessing_a_call_across_files(
    tmp_path: Path,
) -> None:
    # `lru_cache` is functools', not a same-named function in another
    # package (measured on real trees: 30 such wrong guesses in 8,182).
    edges = _graph_edges(
        tmp_path,
        {
            "vendor/utils.py": "def lru_cache(fn):\n    return fn\n",
            "app/cache.py": (
                "from functools import lru_cache\n"
                "\n"
                "\n"
                "def build():\n"
                "    return lru_cache(print)\n"
            ),
        },
    )

    assert not any(
        target == "vendor/utils.py#lru_cache" and relation == "calls"
        for _, target, relation, _ in edges
    )


def test_optional_accelerator_import_keeps_the_same_file_definition(
    tmp_path: Path,
) -> None:
    # aiohttp's pattern: a pure-Python class, then `try: from ._c import`
    # an optional compiled replacement. Both bindings are live, so the
    # same-file definition stays a valid target for calls and bases.
    edges = _graph_edges(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/parser.py": (
                "class RawMessage:\n"
                "    pass\n"
                "\n"
                "\n"
                "class Parser(RawMessage):\n"
                "    def parse(self):\n"
                "        return RawMessage()\n"
                "\n"
                "\n"
                "try:\n"
                "    from ._speedups import RawMessage\n"
                "except ImportError:\n"
                "    pass\n"
            ),
        },
    )

    assert (
        "pkg/parser.py#Parser.parse",
        "pkg/parser.py#RawMessage",
        "calls",
        "extracted",
    ) in edges
    assert (
        "pkg/parser.py#Parser",
        "pkg/parser.py#RawMessage",
        "extends",
        "extracted",
    ) in edges


def test_a_class_never_extends_itself(tmp_path: Path) -> None:
    # `class Error(Error)` re-binds an imported or builtin name; the only
    # `Error` in the repo is the class itself, which is never its parent.
    edges = _graph_edges(
        tmp_path,
        {
            "a.py": (
                "from lib import Error\n\n\nclass Error(Error):\n    pass\n"
            ),
            "b.py": "class Warning(Warning):\n    pass\n",
        },
    )
    extends = {(s, t) for s, t, r, _ in edges if r == "extends"}

    assert extends == {("a.py#Error", "Error"), ("b.py#Warning", "Warning")}


def test_imported_interface_binds_through_its_import(tmp_path: Path) -> None:
    # Two interfaces named `Props`; the import says which one is meant.
    edges = _graph_edges(
        tmp_path,
        {
            "src/a.ts": "export interface Props { a: number; }\n",
            "src/b.ts": "export interface Props { b: number; }\n",
            "src/c.ts": (
                "import { Props } from './a';\n"
                "export class C implements Props { a = 1; }\n"
            ),
        },
    )

    assert (
        "src/c.ts#C",
        "src/a.ts#Props",
        "implements",
        "extracted",
    ) in edges


def test_heritage_edges_and_inherited_self_calls(tmp_path: Path) -> None:
    # P7: `extends` binds through an import, then the same file, then a unique
    # name; `self.m()` walks the resolved chain when the class itself has no `m`
    # (as in the reference implementation, up to three levels).
    files = {
        "pkg/__init__.py": "",
        "pkg/base.py": (
            "class Root:\n"
            "    def save(self):\n"
            "        return 1\n"
            "\n"
            "\n"
            "class Left:\n"
            "    def both(self):\n"
            "        return 1\n"
            "\n"
            "\n"
            "class Right:\n"
            "    def both(self):\n"
            "        return 2\n"
        ),
        "pkg/models.py": (
            "import abc\n"
            "from .base import Left, Right, Root\n"
            "\n"
            "\n"
            "class Middle(Root):\n"
            "    pass\n"
            "\n"
            "\n"
            "class Leaf(Middle, abc.ABC, External):\n"
            "    def run(self):\n"
            "        return self.save()\n"
            "\n"
            "\n"
            "class Diamond(Left, Right):\n"
            "    def run(self):\n"
            "        return self.both()\n"
        ),
    }
    for relative, text in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    build_graph(str(tmp_path))
    graph = read_graph(str(tmp_path / "bytely"))
    assert graph is not None
    edges = {
        (edge.source, edge.target, edge.relation, edge.confidence)
        for edge in graph.edges
    }
    extends = {(s, t, c) for s, t, r, c in edges if r == "extends"}

    assert extends == {
        # Through `from .base import Root`: an explicit binding.
        ("pkg/models.py#Middle", "pkg/base.py#Root", "extracted"),
        ("pkg/models.py#Diamond", "pkg/base.py#Left", "extracted"),
        ("pkg/models.py#Diamond", "pkg/base.py#Right", "extracted"),
        # Same file. `abc.ABC` (qualified) is skipped at extraction.
        ("pkg/models.py#Leaf", "pkg/models.py#Middle", "extracted"),
        # An unknown base keeps an edge to its bare name (P4).
        ("pkg/models.py#Leaf", "External", "inferred"),
    }
    # Leaf -> Middle -> Root: the inherited method is two levels up.
    assert (
        "pkg/models.py#Leaf.run",
        "pkg/base.py#Root.save",
        "calls",
        "inferred",
    ) in edges
    # Both parents define `both`: ambiguous, so no guess.
    assert not any(
        s == "pkg/models.py#Diamond.run" and r == "calls"
        for s, _, r, _ in edges
    )


def test_unindexed_relative_import_blocks_guessing_a_base(
    tmp_path: Path,
) -> None:
    # Review: `from ._pb_generated import Message` names a specific module
    # that is not indexed; `Foo` must not extend an unrelated `Message`.
    edges = _graph_edges(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/model.py": (
                "from ._pb_generated import Message\n"
                "\n"
                "\n"
                "class Foo(Message):\n"
                "    pass\n"
            ),
            "other/messages.py": "class Message:\n    pass\n",
        },
    )

    assert ("pkg/model.py#Foo", "Message", "extends", "inferred") in edges
    assert not any(
        target == "other/messages.py#Message" and relation == "extends"
        for _, target, relation, _ in edges
    )


def test_ambiguous_absolute_import_is_not_treated_as_external(
    tmp_path: Path,
) -> None:
    # `util` matches two files by suffix: ambiguous, not external, so the
    # call keeps its unique-name guess.
    edges = _graph_edges(
        tmp_path,
        {
            "a/util.py": "",
            "b/util.py": "def helper():\n    return 1\n",
            "app.py": (
                "from util import helper\n\n\ndef run():\n    return helper()\n"
            ),
        },
    )

    assert ("app.py#run", "b/util.py#helper", "calls", "inferred") in edges
