"""Method calls through variables whose type the code states (P13)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bytely.graph.build import build_graph
from bytely.graph.write import read_graph

if TYPE_CHECKING:
    from pathlib import Path


def _calls(root: Path, files: dict[str, str]) -> set[tuple[str, str, str]]:
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    build_graph(str(root))
    graph = read_graph(str(root / "bytely"))
    assert graph is not None
    return {
        (edge.source, edge.target, edge.confidence)
        for edge in graph.edges
        if edge.relation == "calls"
    }


PY_FORMATTER = (
    "from typing import Optional, overload\n"
    "\n"
    "\n"
    "class Formatter:\n"
    "    def section(self) -> None:\n"
    "        pass\n"
    "\n"
    "    @overload\n"
    "    def write(self, text: str) -> None: ...\n"
    "    @overload\n"
    "    def write(self, text: bytes) -> None: ...\n"
    "    def write(self, text):\n"
    "        pass\n"
    "\n"
    "\n"
    "class Loud(Formatter):\n"
    "    pass\n"
)


def test_python_annotated_parameters_and_constructed_locals(
    tmp_path: Path,
) -> None:
    calls = _calls(
        tmp_path,
        {
            "fmt.py": PY_FORMATTER,
            "app.py": (
                "from fmt import Formatter, Loud\n"
                "\n"
                "\n"
                "def plain(f: Formatter) -> None:\n"
                "    f.section()\n"
                "\n"
                "\n"
                "def optional(f: 'Formatter | None', g: Optional[Formatter]):\n"
                "    f.section()\n"
                "    g.write('x')\n"
                "\n"
                "\n"
                "def local() -> None:\n"
                "    loud = Loud()\n"
                "    loud.section()\n"
                "\n"
                "\n"
                "def reassigned(f: Formatter) -> None:\n"
                "    f = load()\n"
                "    f.section()\n"
                "\n"
                "\n"
                "def container(items: list[Formatter]) -> None:\n"
                "    items.section()\n"
            ),
        },
    )
    section = "fmt.py#Formatter.section"
    assert ("app.py#plain", section, "extracted") in calls
    assert ("app.py#optional", section, "extracted") in calls
    # `@overload` stubs come first; the implementation is the last `write`.
    assert ("app.py#optional", "fmt.py#Formatter.write~3", "extracted") in calls
    # `Loud` inherits `section` from `Formatter`.
    assert ("app.py#local", section, "extracted") in calls
    # A reassignment from unknown code, or a container type, gives no type.
    assert not {c for c in calls if c[0] == "app.py#reassigned"}
    assert not {c for c in calls if c[0] == "app.py#container"}


def test_rust_parameters_and_let_bindings(tmp_path: Path) -> None:
    calls = _calls(
        tmp_path,
        {
            "src/lib.rs": (
                "pub struct Cache;\n"
                "impl Cache {\n"
                "    pub fn new() -> Self { Cache }\n"
                "    pub fn get(&self) -> u32 { 1 }\n"
                "    pub fn copy(&self) -> Self { let c = Self::new(); c }\n"
                "}\n"
                "pub fn by_ref(c: &Cache) -> u32 { c.get() }\n"
                "pub fn by_mut(c: &mut Cache) -> u32 { c.get() }\n"
                "pub fn boxed(c: Box<Cache>) -> u32 { c.get() }\n"
                "pub fn local() -> u32 {\n"
                "    let c = Cache::new();\n"
                "    let d: Cache = Cache;\n"
                "    let e = Cache {};\n"
                "    c.get() + d.get() + e.get()\n"
                "}\n"
            )
        },
    )
    get = "src/lib.rs#Cache.get"
    for caller in ("by_ref", "by_mut", "boxed", "local"):
        assert (f"src/lib.rs#{caller}", get, "extracted") in calls


def test_javascript_new_expression_locals(tmp_path: Path) -> None:
    calls = _calls(
        tmp_path,
        {
            "app.js": (
                "class View { render() { return 1; } }\n"
                "function show() {\n"
                "  const view = new View();\n"
                "  return view.render();\n"
                "}\n"
            )
        },
    )
    assert ("app.js#show", "app.js#View.render", "extracted") in calls


def test_python_rebinding_forms_clear_the_type(tmp_path: Path) -> None:
    # Each function first types `f` as Formatter, then rebinds it another
    # way; none of the calls may resolve to Formatter.section.
    rebinds = {
        "loop": "for f in items:\n        f.section()",
        "with_as": "with open('x') as f:\n        f.section()",
        "except_as": (
            "try:\n        pass\n    except E as f:\n        f.section()"
        ),
        "tuple": "f, g = pair()\n    f.section()",
        "walrus": "if (f := load()):\n        f.section()",
        "augmented": "f += other\n    f.section()",
    }
    source = "from fmt import Formatter\n"
    for name, body in rebinds.items():
        source += (
            f"\n\ndef {name}(items, pair, load, other):\n"
            f"    f = Formatter()\n"
            f"    {body}\n"
        )
    calls = _calls(tmp_path, {"fmt.py": PY_FORMATTER, "app.py": source})
    for name in rebinds:
        assert (
            f"app.py#{name}",
            "fmt.py#Formatter.section",
            "extracted",
        ) not in calls, name


def test_rust_rebinding_patterns_clear_the_type(tmp_path: Path) -> None:
    calls = _calls(
        tmp_path,
        {
            "src/lib.rs": (
                "pub struct Cache;\n"
                "impl Cache {\n"
                "    pub fn new() -> Self { Cache }\n"
                "    pub fn get(&self) -> u32 { 1 }\n"
                "}\n"
                "pub fn if_let(o: Option<u32>) -> u32 {\n"
                "    let c = Cache::new();\n"
                "    if let Some(c) = o { return c.get(); }\n"
                "    0\n"
                "}\n"
                "pub fn matched(o: Option<u32>) -> u32 {\n"
                "    let c = Cache::new();\n"
                "    match o { Some(c) => c.get(), None => 0 }\n"
                "}\n"
                "pub fn looped(v: Vec<u32>) -> u32 {\n"
                "    let c = Cache::new();\n"
                "    for c in v { return c.get(); }\n"
                "    0\n"
                "}\n"
                "pub fn tupled() -> u32 {\n"
                "    let c = Cache::new();\n"
                "    let (c, _) = pair();\n"
                "    c.get()\n"
                "}\n"
                "pub fn reassigned() -> u32 {\n"
                "    let mut c = Cache::new();\n"
                "    c = other();\n"
                "    c.get()\n"
                "}\n"
            )
        },
    )
    for name in ("if_let", "matched", "looped", "tupled", "reassigned"):
        assert not {
            c
            for c in calls
            if c[0] == f"src/lib.rs#{name}" and c[1].endswith("Cache.get")
        }, name


def test_javascript_parameters_and_patterns_shadow_outer_types(
    tmp_path: Path,
) -> None:
    # TSX parses these files (the type annotations make tree-sitter-
    # javascript fail), so parameters are `required_parameter` nodes.
    calls = _calls(
        tmp_path,
        {
            "app.js": (
                "class Cache { get() { return 1; } }\n"
                "function outer(x: number) {\n"
                "  const c = new Cache();\n"
                "  const typed = (c: number) => c.get();\n"
                "  const bare = c => c.get();\n"
                "  const destructured = ({ c }) => c.get();\n"
                "  for (const c of xs) { c.get(); }\n"
                "  try {} catch (c) { c.get(); }\n"
                "  return c.get();\n"
                "}\n"
            )
        },
    )
    get_calls = {c[0] for c in calls if c[1] == "app.js#Cache.get"}
    # Every nested function rebinds `c` as its own parameter, so none may
    # resolve through the outer `c`. `for` and `catch` rebind `c` inside
    # `outer` itself, so its own `c.get()` is (conservatively) dropped too.
    assert get_calls == set(), get_calls


def test_non_function_values_are_not_scanned_as_functions(
    tmp_path: Path,
) -> None:
    # A member-assigned value that is not a function mints no node, and a
    # const bound to a call is not a function: no scan, no spurious types.
    calls = _calls(
        tmp_path,
        {
            "app.js": (
                "class Cache { get() { return 1; } }\n"
                "const make = partial(build, new Cache());\n"
                "app.handler = factory(function (c) { return c.get(); });\n"
            )
        },
    )
    assert ("app.js#make", "app.js#Cache.get", "extracted") not in calls
