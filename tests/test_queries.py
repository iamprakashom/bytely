"""The query commands: skeleton, callers, grep, and ask."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from bytely.cli import main
from bytely.graph.refresh import refresh_graph
from bytely.query.ask import rank, render_ask, terms
from bytely.query.callers import render_callers
from bytely.query.common import SourceReader, find_file, find_symbols
from bytely.query.grep import render_grep
from bytely.query.skeleton import render_skeleton

if TYPE_CHECKING:
    from pathlib import Path

    from bytely.graph.types import GraphV1

FILES = {
    "shop/cart.py": (
        "from shop.pricing import apply_discount\n"
        "\n"
        "\n"
        "class Cart:\n"
        "    def total(self, items: list[int]) -> int:\n"
        "        return apply_discount(sum(items))\n"
        "\n"
        "    def checkout(self) -> int:\n"
        "        return self.total([1, 2])\n"
        "\n"
        "\n"
        "class GiftCart(Cart, External):\n"
        "    pass\n"
        "\n"
        "\n"
        "TAX_RATE = 0.2  # tax\n"
    ),
    "shop/pricing.py": (
        "def apply_discount(amount: int) -> int:\n"
        "    def rounded(value: int) -> int:\n"
        "        return value  # discount rounding\n"
        "    return rounded(amount - 1)\n"
    ),
    "tests/test_cart.py": (
        "from shop.cart import Cart\n"
        "\n"
        "\n"
        "def test_cart_total_applies_discount() -> None:\n"
        "    cart = Cart()\n"
        "    assert cart.total([3]) == 2\n"
    ),
    "shop/__init__.py": "",
}


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, GraphV1]:
    for relative, text in FILES.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return tmp_path, refresh_graph(str(tmp_path)).graph


def test_find_symbols_prefers_the_most_specific_tier(
    project: tuple[Path, GraphV1],
) -> None:
    _, graph = project

    def ids(symbol: str) -> list[str]:
        return [node.id for node in find_symbols(graph, symbol)]

    assert ids("shop/cart.py#Cart.total") == ["shop/cart.py#Cart.total"]
    assert ids("Cart.total") == ["shop/cart.py#Cart.total"]
    # A bare name matches the method through the qualified-name suffix.
    assert ids("total") == ["shop/cart.py#Cart.total"]
    assert ids("rounded") == ["shop/pricing.py#apply_discount.rounded"]
    assert ids("missing") == []


def test_find_file_accepts_relative_absolute_and_suffix_paths(
    project: tuple[Path, GraphV1],
) -> None:
    root, graph = project
    for spelling in ("shop/cart.py", str(root / "shop" / "cart.py"), "cart.py"):
        node = find_file(graph, root, spelling)
        assert node is not None, spelling
        assert node.path == "shop/cart.py", spelling
    assert find_file(graph, root, "nope.py") is None


def test_skeleton_lists_definitions_in_source_order(
    project: tuple[Path, GraphV1],
) -> None:
    root, graph = project
    file_node = find_file(graph, root, "shop/cart.py")
    assert file_node is not None

    text = render_skeleton(graph, file_node)
    lines = text.splitlines()
    assert lines[0].startswith("skeleton — shop/cart.py (")
    assert lines[1] == "- L4-L9  class Cart  class Cart"
    assert lines[2] == (
        "- L5-L6  method total (in Cart)  "
        "def total(self, items: list[int]) -> int"
    )
    assert "- L12-L13  class GiftCart  class GiftCart(Cart, External)" in lines


def test_callers_in_out_and_depth(project: tuple[Path, GraphV1]) -> None:
    _, graph = project
    discount = find_symbols(graph, "apply_discount")

    one = render_callers(graph, discount, direction="in", depth=1)
    assert "calls ← total · method · shop/cart.py:L5-L6" in one
    assert "checkout" not in one

    two = render_callers(graph, discount, direction="in", depth=2)
    assert (
        "    calls ← checkout · method · shop/cart.py:L8-L9  (via total)" in two
    )
    assert "test_cart_total_applies_discount" in two

    closure = render_callers(graph, discount, direction="in", depth=None)
    assert "checkout" in closure

    out = render_callers(
        graph, find_symbols(graph, "GiftCart"), direction="out"
    )
    assert "extends → Cart · class · shop/cart.py:L4-L9" in out
    # A base the graph cannot resolve is shown as an external name.
    assert "extends → External (external)" in out

    nothing = render_callers(graph, find_symbols(graph, "TAX_RATE"))
    assert nothing.endswith("  (no callers)\n")


def test_grep_groups_hits_by_enclosing_symbol(
    project: tuple[Path, GraphV1],
) -> None:
    root, graph = project
    reader = SourceReader(root)

    text = render_grep(graph, reader, "discount", ignore_case=True)
    assert text.startswith("'discount' — 5 hits in 5 symbols across 3 files")
    # The nested function's comment belongs to it, not its parent.
    assert "rounded · function · shop/pricing.py:L2-L3" in text
    assert "  L3: return value  # discount rounding" in text
    # A top-of-file import is module level.
    assert "shop/cart.py (module level)" in text
    # Referenced groups come before unreferenced ones (the test function).
    assert text.index("apply_discount · function") < text.index(
        "test_cart_total_applies_discount · function"
    )

    scoped = render_grep(graph, reader, "discount", scope="tests")
    assert "across 1 files (searched 1 indexed files)" in scoped
    assert render_grep(graph, reader, "a.b", fixed=True).startswith(
        "'a.b' — 0 hits"
    )
    with pytest.raises(ValueError, match="Invalid pattern"):
        render_grep(graph, reader, "(")


def test_terms_split_identifiers_and_drop_stop_words() -> None:
    assert terms("How does applyDiscount handle the TAX_RATE?") == [
        "apply",
        "discount",
        "handle",
        "tax",
        "rate",
    ]


def test_ask_ranks_implementation_above_tests(
    project: tuple[Path, GraphV1],
) -> None:
    _, graph = project

    top = [hit.node.id for hit in rank(graph, "cart total discount")][:2]
    assert top[0] == "shop/cart.py#Cart.total"
    assert "tests/test_cart.py#test_cart_total_applies_discount" not in top

    about_tests = rank(graph, "tests for cart total")
    assert about_tests[0].node.path == "tests/test_cart.py"
    assert rank(graph, "the of and") == []


TIERS = {
    "lib/exact.py": "def parse_option(value):\n    return value\n",
    "lib/prefix.py": "def parse_optional(value):\n    return value\n",
    "lib/fragment.py": "def parse_adoption(value):\n    return value\n",
    "lib/unrelated.py": "def render(value):\n    return value\n",
}


def test_ask_ranks_exact_words_above_prefixes_above_fragments(
    tmp_path: Path,
) -> None:
    for relative, text in TIERS.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    graph = refresh_graph(str(tmp_path)).graph

    order = [
        hit.node.name
        for hit in rank(graph, "option")
        if hit.node.kind != "file"
    ]
    assert order == ["parse_option", "parse_optional", "parse_adoption"]
    # An abbreviation reaches the full word; ranking is deterministic.
    abbreviated = [hit.node.name for hit in rank(graph, "opt")]
    assert abbreviated[0] in {"parse_option", "parse_optional"}
    assert "parse_adoption" in abbreviated
    assert abbreviated == [hit.node.name for hit in rank(graph, "opt")]


def test_ask_ranks_a_short_match_above_a_long_body_with_the_same_words(
    tmp_path: Path,
) -> None:
    filler = "\n".join(f"    setting_{i} = {i}" for i in range(60))
    (tmp_path / "registry.py").write_text(
        "class Registry:\n"
        '    """Holds entries by name; a dash prefix marks a flag name."""\n'
        f"{filler}\n"
        "    # The prefix and dash are kept with the name.\n",
        encoding="utf-8",
    )
    (tmp_path / "helpers.py").write_text(
        "def strip_marker(text):\n"
        '    """Drop the dash prefix from a name."""\n'
        '    return text.lstrip("-")\n',
        encoding="utf-8",
    )
    graph = refresh_graph(str(tmp_path)).graph

    top = rank(graph, "dash prefix of a name")[0]
    assert top.node.name == "strip_marker"


def test_ask_full_shows_the_top_hits_whole_up_to_a_cap(
    tmp_path: Path,
) -> None:
    steps = "\n".join(f"    step_{i} = {i}" for i in range(100))
    for name in ("first", "second", "third"):
        (tmp_path / f"{name}.py").write_text(
            f"def {name}_routine():\n{steps}\n", encoding="utf-8"
        )
    graph = refresh_graph(str(tmp_path)).graph
    reader = SourceReader(tmp_path)

    text = render_ask(graph, reader, "routine step", limit=3, full=True)
    hits = text.split("\n\n")[1:4]
    # The top two show 80 lines and point at the rest of the file.
    for hit in hits[:2]:
        assert "step_78 = 78" in hit
        assert "step_79 = 79" not in hit
        assert "… +21 more lines (open " in hit
    # The third keeps the short excerpt.
    assert "step_6 = 6" in hits[2]
    assert "step_7 = 7" not in hits[2]


def test_ask_cuts_signatures_that_hold_a_whole_definition(
    tmp_path: Path,
) -> None:
    body = " ".join(f"total += {i};" for i in range(80))
    (tmp_path / "utils.js").write_text(
        f"exports.addAll = function(total) {{ {body} return total; }};\n",
        encoding="utf-8",
    )
    graph = refresh_graph(str(tmp_path)).graph

    text = render_ask(graph, SourceReader(tmp_path), "addAll", limit=1)
    header = text.split("\n")[3]
    assert header.startswith("   exports.addAll = function(total)")
    assert header.endswith("…")
    assert len(header.strip()) == 200


def test_ask_routes_structural_questions_and_shows_source(
    project: tuple[Path, GraphV1],
) -> None:
    root, graph = project
    reader = SourceReader(root)

    structural = render_ask(graph, reader, "who calls apply_discount?")
    assert "(structural: callers of apply_discount)" in structural
    assert "calls ← total · method" in structural

    text = render_ask(graph, reader, "apply discount", limit=1, source=True)
    assert "1. apply_discount · function · shop/pricing.py:L1-L4" in text
    assert "     1  def apply_discount(amount: int) -> int:" in text

    empty = render_ask(graph, reader, "zzzz")
    assert "(lexical, 0 results)" in empty
    assert "no hits; don't re-ask with new wording" in empty


def test_ask_suggests_switching_tools_rather_than_rewording(
    project: tuple[Path, GraphV1],
) -> None:
    root, graph = project
    reader = SourceReader(root)

    few = render_ask(graph, reader, "gift")
    assert few.rstrip().endswith(
        "[bytely] only 1 hit; don't re-ask with new wording, switch tool: "
        'bytely grep "<literal>" for every occurrence · bytely skeleton '
        "shop/cart.py for a file's API · bytely callers "
        "shop/cart.py#GiftCart --direction out for what it calls."
    )
    # Few hits shown because of `limit` is not few matches.
    many = render_ask(graph, reader, "cart total discount", limit=1)
    assert "[bytely] Not here? Don't reword the question" in many
    assert "bytely skeleton shop/cart.py lists the file" in many
    assert "only 1 hit" not in many


def test_query_commands_run_from_the_cli(tmp_path: Path) -> None:
    for relative, text in FILES.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    runner = CliRunner()
    root = ["--root", str(tmp_path)]

    commands = {
        "skeleton": ["skeleton", "cart.py", *root],
        "callers": ["callers", "apply_discount", "--depth", "all", *root],
        "grep": ["grep", "discount", "-i", *root],
        "ask": ["ask", "cart total", "-n", "2", "--full", *root],
    }
    for name, arguments in commands.items():
        result = runner.invoke(main, arguments)
        assert result.exit_code == 0, (name, result.output)

    missing = runner.invoke(main, ["callers", "nothing", *root])
    assert missing.exit_code != 0
    assert "No symbol matches 'nothing'" in missing.output
    bad_depth = runner.invoke(main, ["callers", "Cart", "--depth", "0", *root])
    assert bad_depth.exit_code != 0


def test_find_file_keeps_leading_dots_and_rejects_parent_paths(
    tmp_path: Path,
) -> None:
    (tmp_path / ".config").mkdir()
    (tmp_path / ".config" / "tool.py").write_text("x = 1\n", "utf-8")
    (tmp_path / "tool.py").write_text("y = 2\n", "utf-8")
    graph = refresh_graph(str(tmp_path)).graph

    hidden = find_file(graph, tmp_path, ".config/tool.py")
    assert hidden is not None
    assert hidden.path == ".config/tool.py"
    nested = find_file(graph, tmp_path, "./.config/tool.py")
    assert nested is not None
    assert nested.path == ".config/tool.py"
    # `../tool.py` is outside the tree; it must not match `tool.py`.
    assert find_file(graph, tmp_path, "../tool.py") is None


def test_callers_limit_counts_every_listed_edge(tmp_path: Path) -> None:
    lines = ["def target():\n    return 1\n"]
    lines += [f"def caller{i}():\n    return target()\n" for i in range(5)]
    (tmp_path / "m.py").write_text("\n\n".join(lines), "utf-8")
    graph = refresh_graph(str(tmp_path)).graph
    seeds = find_symbols(graph, "target")

    capped = render_callers(graph, seeds, limit=3)
    assert capped.count("calls ←") == 3
    assert "stopped after 3 symbols; more exist" in capped

    exact = render_callers(graph, seeds, limit=5)
    assert exact.count("calls ←") == 5
    assert "stopped after" not in exact
