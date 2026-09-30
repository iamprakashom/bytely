from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from bytely.cli import main
from bytely.graph.map import render_map
from bytely.graph.refresh import refresh_graph

if TYPE_CHECKING:
    from pathlib import Path


def _project(root: Path) -> None:
    files = {
        "src/util.py": (
            "def helper():\n    return 1\n\n\n"
            "def unused():\n    return 2\n"
        ),
        "src/app.py": (
            "from .util import helper\n\n\n"
            "def main():\n    return helper() + helper()\n\n\n"
            "def other():\n    return helper()\n"
        ),
        "tests/test_app.py": (
            "from src.app import main\n\n\n"
            "def test_main():\n    assert main()\n"
        ),
        "setup.py": "",
    }
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def test_map_lists_clusters_hubs_and_hotspots(tmp_path: Path) -> None:
    _project(tmp_path)
    graph = refresh_graph(str(tmp_path)).graph

    text = render_map(graph)

    lines = text.splitlines()
    assert lines[0].startswith("repo map — 4 files · 5 symbols · ")
    assert lines[0].endswith(" · python")
    assert lines[2] == (
        "src/      2 files · 4 symbols   hubs: helper (util.py, 2←), "
        "main (app.py, 1←)"
    )
    assert lines[3] == "setup.py  1 file · 0 symbols"
    assert lines[4] == "tests/    1 file · 1 symbol"
    assert lines[6] == (
        "hotspots: helper · function · src/util.py:L1-L2 · 2←  "
        "main · function · src/app.py:L4-L5 · 1←"
    )


def test_map_limits_listed_folders(tmp_path: Path) -> None:
    _project(tmp_path)
    graph = refresh_graph(str(tmp_path)).graph

    text = render_map(graph, max_dirs=1)

    assert "setup.py" not in text
    assert "… 2 more (--max-dirs to show)" in text


def test_map_command_refreshes_and_prints(tmp_path: Path) -> None:
    _project(tmp_path)

    result = CliRunner().invoke(main, ["map", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("repo map — 4 files")
    assert (tmp_path / "bytely" / ".graph" / "wiring.json").is_file()
