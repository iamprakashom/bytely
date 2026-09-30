from __future__ import annotations

from typing import TYPE_CHECKING

from bytely.graph.build import build_graph
from bytely.graph.refresh import RefreshResult, refresh_graph
from bytely.graph.write import graph_path

if TYPE_CHECKING:
    from pathlib import Path


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def _names(result: RefreshResult) -> set[str]:
    return {node.name for node in result.graph.nodes}


def test_refresh_after_a_build_loads_without_rebuilding(tmp_path: Path) -> None:
    _write(tmp_path, {"a.py": "def f():\n    return 1\n"})
    build_graph(str(tmp_path))

    result = refresh_graph(str(tmp_path))

    assert not result.rebuilt
    assert "f" in _names(result)


def test_refresh_with_no_graph_builds_one(tmp_path: Path) -> None:
    _write(tmp_path, {"a.py": "def f():\n    return 1\n"})

    result = refresh_graph(str(tmp_path))

    assert result.rebuilt
    assert graph_path(tmp_path / "bytely").is_file()
    assert not refresh_graph(str(tmp_path)).rebuilt


def test_edit_add_and_delete_each_trigger_one_rebuild(tmp_path: Path) -> None:
    _write(tmp_path, {"a.py": "def f():\n    return 1\n"})
    refresh_graph(str(tmp_path))

    _write(tmp_path, {"a.py": "def renamed():\n    return 1\n"})
    edited = refresh_graph(str(tmp_path))
    assert edited.rebuilt
    assert "renamed" in _names(edited)
    assert "f" not in _names(edited)
    assert not refresh_graph(str(tmp_path)).rebuilt

    _write(tmp_path, {"b.py": "def g():\n    return 2\n"})
    added = refresh_graph(str(tmp_path))
    assert added.rebuilt
    assert "g" in _names(added)

    (tmp_path / "b.py").unlink()
    removed = refresh_graph(str(tmp_path))
    assert removed.rebuilt
    assert "g" not in _names(removed)


def test_changed_include_patterns_trigger_a_rebuild(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {"src/a.py": "def f(): pass\n", "lib/b.py": "def g(): pass\n"},
    )
    refresh_graph(str(tmp_path))

    narrowed = refresh_graph(str(tmp_path), include_patterns=["src/**"])

    assert narrowed.rebuilt
    assert "g" not in _names(narrowed)


def test_missing_graph_file_is_rebuilt_even_with_a_saved_snapshot(
    tmp_path: Path,
) -> None:
    _write(tmp_path, {"a.py": "def f(): pass\n"})
    build_graph(str(tmp_path))
    graph_path(tmp_path / "bytely").unlink()

    result = refresh_graph(str(tmp_path))

    assert result.rebuilt
    assert "f" in _names(result)
