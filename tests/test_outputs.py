from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from bytely.graph.build import build_graph
from bytely.graph.check import check_graph
from bytely.graph.outputs import card_paths

if TYPE_CHECKING:
    from pathlib import Path


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def test_build_writes_a_card_per_source_file_and_an_index(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        {
            "pkg/util.py": (
                "def greet(name: str) -> str:\n"
                "    return name\n"
                "\n"
                "\n"
                "class Box:\n"
                "    def open(self) -> None:\n"
                "        pass\n"
            ),
            "pkg/empty.py": "",
        },
    )
    build_graph(str(tmp_path))
    out = tmp_path / "bytely"

    assert (out / "pkg" / "util.md").read_text("utf-8") == (
        "# pkg/util.py\n"
        "\n"
        "- greet · function · L1-L2 — def greet(name: str) -> str\n"
        "- Box · class · L5-L7 — class Box\n"
        "- open · method · L6-L7 — def open(self) -> None\n"
    )
    assert (out / "pkg" / "empty.md").read_text("utf-8") == (
        "# pkg/empty.py\n\n"
    )
    index = (out / "INDEX.md").read_text("utf-8")
    assert "- 2 cards, 3 definitions" in index
    assert "Files by extension: .py 2" in index


def test_unchanged_rebuild_does_not_rewrite_cards(tmp_path: Path) -> None:
    _write(tmp_path, {"a.py": "def f():\n    return 1\n"})
    build_graph(str(tmp_path))
    card = tmp_path / "bytely" / "a.md"
    before = card.stat().st_mtime_ns

    build_graph(str(tmp_path))

    assert card.stat().st_mtime_ns == before


def test_deleted_source_loses_its_card_and_empty_folder(
    tmp_path: Path,
) -> None:
    _write(tmp_path, {"keep.py": "x = 1\n", "old/gone.py": "def g(): pass\n"})
    build_graph(str(tmp_path))
    out = tmp_path / "bytely"
    assert (out / "old" / "gone.md").is_file()

    (tmp_path / "old" / "gone.py").unlink()
    build_graph(str(tmp_path))

    assert not (out / "old").exists()
    assert (out / "keep.md").is_file()


def test_shared_output_folder_keeps_the_users_own_markdown(
    tmp_path: Path,
) -> None:
    # `--dir` may point at a folder that already holds other markdown; only
    # cards this writer created are ever deleted.
    repo, out = tmp_path / "repo", tmp_path / "docs"
    _write(repo, {"a.py": "def f(): pass\n"})
    _write(out, {"notes.md": "mine\n"})

    build_graph(str(repo), str(out))
    (repo / "a.py").unlink()
    _write(repo, {"b.py": "def g(): pass\n"})
    build_graph(str(repo), str(out))

    assert (out / "notes.md").read_text("utf-8") == "mine\n"
    assert (out / "b.md").is_file()
    assert not (out / "a.md").exists()


def test_existing_markdown_where_a_card_goes_is_never_overwritten(
    tmp_path: Path,
) -> None:
    repo, out = tmp_path / "repo", tmp_path / "docs"
    _write(repo, {"a.py": "def f(): pass\n"})
    _write(out, {"a.md": "mine\n"})

    with pytest.raises(FileExistsError, match=r"a\.md"):
        build_graph(str(repo), str(out))

    assert (out / "a.md").read_text("utf-8") == "mine\n"


def test_sources_that_would_share_a_card_keep_their_extensions() -> None:
    assert card_paths(["src/a.js", "src/a.ts", "src/b.py", "Dockerfile"]) == {
        "Dockerfile": "Dockerfile.md",
        "src/a.js": "src/a.js.md",
        "src/a.ts": "src/a.ts.md",
        "src/b.py": "src/b.md",
    }


def test_check_writes_no_cards(tmp_path: Path) -> None:
    _write(tmp_path, {"a.py": "def f(): pass\n"})
    build_graph(str(tmp_path), output_formats=())
    assert not (tmp_path / "bytely" / "a.md").exists()

    fresh, _ = check_graph(str(tmp_path))

    assert fresh
    assert not (tmp_path / "bytely" / "a.md").exists()


def test_unknown_output_format_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, {"a.py": "x = 1\n"})
    with pytest.raises(ValueError, match="Unknown output format 'pdf'"):
        build_graph(str(tmp_path), output_formats=("pdf",))


def test_graph_is_written_to_wiring_json_and_legacy_file_removed(
    tmp_path: Path,
) -> None:
    _write(tmp_path, {"a.py": "def f(): pass\n"})
    (tmp_path / "bytely").mkdir()
    (tmp_path / "bytely" / "graph.json").write_text("{}", encoding="utf-8")

    build_graph(str(tmp_path))

    assert (tmp_path / "bytely" / ".graph" / "wiring.json").is_file()
    assert not (tmp_path / "bytely" / "graph.json").exists()
