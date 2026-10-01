"""`bytely init` / `bytely uninstall`: wiring AI hosts, and undoing it."""

from __future__ import annotations

import filecmp
import json
import shutil
from typing import TYPE_CHECKING

from click.testing import CliRunner

from bytely.cli import main
from bytely.hosts import files
from bytely.hosts.registry import HOST_IDS, Probe
from bytely.hosts.wiring import run_init, run_uninstall, select_hosts

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    repo, home = tmp_path / "repo", tmp_path / "home"
    (repo / ".cursor").mkdir(parents=True)
    (home / ".codex").mkdir(parents=True)
    (home / ".claude").mkdir(parents=True)
    (home / ".gemini" / "config").mkdir(parents=True)
    (repo / "AGENTS.md").write_text(
        "# My project\n\nTeam notes, keep me.\n", encoding="utf-8"
    )
    (repo / ".mcp.json").write_text(
        '{\n  "mcpServers": {\n    "other": {"command": "x"}\n  }\n}\n',
        encoding="utf-8",
    )
    (repo / ".cursor" / "mcp.json").write_text("{ broken", encoding="utf-8")
    (home / ".codex" / "config.toml").write_text(
        '[mcp_servers.other]\ncommand = "o"\n', encoding="utf-8"
    )
    (repo / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return repo, home


def _differences(left: Path, right: Path) -> list[str]:
    """Files that differ, comparing JSON by content (it is re-indented)."""
    comparison = filecmp.dircmp(left, right)
    out = list(comparison.left_only + comparison.right_only)
    for name in comparison.diff_files:
        if name.endswith(".json") and json.loads(
            (left / name).read_text("utf-8")
        ) == json.loads((right / name).read_text("utf-8")):
            continue
        out.append(name)
    for sub in comparison.common_dirs:
        nested = _differences(left / sub, right / sub)
        out += [f"{sub}/{name}" for name in nested]
    return out


def test_host_selection_explicit_all_and_detected(tmp_path: Path) -> None:
    repo, home = _setup(tmp_path)
    probe = Probe(repo, home)

    assert select_hosts(probe, ["cursor", "nope", "cursor"], False) == (
        ["cursor"],
        ["nope"],
    )
    assert select_hosts(probe, None, True) == (list(HOST_IDS), [])
    detected, _ = select_hosts(probe, None, False)
    # Claude Code always; then `.cursor` in the repo, `~/.codex`, and
    # `~/.gemini` (plus its config/).
    assert detected == ["claude", "agents", "cursor", "gemini", "antigravity"]
    assert select_hosts(probe, None, False, others=False) == (["claude"], [])


def test_init_writes_instructions_and_mcp_keeping_user_content(
    tmp_path: Path,
) -> None:
    repo, home = _setup(tmp_path)

    report = run_init(repo, home, all_hosts=True)

    actions = {
        change.path.relative_to(tmp_path).as_posix(): change.action
        for change in report.changes
    }
    assert actions["repo/.cursor/mcp.json"] == "skipped-unparseable"
    assert actions["repo/.gitignore"] == "created"
    assert report.built
    agents = (repo / "AGENTS.md").read_text("utf-8")
    assert agents.startswith("# My project\n\nTeam notes, keep me.\n\n")
    assert agents.count(files.START) == 1  # shared by three hosts, written once
    servers = json.loads((repo / ".mcp.json").read_text("utf-8"))["mcpServers"]
    assert servers == {
        "other": {"command": "x"},
        "bytely": {"command": "bytely", "args": ["mcp"]},
    }
    codex = (home / ".codex" / "config.toml").read_text("utf-8")
    assert codex.startswith('[mcp_servers.other]\ncommand = "o"\n\n')
    assert codex.endswith(
        '[mcp_servers.bytely]\ncommand = "bytely"\nargs = ["mcp"]\n'
    )
    assert (repo / ".claude" / "skills" / "bytely" / "SKILL.md").is_file()
    assert (
        (repo / ".cursor" / "rules" / "bytely.mdc")
        .read_text("utf-8")
        .startswith("---\ndescription:")
    )

    again = run_init(repo, home, all_hosts=True)
    assert {change.action for change in again.changes} <= {
        "unchanged",
        "skipped-unparseable",
    }


def test_uninstall_restores_every_file(tmp_path: Path) -> None:
    repo, home = _setup(tmp_path)
    shutil.copytree(repo, tmp_path / "repo_before")
    shutil.copytree(home, tmp_path / "home_before")

    run_init(repo, home, all_hosts=True)
    run_uninstall(repo, home)

    assert _differences(repo, tmp_path / "repo_before") == []
    assert _differences(home, tmp_path / "home_before") == []
    # An empty folder outside the repo is not bytely's to prune.
    assert (home / ".gemini" / "config").is_dir()


def test_dry_run_and_scope_flags(tmp_path: Path) -> None:
    repo, home = _setup(tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))

    preview = run_init(repo, home, all_hosts=True, apply=False)
    assert any(change.action == "created" for change in preview.changes)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before

    report = run_init(
        repo, home, agents=["agents", "cursor"], mcp=False, global_scope=False
    )
    touched = {change.path for change in report.changes}
    assert repo / ".cursor" / "rules" / "bytely.mdc" in touched
    assert repo / ".cursor" / "mcp.json" not in touched  # --no-mcp
    assert home / ".codex" / "config.toml" not in touched  # --no-global
    assert "bytely" not in (home / ".codex" / "config.toml").read_text("utf-8")


def test_section_upsert_keeps_crlf_and_replaces_in_place(
    tmp_path: Path,
) -> None:
    path = tmp_path / "GEMINI.md"
    path.write_bytes(b"intro\r\n")

    assert files.upsert_section(path, "first", True) == "updated"
    assert files.upsert_section(path, "second", True) == "updated"
    assert files.upsert_section(path, "second", True) == "unchanged"
    text = path.read_bytes().decode()
    assert "\n" not in text.replace("\r\n", "")  # every line ending is CRLF
    assert text.count(files.START) == 1
    assert "first" not in text

    assert files.strip_section(path, True) == "removed"
    assert path.read_bytes() == b"intro\r\n"


def test_cli_init_and_uninstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, home = _setup(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    runner = CliRunner()

    dry = runner.invoke(
        main, ["init", str(repo), "--agents", "cursor", "--dry-run"]
    )
    assert dry.exit_code == 0, dry.output
    assert "would be created: .cursor/rules/bytely.mdc" in dry.output
    assert not (repo / ".cursor" / "rules").exists()

    done = runner.invoke(main, ["init", str(repo), "--agents", "cursor"])
    assert done.exit_code == 0, done.output
    assert "Wiring: cursor" in done.output
    assert (repo / ".cursor" / "rules" / "bytely.mdc").is_file()

    unknown = runner.invoke(main, ["init", str(repo), "--agents", "vim"])
    assert unknown.exit_code != 0
    assert "Unknown host(s): vim" in unknown.output

    listed = runner.invoke(main, ["init", "--list-agents"])
    assert "cursor       Cursor" in listed.output

    preview = runner.invoke(main, ["uninstall", str(repo)])
    assert preview.exit_code == 0, preview.output
    assert "would be deleted: .cursor/rules/bytely.mdc" in preview.output
    assert (repo / ".cursor" / "rules" / "bytely.mdc").is_file()

    removed = runner.invoke(main, ["uninstall", str(repo), "--yes"])
    assert removed.exit_code == 0, removed.output
    assert "deleted: .cursor/rules/bytely.mdc" in removed.output
    assert not (repo / "bytely").exists()
