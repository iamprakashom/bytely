"""Hooks, the status line, and token-savings accounting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from click.testing import CliRunner

from bytely.cli import main
from bytely.graph.build import build_graph
from bytely.hooks import state
from bytely.hooks.handlers import run_hook, sync
from bytely.hooks.metrics import (
    classify_tool_use,
    command_invokes_bytely,
    format_session_stats,
)
from bytely.hooks.savings import Baseline, savings_line, sum_savings
from bytely.hooks.statusline import render
from bytely.hooks.transcript import (
    has_savings_tally,
    last_assistant_turn,
    last_turn_billing,
)
from bytely.hosts.wiring import run_init, run_uninstall

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

BILLING = '''
def compute_invoice_total(lines):
    """Sum every invoice line, then apply tax."""
    subtotal = 0
    for line in lines:
        subtotal += line.price * line.quantity
    return apply_invoice_tax(subtotal)


def apply_invoice_tax(amount):
    return amount * 1.2
''' + "\n".join(f"# padding line {i} " + "x" * 60 for i in range(80))

CHECKOUT = """
from billing import compute_invoice_total


def checkout(cart):
    return compute_invoice_total(cart.lines)
"""


def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "billing.py").write_text(BILLING, encoding="utf-8")
    (repo / "checkout.py").write_text(CHECKOUT, encoding="utf-8")
    build_graph(str(repo))
    monkeypatch.delenv("BYTELY_DIR", raising=False)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
    return repo


def _context(out: str | None) -> str:
    assert out, "expected hook output"
    return str(json.loads(out)["hookSpecificOutput"]["additionalContext"])


def test_savings_line_round_trips_and_skips_tiny_outputs() -> None:
    line = savings_line("x" * 400, Baseline(files=2, chars=40_000))
    assert line.startswith("[bytely] tokens saved ≈ 9,900 (99%)")
    assert "2 file(s)" in line
    # The nudge's example is never counted as a second saving.
    assert sum_savings(line + "\n" + line) == 19_800
    assert savings_line("x" * 400, Baseline(files=1, chars=100)) == ""
    assert "$" in savings_line("x", Baseline(1, 40_000), rate=3.0)


def test_query_outputs_open_with_their_savings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    runner = CliRunner()
    out = runner.invoke(
        main, ["callers", "compute_invoice_total", "--root", str(repo)]
    )
    assert out.exit_code == 0, out.output
    assert out.output.startswith("[bytely] tokens saved ≈")
    assert "checkout" in out.output


def test_session_start_orients_and_user_level_defers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    context = _context(run_hook("session-start", {}))
    assert context.startswith("[bytely] This repo is indexed by bytely.")
    assert "repo map (bytely/INDEX.md)" in context

    settings = repo / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(
        '{"hooks": {"Stop": [{"hooks": [{"command": "bytely hook stop"}]}]}}',
        encoding="utf-8",
    )
    assert run_hook("session-start", {}, user_level=True) is None
    assert run_hook("session-start", {}) is not None


def test_hooks_do_nothing_without_a_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    assert run_hook("session-start", {}) is None
    assert run_hook("prompt", {"prompt": "how is the invoice total"}) is None
    assert not (tmp_path / "bytely").exists()


def test_post_edit_marks_stale_and_shows_dependents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    edit = {"tool_input": {"file_path": str(repo / "billing.py")}}
    context = _context(run_hook("post-edit", edit))
    assert context.startswith("[bytely] blast radius for billing.py")
    assert "checkout (checkout.py)" in context
    stats = state.read_stats(repo) or {}
    assert stats["dirty"]
    assert stats["staleFiles"] == ["billing.py"]
    assert stats["lastFile"] == "billing.py"

    # A Codex patch names the file in its header.
    patch = {
        "tool_input": {
            "command": "*** Begin Patch\n*** Update File: checkout.py\n@@\n"
        }
    }
    run_hook("post-edit", patch)
    assert (state.read_stats(repo) or {})["staleCount"] == 2
    # Edits to the graph's own files are not source edits.
    inside = {"tool_input": {"file_path": str(repo / "bytely" / "INDEX.md")}}
    assert run_hook("post-edit", inside) is None
    assert (state.read_stats(repo) or {})["staleCount"] == 2

    (repo / "checkout.py").write_text(
        CHECKOUT + "\n\ndef refund(cart):\n    return 0\n", encoding="utf-8"
    )
    token = state.acquire_lock(repo)
    assert token
    sync(repo, token)
    stats = state.read_stats(repo) or {}
    assert not stats["dirty"]
    assert stats["staleCount"] == 0
    assert stats["nodeCount"] > 0
    assert not stats["syncing"]
    again = state.acquire_lock(repo)  # the sync released it
    assert again
    state.release_lock(repo, again)


def test_prompt_pointers_are_relevant_and_novel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    payload = {"prompt": "fix the invoice total tax", "session_id": "s1"}
    context = _context(run_hook("prompt", payload))
    assert context.startswith("[bytely] starting points for this task")
    assert "billing.py:L" in context
    # The same pointers are not injected twice in one session...
    assert run_hook("prompt", payload) is None
    # ...but a new session gets them.
    assert run_hook("prompt", {**payload, "session_id": "s2"}) is not None
    assert run_hook("prompt", {"prompt": "hi", "session_id": "s3"}) is None
    session = state.read_session(repo, "s1")
    assert session["lastQuery"] == "fix the invoice total tax"


def test_weak_prompt_nudges_at_most_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repo(tmp_path, monkeypatch)
    nudges = [
        run_hook(
            "prompt",
            {"prompt": "line quantum flux capacitor", "session_id": "w"},
        )
        for i in range(4)
    ]
    shown = [n for n in nudges if n]
    # One weak term of four: nudged, but only twice a session.
    assert len(shown) == 2
    assert all("no strong match" in _context(n) for n in shown)


def test_tool_use_accounting_and_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = [
        {
            "tool_name": "mcp__bytely__bytely_find_code",
            "tool_response": [
                {"type": "text", "text": "[bytely] tokens saved ≈ 1,200 (90%)"}
            ],
        },
        {
            "tool_name": "Bash",
            "tool_input": {"command": "uv run bytely grep foo"},
            "tool_response": {"stdout": "[bytely] tokens saved ≈ 300 (50%)"},
        },
        {"tool_name": "Read", "tool_input": {"file_path": "billing.py"}},
        {"tool_name": "Bash", "tool_input": {"command": "ls"}},
    ]
    for call in calls:
        run_hook("tool-savings", {**call, "session_id": "t"})
    session = state.read_session(repo, "t")
    assert session["bytelyReads"] == 2
    assert session["sourceReads"] == 1
    assert session["savedTokens"] == 1_500

    report = format_session_stats(state.latest_session(repo))
    assert "session t" in report
    assert "~1,500" in report
    assert "67% bytely" in report
    out = CliRunner().invoke(main, ["stats", str(repo)])
    assert "tokens saved:  ~1,500" in out.output


def test_cursor_events_are_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    run_hook(
        "cursor-mcp",
        {
            "conversation_id": "c",
            "tool_name": "bytely_file_api",
            "result_json": {"text": "[bytely] tokens saved ≈ 42 (60%)"},
        },
    )
    run_hook("cursor-post-tool", {"conversation_id": "c", "tool_name": "Grep"})
    run_hook("cursor-session-end", {"conversation_id": "c"})
    session = state.read_session(repo, "c")
    assert session["savedTokens"] == 42
    assert session["sourceReads"] == 1
    assert session["endedAt"]


def _transcript(path: Path, reply: str) -> None:
    usage = {
        "input_tokens": 1000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 10_000,
    }
    lines = [
        {"type": "user", "uuid": "u1", "message": {"content": "do it"}},
        # One API response, written as two entries repeating its usage.
        {
            "type": "assistant",
            "uuid": "a1",
            "message": {
                "id": "m1",
                "model": "claude-sonnet-4-6",
                "usage": usage,
                "content": [{"type": "tool_use"}],
            },
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "message": {
                "id": "m1",
                "model": "claude-sonnet-4-6",
                "usage": usage,
                "content": [{"type": "text", "text": reply}],
            },
        },
        {
            "type": "assistant",
            "uuid": "s1",
            "isSidechain": True,
            "message": {
                "id": "m9",
                "model": "claude-sonnet-4-6",
                "usage": usage,
                "content": [{"type": "text", "text": "subagent"}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def test_transcript_billing_and_tally(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    _transcript(transcript, "Done. ⚡ bytely saved ~12,400 tokens this turn")
    billing = last_turn_billing(str(transcript))
    assert billing is not None
    assert billing.tokens == 11_000  # counted once, sidechain excluded
    assert billing.cost_micros == 6_000  # (1000 + 10000 * 0.1) * $3/Mtok
    turn = last_assistant_turn(str(transcript))
    assert turn is not None
    assert has_savings_tally(turn.text)
    assert last_turn_billing(str(tmp_path / "missing.jsonl")) is None
    assert last_turn_billing(None) is None


def test_stop_prices_the_turn_once_and_counts_the_tally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    transcript = tmp_path / "t.jsonl"
    _transcript(transcript, "Done. ⚡ bytely saved ~300 tokens this turn")
    run_hook(
        "tool-savings",
        {
            "session_id": "p",
            "tool_name": "mcp__bytely__bytely_repo_map",
            "tool_response": "[bytely] tokens saved ≈ 300 (40%)",
        },
    )
    stop = {"session_id": "p", "transcript_path": str(transcript)}
    run_hook("stop", stop)
    run_hook("stop", stop)  # the same turn is not billed twice
    session = state.read_session(repo, "p")
    assert session["inputCostMicros"] == 6_000
    assert session["inputTokensBilled"] == 11_000
    assert session["bytelyTurns"] == 1
    assert session["reportedTurns"] == 1
    assert "value saved:   ~<$0.01" in format_session_stats(
        {"id": "p", **session}
    )


def test_statusline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo(tmp_path, monkeypatch)
    state.write_session(
        repo, "x", {**state.empty_session(), "savedTokens": 2_500}
    )
    text = render(
        {"session_id": "x", "context_window": {"used_percentage": 41.6}}
    )
    assert "bytely" in text
    assert "nodes /" in text
    assert "✓ synced" in text
    assert "~2,500 tok saved" in text
    assert "ctx 42%" in text
    run_hook("post-edit", {"tool_input": {"file_path": "billing.py"}})
    text = render({"session_id": "x"})
    assert "⚠ 1 stale" in text
    assert "last: " in text
    assert "billing.py" in render({"agent": {"name": "billing.py"}})

    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "nowhere"))
    assert "not built" in render({})


def test_command_classification() -> None:
    assert command_invokes_bytely("bytely ask 'x' --source")
    assert command_invokes_bytely("cd src && uv run bytely grep foo")
    assert command_invokes_bytely("python -m bytely map")
    assert not command_invokes_bytely("echo notbytely")
    assert classify_tool_use("mcp__bytely__bytely_trace_calls") == "bytely"
    assert classify_tool_use("bytely:bytely_file_api") == "bytely"
    assert classify_tool_use("Glob") == "source"
    assert classify_tool_use("Bash", "git status") is None


def _home_and_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo, home = tmp_path / "repo", tmp_path / "home"
    (repo / ".cursor").mkdir(parents=True)
    for folder in (".claude", ".codex", ".gemini/config"):
        (home / folder).mkdir(parents=True)
    (repo / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return repo, home


def test_init_wires_hooks_and_uninstall_keeps_user_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BYTELY_NO_STATUSLINE", raising=False)
    repo, home = _home_and_repo(tmp_path)
    user_settings = {
        "model": "opus",
        "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "say"}]}]},
    }
    (repo / ".claude").mkdir()
    (repo / ".claude" / "settings.json").write_text(json.dumps(user_settings))
    (home / ".claude" / "settings.json").write_text(json.dumps(user_settings))

    run_init(repo, home, all_hosts=True, build=False)

    settings = json.loads((repo / ".claude" / "settings.json").read_text())
    assert settings["model"] == "opus"
    assert settings["statusLine"]["command"] == "bytely statusline"
    assert settings["hooks"]["Stop"][0]["hooks"][0]["command"] == "say"
    assert settings["hooks"]["Stop"][1]["hooks"][0] == {
        "type": "command",
        "command": "bytely hook stop",
        "timeout": 8,
    }
    assert "Bash(bytely:*)" in settings["permissions"]["allow"]
    global_settings = json.loads(
        (home / ".claude" / "settings.json").read_text()
    )
    commands = json.dumps(global_settings["hooks"])
    assert "bytely hook prompt --user-level" in commands
    claude_json = json.loads((home / ".claude.json").read_text())
    assert claude_json["mcpServers"]["bytely"]["args"] == ["mcp"]
    cursor = json.loads((repo / ".cursor" / "hooks.json").read_text())
    assert cursor["version"] == 1
    assert cursor["hooks"]["afterMCPExecution"] == [
        {"command": "bytely hook cursor-mcp"}
    ]
    codex = json.loads((home / ".codex" / "hooks.json").read_text())
    assert codex["hooks"]["SessionStart"][0]["matcher"] == (
        "startup|resume|compact"
    )
    assert (home / ".gemini" / "skills" / "bytely" / "SKILL.md").is_file()

    again = run_init(repo, home, all_hosts=True, build=False)
    assert {c.action for c in again.changes} == {"unchanged"}

    run_uninstall(repo, home)
    for settings_path in (
        repo / ".claude" / "settings.json",
        home / ".claude" / "settings.json",
    ):
        assert json.loads(settings_path.read_text()) == user_settings
    assert not (home / ".claude.json").exists()
    assert not (repo / ".cursor" / "hooks.json").exists()
    assert not (home / ".codex" / "hooks.json").exists()
    assert not (home / ".gemini" / "skills").exists()
    assert (home / ".gemini" / "config").is_dir()


def test_foreign_statusline_and_opt_outs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BYTELY_NO_STATUSLINE", raising=False)
    repo, home = _home_and_repo(tmp_path)
    (repo / ".claude").mkdir()
    mine = {"statusLine": {"type": "command", "command": "my-line"}}
    (repo / ".claude" / "settings.json").write_text(json.dumps(mine))

    report = run_init(repo, home, agents=["claude"], build=False)
    actions = {c.what: c.action for c in report.changes}
    assert actions["statusLine"] == "skipped-foreign"
    settings = json.loads((repo / ".claude" / "settings.json").read_text())
    assert settings["statusLine"]["command"] == "my-line"
    run_uninstall(repo, home)
    assert json.loads((repo / ".claude" / "settings.json").read_text()) == mine

    report = run_init(
        repo, home, agents=["claude", "cursor"], hooks=False, build=False
    )
    touched = {c.path for c in report.changes}
    assert repo / ".cursor" / "hooks.json" not in touched
    settings = json.loads((repo / ".claude" / "settings.json").read_text())
    assert "hooks" not in settings

    monkeypatch.setenv("BYTELY_NO_STATUSLINE", "1")
    (repo / ".claude" / "settings.json").unlink()
    run_init(repo, home, agents=["claude"], build=False)
    settings = json.loads((repo / ".claude" / "settings.json").read_text())
    assert "statusLine" not in settings
    assert "hooks" in settings


def test_tool_savings_matcher_covers_mcp_tools() -> None:
    import re

    from bytely.hosts.hookconfig import TOOL_SAVINGS_MATCHER

    # Claude Code tests a matcher with `.*` as an unanchored JS regex; for
    # this pattern Python's `re.search` behaves the same.
    matcher = re.compile(TOOL_SAVINGS_MATCHER)
    for tool in ("Bash", "Read", "Grep", "mcp__bytely__bytely_find_code"):
        assert matcher.search(tool), tool
    for tool in ("ReadMcpResourceTool", "mcp__other__read", "Edit"):
        assert not matcher.search(tool), tool


def test_concurrent_hooks_do_not_lose_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    import sys

    repo = _repo(tmp_path, monkeypatch)
    call = json.dumps(
        {
            "session_id": "par",
            "tool_name": "mcp__bytely__bytely_find_code",
            "tool_response": "[bytely] tokens saved ≈ 10 (50%)",
        }
    )
    workers = [
        subprocess.Popen(
            [sys.executable, "-m", "bytely", "hook", "tool-savings"],
            stdin=subprocess.PIPE,
            env={**__import__("os").environ, "CLAUDE_PROJECT_DIR": str(repo)},
        )
        for _ in range(12)
    ]
    for worker in workers:
        assert worker.stdin is not None
        worker.stdin.write(call.encode())
        worker.stdin.close()
    for worker in workers:
        assert worker.wait(timeout=60) == 0
    session = state.read_session(repo, "par")
    assert session["bytelyReads"] == 12
    assert session["savedTokens"] == 120


def test_sync_lock_is_released_only_by_its_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    first = state.acquire_lock(repo)
    assert first
    assert state.acquire_lock(repo) is None
    # The first lock went stale and a second sync took over.
    lock = state.cache_dir(repo) / state.LOCK_FILE
    old = lock.stat().st_mtime - state.LOCK_STALE_SECONDS - 1
    __import__("os").utime(lock, (old, old))
    second = state.acquire_lock(repo)
    assert second
    assert second != first
    # The first sync finishing must not free the second one's lock.
    state.release_lock(repo, first)
    assert not state.touch_lock(repo, first)
    assert state.lock_owner(repo) == second
    assert state.acquire_lock(repo) is None
    # A direct sync while another holds the lock does nothing.
    sync(repo)
    assert state.lock_owner(repo) == second
    state.release_lock(repo, second)
    assert state.lock_owner(repo) is None


def test_statusline_ignores_a_dead_syncs_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    # A sync set the flag, then was killed: no live lock behind it.
    state.patch_stats(repo, nodeCount=5, edgeCount=4, syncing=True)
    assert "syncing" not in render({})
    token = state.acquire_lock(repo)
    assert token
    assert "syncing" in render({})
    state.release_lock(repo, token)


def test_stale_lock_reclaim_keeps_a_live_owner(tmp_path: Path) -> None:
    import os

    from bytely.util import lock

    path = tmp_path / "x.lock"
    first = lock.try_acquire(path, stale=60)
    assert first
    # A claimant that judged the lock stale just before its owner
    # refreshed it must hand it back, not take it.
    real_alive = lock.alive
    calls = {"n": 0}

    def stale_once(p: Path, stale: float) -> bool:
        calls["n"] += 1
        return False if calls["n"] == 1 else real_alive(p, stale)

    lock.alive = stale_once  # type: ignore[assignment]
    try:
        assert lock.try_acquire(path, stale=60) is None
    finally:
        lock.alive = real_alive  # type: ignore[assignment]
    assert lock.owner(path) == first
    assert not list(tmp_path.glob("*.stale"))
    # A really dead lock is taken over.
    old = path.stat().st_mtime - 120
    os.utime(path, (old, old))
    second = lock.try_acquire(path, stale=60)
    assert second
    assert lock.owner(path) == second


def test_lock_is_reentrant_within_a_thread(tmp_path: Path) -> None:
    from bytely.util import lock

    path = tmp_path / "b.lock"
    with lock.held(path, stale=60, wait=1) as outer:
        assert outer
        with lock.held(path, stale=60, wait=0) as inner:
            assert inner == outer
        assert lock.owner(path) == outer  # the inner exit kept it
    assert lock.owner(path) is None


def test_concurrent_rebuilds_are_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import subprocess
    import sys

    repo = _repo(tmp_path, monkeypatch)
    for i in range(30):
        (repo / f"mod{i}.py").write_text(
            f"def f{i}():\n    return {i}\n", encoding="utf-8"
        )
    # Six processes find the tree changed at once; each refreshes.
    workers = [
        subprocess.Popen(
            [sys.executable, "-m", "bytely", "map", str(repo)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ},
        )
        for _ in range(6)
    ]
    for worker in workers:
        _, err = worker.communicate(timeout=120)
        assert worker.returncode == 0, err.decode()
    # One rebuilt; the others waited and reused it. The result is intact.
    from bytely.graph.check import check_graph

    fresh, message = check_graph(str(repo))
    assert fresh, message
    assert not (repo / "bytely" / "cache" / ".build.lock").exists()
