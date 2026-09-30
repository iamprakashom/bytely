"""Hook and status-line entries bytely adds to hosts' JSON settings.

An entry is bytely's when its command runs `bytely hook` or
`bytely statusline`; installing drops every earlier bytely entry before
adding the current set (so re-running converges), and removing drops
exactly those, leaving the user's own hooks in place.

Timeouts are in seconds, the unit Claude Code and Codex both read.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from bytely.hosts import files

if TYPE_CHECKING:
    from pathlib import Path

HOOK_COMMAND = "bytely hook"
STATUSLINE_COMMAND = "bytely statusline"
FOOTER_REGEX = r"bytely/[\w./-]+\.md"
TOOL_SAVINGS_MATCHER = "^(Bash|PowerShell|Read|Grep|Glob|mcp__bytely__.*)$"
ALLOW_ENTRIES = ("Bash(bytely:*)", "PowerShell(bytely:*)")


def _ours(entry: object) -> bool:
    text = json.dumps(entry)
    return f"{HOOK_COMMAND} " in text or STATUSLINE_COMMAND in text


def _command(event: str, user_level: bool) -> str:
    return f"{HOOK_COMMAND} {event}" + (" --user-level" if user_level else "")


def claude_blocks(user_level: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Claude Code's hook blocks, event by event."""

    def run(event: str, timeout: int) -> list[dict[str, Any]]:
        return [
            {
                "type": "command",
                "command": _command(event, user_level),
                "timeout": timeout,
            }
        ]

    return {
        "PostToolUse": [
            {"matcher": "Write|Edit|MultiEdit", "hooks": run("post-edit", 10)},
            {
                # A regex (it has `.*`), so anchored: Claude Code tests it
                # unanchored, and a matcher of plain names only would
                # compare `mcp__bytely__` as an exact tool name.
                "matcher": TOOL_SAVINGS_MATCHER,
                "hooks": run("tool-savings", 8),
            },
        ],
        "UserPromptSubmit": [{"hooks": run("prompt", 15)}],
        "SessionStart": [{"hooks": run("session-start", 8)}],
        "Stop": [{"hooks": run("stop", 8)}],
    }


def codex_blocks() -> dict[str, list[dict[str, Any]]]:
    """Codex's hook blocks (`~/.codex/hooks.json`)."""

    def run(event: str) -> list[dict[str, Any]]:
        timeout = 15 if event == "prompt" else 10
        return [
            {
                "type": "command",
                "command": _command(event, False),
                "timeout": timeout,
            }
        ]

    return {
        "SessionStart": [
            {"matcher": "startup|resume|compact", "hooks": run("session-start")}
        ],
        "UserPromptSubmit": [{"hooks": run("prompt")}],
        "PostToolUse": [
            {
                "matcher": "apply_patch|Write|Edit|MultiEdit",
                "hooks": run("post-edit"),
            }
        ],
        "Stop": [{"hooks": run("stop")}],
    }


def cursor_blocks() -> dict[str, list[dict[str, Any]]]:
    """Cursor's hook entries (`.cursor/hooks.json`, version 1)."""
    return {
        "postToolUse": [
            {
                "matcher": "Read|Grep|Glob|Search|Shell",
                "command": _command("cursor-post-tool", False),
            }
        ],
        "afterMCPExecution": [{"command": _command("cursor-mcp", False)}],
        "sessionEnd": [{"command": _command("cursor-session-end", False)}],
    }


def _merge_hooks(
    data: dict[str, Any], blocks: dict[str, list[dict[str, Any]]]
) -> files.Action | None:
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        return "skipped-unparseable"
    for event, entries in blocks.items():
        prior = hooks.get(event, [])
        if not isinstance(prior, list):
            return "skipped-unparseable"
        hooks[event] = [e for e in prior if not _ours(e)] + entries
    return None


def _strip_hooks(data: dict[str, Any]) -> None:
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return
    for event in list(hooks):
        entries = hooks[event]
        if not isinstance(entries, list):
            continue
        kept = [entry for entry in entries if not _ours(entry)]
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]
    if not hooks:
        del data["hooks"]


def install_hooks(
    path: Path,
    blocks: dict[str, list[dict[str, Any]]],
    apply: bool,
    *,
    version: int | None = None,
) -> files.Action:
    """Add bytely's hook entries to a settings file."""

    def change(data: dict[str, Any]) -> files.Action | None:
        if version is not None:
            data.setdefault("version", version)
        return _merge_hooks(data, blocks)

    return files.edit_json(path, change, apply)


def remove_hooks(
    path: Path,
    apply: bool,
    *,
    prune: bool | Path = True,
    version: bool = False,
) -> files.Action:
    """Remove bytely's hook entries from a settings file."""

    def change(data: dict[str, Any]) -> None:
        _strip_hooks(data)
        # A version key alone is what installing added; nothing else is left.
        if version and set(data) == {"version"}:
            data.clear()

    return files.edit_json(path, change, apply, removing=True, prune=prune)


def _without(values: object, drop: Any) -> list[Any]:
    return (
        [v for v in values if not drop(v)] if isinstance(values, list) else []
    )


def _is_our_allow(entry: object) -> bool:
    return str(entry) in ALLOW_ENTRIES


def install_claude_extras(path: Path, apply: bool) -> files.Action:
    """The footer link pattern and the permission to run `bytely`."""

    def change(data: dict[str, Any]) -> files.Action | None:
        permissions = data.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            return "skipped-unparseable"
        data["footerLinksRegexes"] = [
            *_without(
                data.get("footerLinksRegexes"), lambda r: str(r) == FOOTER_REGEX
            ),
            FOOTER_REGEX,
        ]
        permissions["allow"] = [
            *_without(permissions.get("allow"), _is_our_allow),
            *ALLOW_ENTRIES,
        ]
        return None

    return files.edit_json(path, change, apply)


def remove_claude_extras(path: Path, apply: bool) -> files.Action:
    """Undo `install_claude_extras`."""

    def change(data: dict[str, Any]) -> None:
        if isinstance(data.get("footerLinksRegexes"), list):
            kept = _without(
                data["footerLinksRegexes"], lambda r: str(r) == FOOTER_REGEX
            )
            if kept:
                data["footerLinksRegexes"] = kept
            else:
                del data["footerLinksRegexes"]
        permissions = data.get("permissions")
        if isinstance(permissions, dict) and isinstance(
            permissions.get("allow"), list
        ):
            kept = _without(permissions["allow"], _is_our_allow)
            if kept:
                permissions["allow"] = kept
            else:
                del permissions["allow"]
            if not permissions:
                del data["permissions"]

    return files.edit_json(path, change, apply, removing=True)


_STATUS_KEYS = ("statusLine", "subagentStatusLine")


def install_statusline(path: Path, apply: bool) -> files.Action:
    """Point Claude Code's status lines at bytely, unless one is set.

    A session has one status line; someone else's is never replaced.
    """

    def change(data: dict[str, Any]) -> files.Action | None:
        if any(data.get(key) and not _ours(data[key]) for key in _STATUS_KEYS):
            return "skipped-foreign"
        for key in _STATUS_KEYS:
            data[key] = {"type": "command", "command": STATUSLINE_COMMAND}
        return None

    return files.edit_json(path, change, apply)


def remove_statusline(path: Path, apply: bool) -> files.Action:
    """Remove bytely's status lines, leaving anyone else's."""

    def change(data: dict[str, Any]) -> None:
        for key in _STATUS_KEYS:
            if data.get(key) and _ours(data[key]):
                del data[key]

    return files.edit_json(path, change, apply, removing=True)
