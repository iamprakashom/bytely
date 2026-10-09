"""Every file `bytely init` can write, per host, and how to undo it.

Adding a host means adding entries here; `init`, `uninstall`, and
`--dry-run` all derive from the same list, so they cannot drift apart.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from bytely.hosts import files, hookconfig
from bytely.hosts.instructions import (
    cursor_rule,
    instruction_body,
    kiro_steering,
    plain_rule,
    skill,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

Scope = Literal["repo", "global"]
# What an edit is for; `init --no-mcp / --no-hooks / --no-statusline`
# filter on it.
Kind = Literal["instructions", "mcp", "hook", "statusline", "graph"]


@dataclass(frozen=True)
class Probe:
    """Where to look when detecting hosts."""

    repo: Path
    home: Path

    def has(self, *paths: Path) -> bool:
        """Whether any of the directories exists."""
        return any(path.is_dir() for path in paths)


@dataclass(frozen=True)
class Host:
    """One AI coding tool bytely can wire into."""

    id: str
    name: str
    detect: Callable[[Probe], bool]


@dataclass(frozen=True)
class Target:
    """One file edit: what it is, and how to apply or undo it."""

    host: str
    path: Path
    what: str
    scope: Scope
    write: Callable[[bool], files.Action]
    remove: Callable[[bool], files.Action]
    kind: Kind = "instructions"

    @property
    def is_mcp(self) -> bool:
        """Whether this edit registers the MCP server."""
        return self.kind == "mcp"


HOSTS: tuple[Host, ...] = (
    Host(
        "claude",
        "Claude Code",
        lambda p: p.has(p.home / ".claude", p.repo / ".claude"),
    ),
    Host(
        "agents",
        "AGENTS.md hosts (Codex, opencode, and editors that read AGENTS.md)",
        lambda p: p.has(
            p.home / ".codex",
            p.home / ".config" / "opencode",
            p.home / ".config" / "agents",
        ),
    ),
    Host("adal", "AdaL", lambda p: p.has(p.home / ".adal", p.repo / ".adal")),
    Host(
        "cursor",
        "Cursor",
        lambda p: p.has(p.home / ".cursor", p.repo / ".cursor"),
    ),
    Host("gemini", "Gemini CLI", lambda p: p.has(p.home / ".gemini")),
    Host(
        "grok",
        "Grok (xAI)",
        lambda p: p.has(p.home / ".grok", p.repo / ".grok"),
    ),
    Host(
        "hermes",
        "Hermes Agent",
        lambda p: p.has(
            p.home / ".hermes",
            p.home / "AppData" / "Local" / "hermes",
            p.repo / ".hermes",
        ),
    ),
    Host(
        "antigravity",
        "Google Antigravity",
        # Its own markers, not the bare ~/.gemini that Gemini CLI also uses.
        lambda p: p.has(
            p.home / ".gemini" / "config",
            p.home / ".gemini" / "antigravity-cli",
            p.repo / ".agents",
        ),
    ),
    Host(
        "omnirush",
        "OmniRush",
        lambda p: p.has(p.home / ".omnirush", p.repo / ".omnirush"),
    ),
    Host("copilot", "GitHub Copilot", lambda p: p.has(p.repo / ".github")),
    Host("kiro", "Kiro", lambda p: p.has(p.home / ".kiro", p.repo / ".kiro")),
    Host(
        "windsurf",
        "Windsurf",
        lambda p: p.has(p.home / ".codeium" / "windsurf", p.repo / ".windsurf"),
    ),
)
HOST_IDS = tuple(host.id for host in HOSTS)


def server_launch() -> tuple[str, list[str]]:
    """How hosts start the MCP server.

    A bare command name, never an absolute path: these configs are committed
    and shared, and a path from one machine breaks on every other.
    """
    return "bytely", ["mcp"]


def on_path() -> bool:
    """Whether hosts will be able to start `bytely` by name."""
    return shutil.which("bytely") is not None


def _prune(path: Path, scope: Scope, home: Path | None) -> bool | Path:
    """How far removing `path` may prune the empty directories it leaves.

    In the repository, freely. Elsewhere, only below the tool's own folder
    in the home directory (`~/.gemini` for
    `~/.gemini/skills/bytely/SKILL.md`), never that folder itself; a file
    directly in the home directory prunes nothing.
    """
    if scope == "repo":
        return True
    if home is None:
        return False
    try:
        parts = path.relative_to(home).parts
    except ValueError:
        return False
    if len(parts) > 1 and parts[0].startswith("."):
        return home / parts[0]
    return False


def _owned(
    host: str,
    path: Path,
    what: str,
    content: str,
    scope: Scope = "repo",
    home: Path | None = None,
) -> Target:
    return Target(
        host,
        path,
        what,
        scope,
        lambda apply: files.write_owned(path, content, apply),
        lambda apply: files.remove_file(
            path, apply, prune=_prune(path, scope, home)
        ),
    )


def _hooks(
    host: str,
    path: Path,
    what: str,
    blocks: dict[str, list[dict[str, Any]]],
    scope: Scope = "repo",
    version: int | None = None,
    home: Path | None = None,
) -> Target:
    return Target(
        host,
        path,
        what,
        scope,
        lambda apply: hookconfig.install_hooks(
            path, blocks, apply, version=version
        ),
        lambda apply: hookconfig.remove_hooks(
            path,
            apply,
            prune=_prune(path, scope, home),
            version=version is not None,
        ),
        "hook",
    )


def _claude_targets(repo: Path, home: Path) -> list[Target]:
    settings = repo / ".claude" / "settings.json"
    return [
        _hooks(
            "claude",
            settings,
            "hooks (SessionStart / UserPromptSubmit / PostToolUse / Stop)",
            hookconfig.claude_blocks(),
        ),
        Target(
            "claude",
            settings,
            "bytely permission and footer links",
            "repo",
            lambda apply: hookconfig.install_claude_extras(settings, apply),
            lambda apply: hookconfig.remove_claude_extras(settings, apply),
            "hook",
        ),
        Target(
            "claude",
            settings,
            "statusLine",
            "repo",
            lambda apply: hookconfig.install_statusline(settings, apply),
            lambda apply: hookconfig.remove_statusline(settings, apply),
            "statusline",
        ),
        _hooks(
            "claude",
            home / ".claude" / "settings.json",
            "user-level hooks (every project with a graph)",
            hookconfig.claude_blocks(user_level=True),
            "global",
            home=home,
        ),
    ]


def _section(host: str, path: Path) -> Target:
    body = instruction_body()
    return Target(
        host,
        path,
        "bytely section",
        "repo",
        lambda apply: files.upsert_section(path, body, apply),
        lambda apply: files.strip_section(path, apply),
    )


def _json_server(
    host: str,
    path: Path,
    top_key: str,
    entry: dict[str, Any],
    scope: Scope = "repo",
    home: Path | None = None,
) -> Target:
    return Target(
        host,
        path,
        f"{top_key}.{files.SERVER_KEY}",
        scope,
        lambda apply: files.merge_json_key(path, top_key, entry, apply),
        lambda apply: files.remove_json_key(
            path, top_key, apply, prune=_prune(path, scope, home)
        ),
        "mcp",
    )


def _toml_server(
    host: str, path: Path, scope: Scope, home: Path | None = None
) -> Target:
    command, args = server_launch()
    return Target(
        host,
        path,
        files.TOML_HEADER,
        scope,
        lambda apply: files.upsert_toml_server(path, command, args, apply),
        lambda apply: files.remove_toml_server(
            path, apply, prune=_prune(path, scope, home)
        ),
        "mcp",
    )


def targets_for(
    host_id: str, repo: Path, home: Path, *, every: bool = False
) -> list[Target]:
    """Every edit wiring `host_id` makes, instructions first, then MCP.

    Some edits apply only when a tool is installed (Codex's and opencode's
    configs); `every=True` includes them anyway, which is what uninstalling
    needs.
    """
    command, args = server_launch()
    entry = {"command": command, "args": args}
    if host_id == "claude":
        return [
            _owned(
                "claude",
                repo / ".claude" / "skills" / "bytely" / "SKILL.md",
                "bytely skill",
                skill(),
            ),
            _json_server("claude", repo / ".mcp.json", "mcpServers", entry),
            *_claude_targets(repo, home),
            # User-level registration, so the server is there in every
            # project; it offers no tools where there is no graph.
            _json_server(
                "claude",
                home / ".claude.json",
                "mcpServers",
                entry,
                "global",
                home,
            ),
        ]
    if host_id == "agents":
        wired = [_section("agents", repo / "AGENTS.md")]
        if every or (home / ".codex").is_dir():
            wired.append(
                _toml_server(
                    "agents", home / ".codex" / "config.toml", "global", home
                )
            )
            wired.append(
                _hooks(
                    "agents",
                    home / ".codex" / "hooks.json",
                    "Codex hooks (SessionStart / UserPromptSubmit / "
                    "PostToolUse / Stop)",
                    hookconfig.codex_blocks(),
                    "global",
                    home=home,
                )
            )
        if every or (home / ".config" / "opencode").is_dir():
            wired.append(
                _json_server(
                    "agents",
                    repo / "opencode.json",
                    "mcp",
                    {
                        "type": "local",
                        "command": [command, *args],
                        "enabled": True,
                    },
                )
            )
        return wired
    if host_id == "adal":
        return [
            _owned(
                "adal",
                repo / ".adal" / "skills" / "bytely" / "SKILL.md",
                "bytely skill",
                skill(),
            )
        ]
    if host_id == "cursor":
        return [
            _owned(
                "cursor",
                repo / ".cursor" / "rules" / "bytely.mdc",
                "bytely rule",
                cursor_rule(),
            ),
            _json_server(
                "cursor", repo / ".cursor" / "mcp.json", "mcpServers", entry
            ),
            _hooks(
                "cursor",
                repo / ".cursor" / "hooks.json",
                "hooks (postToolUse / afterMCPExecution / sessionEnd)",
                hookconfig.cursor_blocks(),
                version=1,
            ),
        ]
    if host_id == "gemini":
        return [
            _section("gemini", repo / "GEMINI.md"),
            _json_server(
                "gemini",
                repo / ".gemini" / "settings.json",
                "mcpServers",
                entry,
            ),
        ]
    if host_id == "grok":
        return [
            _owned(
                "grok",
                repo / ".grok" / "skills" / "bytely" / "SKILL.md",
                "bytely skill",
                skill(),
            ),
            _toml_server("grok", repo / ".grok" / "config.toml", "repo"),
        ]
    if host_id == "hermes":
        return [_section("hermes", repo / "AGENTS.md")]
    if host_id == "antigravity":
        return [
            _section("antigravity", repo / "AGENTS.md"),
            _owned(
                "antigravity",
                home / ".gemini" / "skills" / "bytely" / "SKILL.md",
                "bytely skill (user level)",
                skill(),
                "global",
                home,
            ),
            # Antigravity reads MCP servers from its own global registry,
            # not Gemini CLI's per-repo settings. No `home`: its empty
            # `config/` folder is how Antigravity is detected; keep it.
            _json_server(
                "antigravity",
                home / ".gemini" / "config" / "mcp_config.json",
                "mcpServers",
                entry,
                "global",
            ),
        ]
    if host_id == "omnirush":
        # OmniRush reads the project's AGENTS.md, and MCP servers only from
        # its own state folder (`~/.omnirush/mcp.json`), never the repo's
        # `.mcp.json`. That registry is written only where OmniRush is
        # installed, like Codex's.
        wired = [_section("omnirush", repo / "AGENTS.md")]
        if every or (home / ".omnirush").is_dir():
            wired.append(
                _json_server(
                    "omnirush",
                    home / ".omnirush" / "mcp.json",
                    "mcpServers",
                    entry,
                    "global",
                    home,
                )
            )
        return wired
    if host_id == "copilot":
        return [
            _section("copilot", repo / ".github" / "copilot-instructions.md")
        ]
    if host_id == "kiro":
        return [
            _owned(
                "kiro",
                repo / ".kiro" / "steering" / "bytely.md",
                "bytely steering",
                kiro_steering(),
            ),
            _json_server(
                "kiro",
                repo / ".kiro" / "settings" / "mcp.json",
                "mcpServers",
                entry,
            ),
        ]
    if host_id == "windsurf":
        return [
            _owned(
                "windsurf",
                repo / ".windsurf" / "rules" / "bytely.md",
                "bytely rule",
                plain_rule(),
            )
        ]
    raise KeyError(host_id)


def ignore_target(repo: Path) -> Target:
    """The `.gitignore` entry for the `bytely/` output folder."""
    path = repo / ".gitignore"
    return Target(
        "graph",
        path,
        "/bytely/ ignore entry",
        "repo",
        lambda apply: files.add_ignore_entry(path, apply),
        lambda apply: files.remove_ignore_entry(path, apply),
        "graph",
    )
