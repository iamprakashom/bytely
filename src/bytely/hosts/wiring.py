"""`bytely init` and `bytely uninstall`: wire bytely into hosts, or undo it.

Selection: explicit `--agents` ids, else `--all`, else the hosts detected on
this machine or in this repository. `uninstall` walks every file `init`
could have written, for every host, so it cleans up whatever an earlier run
left, and `uninstall` + `init` converges on exactly the current wiring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bytely.graph.write import graph_path
from bytely.hosts.registry import (
    HOST_IDS,
    HOSTS,
    Probe,
    Target,
    ignore_target,
    on_path,
    targets_for,
)

if TYPE_CHECKING:
    from pathlib import Path

    from bytely.hosts import files


@dataclass(frozen=True)
class Change:
    """One file bytely touched (or would touch, in a dry run)."""

    host: str
    path: Path
    what: str
    scope: str
    action: files.Action


@dataclass
class InitReport:
    """What `init` selected and did."""

    selected: list[str]
    unknown: list[str]
    changes: list[Change] = field(default_factory=list)
    built: bool = False
    on_path: bool = True


def select_hosts(
    probe: Probe,
    agents: list[str] | None,
    all_hosts: bool,
    *,
    others: bool = True,
) -> tuple[list[str], list[str]]:
    """The host ids to wire, and any requested ids that do not exist.

    Explicit ids win. Otherwise Claude Code is always wired (as the
    reference implementation does) and the other hosts are every known one
    (`all_hosts`), none (`others=False`), or the ones detected here.
    """
    if agents is not None:
        known = [agent for agent in agents if agent in HOST_IDS]
        unknown = [agent for agent in agents if agent not in HOST_IDS]
        return list(dict.fromkeys(known)), unknown
    if all_hosts:
        return list(HOST_IDS), []
    if not others:
        return ["claude"], []
    detected = [host.id for host in HOSTS if host.detect(probe)]
    return ["claude", *(host for host in detected if host != "claude")], []


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value not in ("", "0", "false")


def _unique(targets: list[Target]) -> list[Target]:
    # Several hosts share AGENTS.md; each file edit happens once.
    seen: set[tuple[Path, str]] = set()
    out = []
    for target in targets:
        key = (target.path, target.what)
        if key not in seen:
            seen.add(key)
            out.append(target)
    return out


def run_init(
    repo: Path,
    home: Path,
    *,
    agents: list[str] | None = None,
    all_hosts: bool = False,
    others: bool = True,
    mcp: bool = True,
    hooks: bool = True,
    statusline: bool = True,
    global_scope: bool = True,
    build: bool = True,
    apply: bool = True,
) -> InitReport:
    """Write each selected host's instructions, MCP server, and hooks.

    `hooks=False` skips hooks and the status line; `statusline=False`
    (or `BYTELY_NO_STATUSLINE`) skips only the status line.
    """
    statusline = statusline and hooks and not _env_flag("BYTELY_NO_STATUSLINE")
    skipped_kinds = {
        kind
        for kind, wanted in (
            ("mcp", mcp),
            ("hook", hooks),
            ("statusline", statusline),
        )
        if not wanted
    }
    selected, unknown = select_hosts(
        Probe(repo, home), agents, all_hosts, others=others
    )
    report = InitReport(selected, unknown, on_path=on_path())
    targets = _unique(
        [
            target
            for host_id in selected
            for target in targets_for(host_id, repo, home)
            if target.kind not in skipped_kinds
            and (global_scope or target.scope != "global")
        ]
        + [ignore_target(repo)]
    )
    report.changes = [
        Change(
            target.host,
            target.path,
            target.what,
            target.scope,
            target.write(apply),
        )
        for target in targets
    ]
    if build and apply and not graph_path(repo / "bytely").is_file():
        from bytely.graph.build import build_graph

        build_graph(str(repo))
        report.built = True
    return report


def run_uninstall(
    repo: Path,
    home: Path,
    *,
    global_scope: bool = True,
    keep_graph: bool = False,
    apply: bool = True,
) -> list[Change]:
    """Undo every edit `init` could have made; report each file considered.

    `keep_graph` keeps the `bytely/` folder and its `.gitignore` entry.
    """
    targets = _unique(
        [
            target
            for host_id in HOST_IDS
            for target in targets_for(host_id, repo, home, every=True)
            if global_scope or target.scope != "global"
        ]
        + ([] if keep_graph else [ignore_target(repo)])
    )
    changes = [
        Change(
            target.host,
            target.path,
            target.what,
            target.scope,
            target.remove(apply),
        )
        for target in targets
    ]
    if not keep_graph:
        graph_dir = repo / "bytely"
        action: files.Action = "absent"
        if graph_path(graph_dir).is_file():
            action = "deleted"
            if apply:
                import shutil

                shutil.rmtree(graph_dir)
        changes.append(
            Change("graph", graph_dir, "local graph and cards", "repo", action)
        )
    return changes
