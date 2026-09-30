"""`bytely statusline`: Claude Code's status line for this project."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bytely.graph.write import read_graph
from bytely.hooks import state
from bytely.hooks.format import render_statusline, render_subagent
from bytely.hooks.handlers import project_dir

if TYPE_CHECKING:
    from pathlib import Path


def resolve_stats(project: Path) -> dict[str, Any] | None:
    """Cached stats; the first time, counted from the graph and cached.

    The status line redraws often, so the graph file is read once, not on
    every redraw.
    """
    cached = state.read_stats(project)
    if cached and cached.get("syncing") and not state.sync_running(project):
        # The sync died without clearing its flag (killed, machine slept).
        cached = {**cached, "syncing": False}
    if cached and cached.get("nodeCount"):
        return cached
    graph = read_graph(str(state.context_dir(project)))
    if graph is None:
        return None
    counts = {"nodeCount": len(graph.nodes), "edgeCount": len(graph.edges)}
    try:
        return state.patch_stats(project, **counts)
    except OSError:
        return {**state.empty_stats(), **(cached or {}), **counts}


def render(payload: dict[str, Any]) -> str:
    """The status line text for a host's status-line payload."""
    project = project_dir(payload)
    session_id = payload.get("session_id")
    session = state.read_session(
        project, session_id if isinstance(session_id, str) else "default"
    )
    agent = payload.get("agent")
    if isinstance(agent, dict) and isinstance(agent.get("name"), str):
        return render_subagent(agent["name"], session)
    window = payload.get("context_window")
    used = window.get("used_percentage") if isinstance(window, dict) else None
    percent = round(used) if isinstance(used, (int, float)) else None
    return "\n".join(
        render_statusline(resolve_stats(project), session, percent)
    )
