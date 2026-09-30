"""`bytely hook <event>`: what each host hook does.

Hosts pipe a JSON payload to the hook on stdin and read an optional JSON
reply on stdout. Every handler is best-effort: a hook that fails must
never break the agent's turn, so the CLI swallows errors and exits 0.

Events (Claude Code and Codex share the first five):

- `session-start`: orient the agent (directive + the start of INDEX.md),
  flag a stale graph, and start a sync if one is due.
- `prompt`: inject pointers to the definitions the prompt is about, gated
  on relevance and novelty so a session is not spammed.
- `post-edit`: mark the graph stale and show who depends on the edited
  file, so the agent checks callers before moving on.
- `tool-savings`: count bytely reads, source reads, and tokens saved.
- `stop`: price the turn from the transcript, note whether the reply
  reported its savings, and sync the graph in the background if edits
  made it stale.
- `cursor-post-tool`, `cursor-mcp`, `cursor-session-end`: Cursor's
  equivalents of `tool-savings`.
- `sync`: the background sync itself (started by `stop`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

from bytely.graph.write import graph_path, read_graph
from bytely.hooks import state
from bytely.hooks.format import blast_radius, orientation, probe, retrieval
from bytely.hooks.metrics import (
    is_bytely_mcp_tool,
    is_mcp_tool_name,
    record_tool_use,
    score_tool_use,
)
from bytely.hooks.savings import sum_savings
from bytely.hooks.transcript import (
    has_savings_tally,
    last_assistant_turn,
    last_turn_billing,
)
from bytely.util import lock

if TYPE_CHECKING:
    from bytely.graph.types import GraphV1

MIN_PROMPT_CHARS = 12
SYNC_TOKEN_ENV = "BYTELY_SYNC_TOKEN"
LOCK_HEARTBEAT_SECONDS = 60.0
EVENTS = (
    "session-start",
    "prompt",
    "post-edit",
    "tool-savings",
    "stop",
    "cursor-post-tool",
    "cursor-mcp",
    "cursor-session-end",
    "sync",
)
HOOK_MARKER = "bytely hook "
_EVENT_NAMES = {
    "session-start": "SessionStart",
    "prompt": "UserPromptSubmit",
    "post-edit": "PostToolUse",
}


def project_dir(payload: dict[str, Any]) -> Path:
    """The repository a hook call is about.

    The host's project directory, else the payload's `cwd`, else ours;
    then the nearest ancestor that has a graph, so a session started in
    a subfolder still finds it.
    """
    start = (
        os.environ.get("CLAUDE_PROJECT_DIR")
        or (payload.get("cwd") if isinstance(payload.get("cwd"), str) else "")
        or os.getcwd()
    )
    path = Path(start).resolve()
    if os.environ.get("BYTELY_DIR"):
        return path
    for candidate in (path, *path.parents):
        if graph_path(candidate / "bytely").is_file():
            return candidate
    return path


def has_graph(project: Path) -> bool:
    """Whether the project has been indexed."""
    return graph_path(state.context_dir(project)).is_file()


def repo_hooks_installed(project: Path) -> bool:
    """Whether the project's own Claude settings run bytely's hooks.

    User-level hooks step aside then, so no event is handled twice.
    """
    for name in ("settings.json", "settings.local.json"):
        try:
            text = (project / ".claude" / name).read_text("utf-8")
        except OSError:
            continue
        if HOOK_MARKER in text:
            return True
    return False


def _emit(event: str, context: str) -> str:
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": _EVENT_NAMES[event],
                "additionalContext": context,
            }
        },
        ensure_ascii=False,
    )


def _session_id(payload: dict[str, Any], *keys: str) -> str:
    for key in (*keys, "session_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return "default"


def _tool_input(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("tool_input")
    return value if isinstance(value, dict) else {}


def _relative(project: Path, file: str) -> str:
    path = Path(file)
    if not path.is_absolute():
        path = project / path
    try:
        return path.resolve().relative_to(project).as_posix()
    except (OSError, ValueError):
        return file.replace("\\", "/")


def edited_file(payload: dict[str, Any]) -> str | None:
    """The file an edit tool wrote: a path field, or a Codex patch header."""
    tool_input = _tool_input(payload)
    for key in ("file_path", "path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value
    command = tool_input.get("command") or tool_input.get("patch")
    if isinstance(command, str):
        for line in command.splitlines():
            for marker in ("*** Update File:", "*** Add File:"):
                if line.startswith(marker):
                    return line[len(marker) :].strip()
    return None


def session_start(payload: dict[str, Any], project: Path) -> str | None:
    """Orientation, plus a banner and a sync when edits left it stale."""
    try:
        index = (state.context_dir(project) / "INDEX.md").read_text("utf-8")
    except OSError:
        return None
    stats = state.read_stats(project) or {}
    banner = None
    if stats.get("dirty"):
        stale = len(stats.get("staleFiles") or [])
        banner = (
            f"[bytely] {stale or 'some'} file(s) changed since the graph "
            "was last synced; it is re-syncing in the background, and the "
            "query tools refresh before they answer, so results are current."
        )
        start_sync(project)
    return _emit("session-start", orientation(index, banner=banner))


def _scope_hint(graph: GraphV1, last_path: object) -> str | None:
    """In a multi-project repo, the sub-project the last edit was in."""
    scopes = [scope for scope in graph.scopes if scope.prefix]
    if not scopes or not isinstance(last_path, str):
        return None
    matches = [s.prefix for s in scopes if last_path.startswith(s.prefix)]
    return max(matches, key=len) if matches else None


def prompt(payload: dict[str, Any], project: Path) -> str | None:
    """Pointers for the prompt, when the graph has something relevant."""
    text = str(payload.get("prompt") or "").strip()
    if len(text) < MIN_PROMPT_CHARS:
        return None
    graph = read_graph(str(state.context_dir(project)))
    if graph is None:
        return None
    stats = state.read_stats(project) or {}
    found = probe(graph, text, scope=_scope_hint(graph, stats.get("lastPath")))
    session_id = _session_id(payload)
    with state.update_session(project, session_id) as session:
        session["lastQuery"] = text[:200]
        agent = payload.get("agent")
        if isinstance(agent, dict) and isinstance(agent.get("name"), str):
            session.setdefault("perAgentQuery", {})[agent["name"]] = text[:200]
        context = retrieval(graph, found, session)
    return _emit("prompt", context) if context else None


def post_edit(payload: dict[str, Any], project: Path) -> str | None:
    """Mark the edited file stale and show who depends on it."""
    file = edited_file(payload)
    if not file:
        return None
    relative = _relative(project, file)
    context = state.context_dir(project)
    try:
        inside = context.resolve().relative_to(project).as_posix() + "/"
    except (OSError, ValueError):
        inside = "bytely/"
    if relative.startswith(inside):
        return None  # the graph's own files are not source
    with state.update_stats(project) as stats:
        stale = list(dict.fromkeys([*(stats["staleFiles"] or []), relative]))
        stats.update(
            dirty=True,
            staleFiles=stale,
            staleCount=len(stale),
            lastFile=PurePath(relative).name,
            lastPath=relative,
        )
    graph = read_graph(str(context))
    if graph is None:
        return None
    radius = blast_radius(graph, relative)
    return _emit("post-edit", radius) if radius else None


def tool_savings(payload: dict[str, Any], project: Path) -> None:
    """Count a Claude Code tool call."""
    kind, saved = score_tool_use(
        payload.get("tool_name"),
        _tool_input(payload).get("command"),
        payload.get("tool_response", payload),
    )
    record_tool_use(project, _session_id(payload), kind, saved, "claude-code")


def cursor_post_tool(payload: dict[str, Any], project: Path) -> None:
    """Count a Cursor built-in tool call (MCP calls come via cursor-mcp)."""
    tool = str(payload.get("tool_name") or "")
    if is_mcp_tool_name(tool) or is_bytely_mcp_tool(tool):
        return
    tool_input = _tool_input(payload)
    kind, saved = score_tool_use(
        tool,
        tool_input.get("command") or tool_input.get("cmd"),
        payload.get("tool_output", payload.get("tool_response", payload)),
    )
    record_tool_use(
        project, _session_id(payload, "conversation_id"), kind, saved, "cursor"
    )


def cursor_mcp(payload: dict[str, Any], project: Path) -> None:
    """Count a Cursor call to one of bytely's MCP tools."""
    if not is_bytely_mcp_tool(str(payload.get("tool_name") or "")):
        return
    result = payload.get("result_json", payload.get("result", payload))
    text = (
        result
        if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False)
    )
    record_tool_use(
        project,
        _session_id(payload, "conversation_id"),
        "bytely",
        sum_savings(text),
        "cursor",
    )


def cursor_session_end(payload: dict[str, Any], project: Path) -> None:
    """Close a Cursor session: stamp its end, keeping it for `stats`."""
    session_id = _session_id(payload, "conversation_id")
    with state.update_session(project, session_id) as session:
        session["endedAt"] = time.time()


def _sample_turn_cost(payload: dict[str, Any], project: Path) -> None:
    billing = last_turn_billing(payload.get("transcript_path"))
    if billing is None:
        return
    with state.update_session(project, _session_id(payload)) as session:
        if session.get("lastBillingUuid") == billing.uuid:
            return
        session["inputCostMicros"] = (
            int(session.get("inputCostMicros") or 0) + billing.cost_micros
        )
        session["inputTokensBilled"] = (
            int(session.get("inputTokensBilled") or 0) + billing.tokens
        )
        session["lastBillingUuid"] = billing.uuid


def _count_tally_turn(payload: dict[str, Any], project: Path) -> None:
    # Read the transcript first: the session stays locked only briefly.
    turn = last_assistant_turn(payload.get("transcript_path"))
    with state.update_session(project, _session_id(payload)) as session:
        if not session.get("turnUsedBytely"):
            return
        session["turnUsedBytely"] = False
        if turn is not None and turn.uuid != session.get("lastTallyUuid"):
            session["bytelyTurns"] = int(session.get("bytelyTurns") or 0) + 1
            if has_savings_tally(turn.text):
                session["reportedTurns"] = (
                    int(session.get("reportedTurns") or 0) + 1
                )
            session["lastTallyUuid"] = turn.uuid


def stop(payload: dict[str, Any], project: Path) -> None:
    """End of a turn: price it, check its tally, sync a stale graph."""
    _sample_turn_cost(payload, project)
    _count_tally_turn(payload, project)
    if (state.read_stats(project) or {}).get("dirty"):
        start_sync(project)


def start_sync(project: Path) -> bool:
    """Sync the graph in a detached process, unless one is running."""
    if not has_graph(project):
        return False
    token = state.acquire_lock(project)
    if token is None:
        return False
    state.patch_stats(project, syncing=True)
    flags: dict[str, Any] = {}
    if sys.platform == "win32":
        # Looked up by name: these constants exist only on Windows.
        flags["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        flags["start_new_session"] = True
    try:
        subprocess.Popen(  # noqa: S603 - our own interpreter and module
            [sys.executable, "-m", "bytely", "hook", "sync"],
            cwd=project,
            env={
                **os.environ,
                "CLAUDE_PROJECT_DIR": str(project),
                SYNC_TOKEN_ENV: token,
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            **flags,
        )
    except OSError:
        state.patch_stats(project, syncing=False)
        state.release_lock(project, token)
        return False
    return True


def sync(project: Path, token: str | None = None) -> None:
    """Bring the graph up to date, then mark the synced files clean.

    Runs under the sync lock: the one `start_sync` took (its token comes
    in `BYTELY_SYNC_TOKEN`), or its own when run directly. The lock is
    kept fresh while the refresh runs, so a long sync is never taken for
    a dead one, and it is released only by its owner.

    Files edited while the sync ran stay stale: only the ones pending
    when it started are cleared.
    """
    from bytely.graph.refresh import refresh_graph

    token = token or os.environ.get(SYNC_TOKEN_ENV)
    if token is None or state.lock_owner(project) != token:
        token = state.acquire_lock(project)
        if token is None:
            return  # another sync is running
    pending = set((state.read_stats(project) or {}).get("staleFiles") or [])
    try:
        custom = os.environ.get("BYTELY_DIR")
        with lock.heartbeat(
            state.cache_dir(project) / state.LOCK_FILE,
            token,
            LOCK_HEARTBEAT_SECONDS,
        ):
            result = refresh_graph(
                str(project),
                str(state.context_dir(project)) if custom else None,
                create=False,
            )
        with state.update_stats(project) as stats:
            left = [f for f in stats["staleFiles"] or [] if f not in pending]
            stats.update(
                nodeCount=len(result.graph.nodes),
                edgeCount=len(result.graph.edges),
                staleFiles=left,
                staleCount=len(left),
                dirty=bool(left),
                syncedAt=time.time(),
            )
    finally:
        state.patch_stats(project, syncing=False)
        state.release_lock(project, token)


_HANDLERS = {
    "session-start": session_start,
    "prompt": prompt,
    "post-edit": post_edit,
    "tool-savings": tool_savings,
    "stop": stop,
    "cursor-post-tool": cursor_post_tool,
    "cursor-mcp": cursor_mcp,
    "cursor-session-end": cursor_session_end,
}


def run_hook(
    event: str, payload: dict[str, Any], *, user_level: bool = False
) -> str | None:
    """Handle one hook call; return what to print, if anything.

    `user_level` marks a hook installed in the user's own settings: it
    does nothing in a project that has no graph, or whose own settings
    already run bytely's hooks.
    """
    project = project_dir(payload)
    if event == "sync":
        sync(project)
        return None
    if not has_graph(project):
        return None
    if user_level and repo_hooks_installed(project):
        return None
    result = _HANDLERS[event](payload, project)
    return result if isinstance(result, str) else None
