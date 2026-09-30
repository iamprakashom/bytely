"""State the hooks keep between calls, under `<context>/cache/`.

- `stats.json`: graph size and freshness, for the status line.
- `session/<id>.json`: one agent session's reads, savings, and the
  pointers already injected into it.
- `.sync.lock`: held while a background sync runs.

Every read tolerates a missing or broken file (a hook must never fail the
host's turn), and every write is atomic, since hooks of one session can
run concurrently.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bytely.util import lock

if TYPE_CHECKING:
    from collections.abc import Generator

LOCK_FILE = ".sync.lock"
LOCK_STALE_SECONDS = 300.0
EDIT_LOCK_WAIT_SECONDS = 2.0
EDIT_LOCK_STALE_SECONDS = 10.0
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]")


def context_dir(project: Path) -> Path:
    """The graph folder: `BYTELY_DIR` (relative to the project) or `bytely`."""
    override = os.environ.get("BYTELY_DIR")
    if not override:
        return project / "bytely"
    path = Path(override)
    return path if path.is_absolute() else project / path


def cache_dir(project: Path) -> Path:
    """Where hook state lives."""
    return context_dir(project) / "cache"


def read_json(path: Path) -> dict[str, Any] | None:
    """A JSON object from disk, or None when missing or unreadable."""
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON atomically (temp file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2), "utf-8")
        # On Windows, replacing a file another process has open (a status
        # line reading it) fails with "access denied" for a moment.
        for attempt in range(20):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01)
    except OSError:
        with contextlib.suppress(OSError):
            temp.unlink()
        raise


def empty_stats() -> dict[str, Any]:
    """Stats before anything is known."""
    return {
        "nodeCount": 0,
        "edgeCount": 0,
        "staleCount": 0,
        "staleFiles": [],
        "dirty": False,
        "syncing": False,
        "syncedAt": None,
        "lastFile": None,
    }


def _stats_path(project: Path) -> Path:
    return cache_dir(project) / "stats.json"


@contextlib.contextmanager
def _locked(path: Path) -> Generator[None]:
    """Serialize read-modify-write of one state file across processes.

    Hooks of one session run concurrently (parallel tool calls), so an
    unguarded read, change, write loses updates. The lock is a sibling
    file created exclusively; one older than `EDIT_LOCK_STALE_SECONDS` is
    abandoned (its process died). Waiting is bounded: after
    `EDIT_LOCK_WAIT_SECONDS` the edit goes ahead unlocked, because a hook
    must never hang the host.
    """
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + EDIT_LOCK_WAIT_SECONDS
    held = False
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            held = True
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > EDIT_LOCK_STALE_SECONDS:
                    lock.unlink()
                    continue
            except OSError:
                continue
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        except OSError:
            break
    try:
        yield
    finally:
        if held:
            with contextlib.suppress(OSError):
                lock.unlink()


def read_stats(project: Path) -> dict[str, Any] | None:
    """The cached stats, if any."""
    return read_json(_stats_path(project))


@contextlib.contextmanager
def update_stats(project: Path) -> Generator[dict[str, Any]]:
    """Edit the stats in place, atomically with respect to other hooks."""
    path = _stats_path(project)
    with _locked(path):
        stats = {**empty_stats(), **(read_json(path) or {})}
        yield stats
        write_json(path, stats)


def patch_stats(project: Path, **patch: Any) -> dict[str, Any]:
    """Merge `patch` into the stats and save them."""
    with update_stats(project) as stats:
        stats.update(patch)
    return stats


def empty_session() -> dict[str, Any]:
    """A session before any hook has recorded anything."""
    return {
        "lastQuery": None,
        "perAgentQuery": {},
        "bytelyReads": 0,
        "sourceReads": 0,
        "savedTokens": 0,
        "injectedPointers": [],
        "nudges": 0,
    }


def session_dir(project: Path) -> Path:
    """Where per-session files live."""
    return cache_dir(project) / "session"


def _session_path(project: Path, session_id: str) -> Path:
    safe = _SAFE_ID.sub("_", session_id or "default")[:120] or "default"
    return session_dir(project) / f"{safe}.json"


def read_session(project: Path, session_id: str) -> dict[str, Any]:
    """A session's state (empty when new)."""
    stored = read_json(_session_path(project, session_id))
    return {**empty_session(), **(stored or {})}


def write_session(
    project: Path, session_id: str, session: dict[str, Any]
) -> None:
    """Save a whole session; hooks editing one use `update_session`."""
    path = _session_path(project, session_id)
    with _locked(path):
        write_json(path, session)


@contextlib.contextmanager
def update_session(project: Path, session_id: str) -> Generator[dict[str, Any]]:
    """Edit a session in place, atomically with respect to other hooks."""
    path = _session_path(project, session_id)
    with _locked(path):
        session = {**empty_session(), **(read_json(path) or {})}
        yield session
        write_json(path, session)


def latest_session(project: Path) -> dict[str, Any] | None:
    """The most recently written session, with its id under `id`."""
    try:
        candidates = [
            (path.stat().st_mtime, path)
            for path in session_dir(project).glob("*.json")
        ]
    except OSError:
        return None
    if not candidates:
        return None
    _, newest = max(candidates)
    return {
        "id": newest.stem,
        **empty_session(),
        **(read_json(newest) or {}),
    }


def _lock_path(project: Path) -> Path:
    return cache_dir(project) / LOCK_FILE


def acquire_lock(project: Path) -> str | None:
    """Take the sync lock and return its owner token, or None if held.

    A lock not refreshed for five minutes is reclaimed (its sync died);
    a live sync keeps it fresh with `touch_lock`.
    """
    return lock.try_acquire(_lock_path(project), LOCK_STALE_SECONDS)


def lock_owner(project: Path) -> str | None:
    """The token of whoever holds the sync lock."""
    return lock.owner(_lock_path(project))


def sync_running(project: Path) -> bool:
    """Whether a sync holds the lock and is still alive."""
    return lock.alive(_lock_path(project), LOCK_STALE_SECONDS)


def touch_lock(project: Path, token: str) -> bool:
    """Keep a held lock fresh; False once someone else owns it."""
    return lock.touch(_lock_path(project), token)


def release_lock(project: Path, token: str) -> None:
    """Drop the sync lock, only if `token` still owns it."""
    lock.release(_lock_path(project), token)
