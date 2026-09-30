"""Cross-process locks held as files, with owner tokens and liveness.

A lock is a JSON file created exclusively (`O_CREAT | O_EXCL`), so only one
process can hold it. Its owner keeps its modification time fresh while it
works (`Heartbeat`); a lock not refreshed within `stale` seconds belongs to
a process that died, and may be reclaimed. Only the owner, identified by
its token, can release it.

Reclaiming is race-free: the stale file is renamed aside first, which only
one process can do, and the claimant then checks that what it renamed was
really stale — if the owner refreshed it in the meantime, it is put back.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


def owner(path: Path) -> str | None:
    """The token of whoever holds the lock, if anyone."""
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    token = data.get("token") if isinstance(data, dict) else None
    return token if isinstance(token, str) else None


def alive(path: Path, stale: float) -> bool:
    """Whether the lock is held by a process that is still refreshing it."""
    try:
        return time.time() - path.stat().st_mtime < stale
    except OSError:
        return False


def _create(path: Path, token: str) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(
            {"token": token, "pid": os.getpid(), "at": time.time()}, handle
        )
    return True


def _reclaim(path: Path, stale: float) -> None:
    """Move a dead lock aside, restoring it if it turned out to be live."""
    aside = path.with_name(f"{path.name}.{uuid.uuid4().hex}.stale")
    try:
        os.rename(path, aside)
    except OSError:
        return  # gone already, or another process took it aside
    if alive(aside, stale):
        # Its owner refreshed it between our check and the rename: give it
        # back, unless a new lock was created in the gap (then two holders
        # is unavoidable, and the file we hold is simply dropped).
        with contextlib.suppress(OSError):
            if not path.exists():
                os.rename(aside, path)
                return
    with contextlib.suppress(OSError):
        aside.unlink()


def try_acquire(path: Path, stale: float) -> str | None:
    """Take the lock now, or return None if a live owner holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    for _ in range(3):
        if _create(path, token):
            return token
        if alive(path, stale):
            return None
        _reclaim(path, stale)
    return None


def acquire(path: Path, stale: float, wait: float) -> str | None:
    """Take the lock, waiting up to `wait` seconds for a live owner."""
    deadline = time.monotonic() + wait
    delay = 0.01
    while True:
        token = try_acquire(path, stale)
        if token is not None or time.monotonic() >= deadline:
            return token
        time.sleep(delay)
        delay = min(delay * 2, 0.25)


def touch(path: Path, token: str) -> bool:
    """Refresh a held lock; False once `token` no longer owns it."""
    if owner(path) != token:
        return False
    with contextlib.suppress(OSError):
        os.utime(path)
    return True


def release(path: Path, token: str) -> None:
    """Drop the lock, only if `token` still owns it."""
    if owner(path) == token:
        with contextlib.suppress(OSError):
            path.unlink()


@contextlib.contextmanager
def heartbeat(path: Path, token: str, every: float) -> Generator[None]:
    """Keep a held lock fresh from a background thread during the block."""
    done = threading.Event()

    def beat() -> None:
        while not done.wait(every):
            if not touch(path, token):
                return

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()


@contextlib.contextmanager
def held(path: Path, *, stale: float, wait: float) -> Generator[str | None]:
    """Hold the lock for the block, kept fresh while it runs.

    Yields the token, or None if the wait ran out (the block then runs
    unlocked: the caller decides whether that is acceptable). Re-entrant
    within a thread: a nested `held` on a lock the thread already holds
    yields the same token without waiting on itself.
    """
    key = str(path.resolve())
    depth = _held_here.__dict__.setdefault("locks", {})
    if key in depth:
        token, count = depth[key]
        depth[key] = (token, count + 1)
        try:
            yield token
        finally:
            depth[key] = (token, depth[key][1] - 1)
            if depth[key][1] == 0:
                del depth[key]
        return
    token = acquire(path, stale, wait)
    if token is None:
        yield None
        return
    depth[key] = (token, 1)
    try:
        with heartbeat(path, token, stale / 4):
            yield token
    finally:
        del depth[key]
        release(path, token)


_held_here = threading.local()
