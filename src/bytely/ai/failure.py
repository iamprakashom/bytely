"""When an LLM pass stops calling a provider that is failing.

Without this, a spent quota turns into one failed call per file and a
build that still looks successful. One gate per pass, shared by its
worker threads:

- a quota or authentication error stops the pass at once (retrying cannot
  fix it);
- `MAX_CONSECUTIVE_FAILURES` failures in a row stop it too;
- a content-quality miss (the model answered, but unusably) counts as a
  failed file, so it is not cached as success, without ending the pass.
"""

from __future__ import annotations

import re
import threading

MAX_CONSECUTIVE_FAILURES = 5
_QUOTA = re.compile(
    r"quota|insufficient[_ ]funds|insufficient[_ ]quota|billing"
    r"|payment required|\b402\b|credit balance",
    re.IGNORECASE,
)
_AUTH = re.compile(
    r"\b401\b|\b403\b|unauthorized|invalid api key|invalid_api_key"
    r"|authentication|permission denied",
    re.IGNORECASE,
)


def terminal_reason(message: str) -> str | None:
    """Why no retry can help, or None if the failure may be transient."""
    if _QUOTA.search(message):
        return "the provider reports the quota/credit for this key is exhausted"
    if _AUTH.search(message):
        return "the provider rejected the API key"
    return None


class FailureGate:
    """Failure bookkeeping for one pass (thread-safe)."""

    def __init__(self) -> None:
        """Start with nothing failed."""
        self.failed = 0
        self.skipped = 0
        self.fatal: str | None = None
        self._consecutive = 0
        self._lock = threading.Lock()

    @property
    def stopped(self) -> bool:
        """Whether the pass should stop issuing calls."""
        return self.fatal is not None

    def record(self, message: str, *, quality: bool = False) -> None:
        """Count a failed unit; decide whether it ends the pass."""
        with self._lock:
            self.failed += 1
            if self.fatal is not None:
                return
            terminal = terminal_reason(message)
            if terminal:
                self.fatal = (
                    f"{terminal} — stopped after {self.failed} failed "
                    f"file(s). First error: {message}"
                )
                return
            if quality:
                return
            self._consecutive += 1
            if self._consecutive >= MAX_CONSECUTIVE_FAILURES:
                self.fatal = (
                    f"{self._consecutive} files in a row failed, so the pass "
                    f"stopped rather than keep calling. Last error: {message}"
                )

    def succeeded(self) -> None:
        """A success breaks a run of failures."""
        with self._lock:
            self._consecutive = 0

    def skip(self) -> None:
        """Count a unit never attempted because the pass had stopped."""
        with self._lock:
            self.skipped += 1
