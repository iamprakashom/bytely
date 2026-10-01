"""Optional per-stage timings, for benchmarking a build.

When `BYTELY_PHASE_LOG` names a file, each timed run appends one JSON line
to it: `{"phases": {stage: seconds, ...}, ...}`. Without the variable the
timer only reads the clock, so it costs nothing worth measuring.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

PHASE_LOG_ENV = "BYTELY_PHASE_LOG"


class PhaseTimer:
    """Time consecutive stages of one run."""

    def __init__(self) -> None:
        """Start timing the first stage."""
        self.phases: dict[str, float] = {}
        self._last = time.perf_counter()

    def mark(self, phase: str) -> None:
        """End the current stage, recording it under `phase`."""
        now = time.perf_counter()
        self.phases[phase] = self.phases.get(phase, 0.0) + now - self._last
        self._last = now

    def finish(self, **info: Any) -> None:
        """Append this run's timings to the phase log, if one is set."""
        target = os.environ.get(PHASE_LOG_ENV)
        if not target:
            return
        record = {
            "phases": {k: round(v, 4) for k, v in self.phases.items()},
            **info,
        }
        try:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            pass  # timings are diagnostics; never fail a build over them
