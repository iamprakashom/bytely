"""The meaning layer on graph nodes: a summary and a crux per definition.

The saved graph is its own cache. Every build (not only `--deep`) carries
the meaning layer over by node id and body hash:

- ready: same id and body as a node summarized before; kept, no call.
- stale: summarized before, but its body changed since; the old text is
  kept as a hint, marked stale, and recomputed by the next `--deep`.
- pending: never summarized.

With a summarizer (`--deep`), stale and pending nodes get one LLM call per
file, several files at a time. The model returns line numbers; the crux
code is cut here, once, verbatim from the source, clamped to the node's
own span. A killed run keeps what it computed: `checkpoint` is called
periodically to save the graph.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bytely.ai.crux import Target
from bytely.ai.failure import FailureGate
from bytely.graph.types import Crux

if TYPE_CHECKING:
    from collections.abc import Callable

    from bytely.ai.crux import CruxSummarizer, SymbolNote
    from bytely.graph.types import GraphV1, NodeV1

MAX_CRUX_LINES = 12
DEFAULT_CONCURRENCY = 5
CHECKPOINT_SECONDS = 15.0


@dataclass
class MeaningStats:
    """What the meaning pass did."""

    cached: int = 0
    computed: int = 0
    stale: int = 0
    pending: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    fatal: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> int:
        """Nodes with a current summary."""
        return self.cached + self.computed


def carry_over(nodes: list[NodeV1], prior: GraphV1 | None) -> list[NodeV1]:
    """Copy the meaning layer from `prior`; return the nodes that need it."""
    before = {node.id: node for node in prior.nodes} if prior else {}
    dirty = []
    for node in nodes:
        was = before.get(node.id)
        if was is None or not was.summary:
            node.summary, node.crux, node.summary_state = None, None, "pending"
            dirty.append(node)
            continue
        node.summary, node.crux = was.summary, was.crux
        if was.summary_state == "ready" and was.body_hash == node.body_hash:
            node.summary_state = "ready"
            node.crux = _shift_crux(was, node)
        else:
            node.summary_state = "stale"
            dirty.append(node)
    return dirty


def _shift_crux(was: NodeV1, node: NodeV1) -> Crux | None:
    """The crux moved with its definition (same body, new position).

    The crux stores file line numbers; lines added or removed above the
    definition move them by the same amount as the definition's start.
    """
    if was.crux is None:
        return None
    try:
        old_start = int(was.span.removeprefix("L").split("-L")[0])
        new_start = int(node.span.removeprefix("L").split("-L")[0])
        first, last = (
            int(part) for part in was.crux.span.removeprefix("L").split("-L")
        )
    except ValueError:
        return was.crux
    delta = new_start - old_start
    if delta == 0:
        return was.crux
    return Crux(was.crux.code, f"L{first + delta}-L{last + delta}")


def span_lines(span: str, line_count: int) -> tuple[int, int]:
    """A `L12-L30` span as clamped 1-based lines (the whole file if bad)."""
    try:
        start_text, end_text = span.removeprefix("L").split("-L")
        start, end = int(start_text), int(end_text)
    except ValueError:
        return 1, max(1, line_count)
    start = max(1, min(start, line_count))
    return start, max(start, min(end, line_count))


def cut_crux(note: SymbolNote, node: NodeV1, lines: list[str]) -> Crux | None:
    """The crux cut from the source, or None for 0/0 or an unusable range."""
    if note.crux_start < 1 or note.crux_end < note.crux_start:
        return None
    first, last = span_lines(node.span, len(lines))
    start = max(first, min(note.crux_start, last))
    end = max(start, min(note.crux_end, last))
    end = min(end, start + MAX_CRUX_LINES - 1)
    code = "\n".join(lines[start - 1 : end])
    return Crux(code, f"L{start}-L{end}") if code.strip() else None


def _describe_all(
    summarizer: CruxSummarizer,
    path: str,
    source: str,
    targets: list[Target],
) -> tuple[dict[str, SymbolNote], str | None, bool]:
    """Notes for every target, re-asking once for any the model omitted.

    Returns the notes, the last error, and whether that error is a
    content-quality miss (the model answered, unusably).
    """
    notes: dict[str, SymbolNote] = {}
    missing = targets
    miss: str | None = None
    for _ in range(2):
        if not missing:
            break
        try:
            found, miss = summarizer.describe(path, source, missing)
        except Exception as error:  # noqa: BLE001 - reported per file
            return notes, str(error) or type(error).__name__, False
        for note in found:
            notes.setdefault(note.id, note)
        missing = [t for t in targets if t.id not in notes]
    if not any(note.summary for note in notes.values()):
        return notes, miss or "model returned no usable symbol summaries", True
    return notes, None, False


def enrich(
    nodes: list[NodeV1],
    dirty: list[NodeV1],
    sources: dict[str, str],
    summarizer: CruxSummarizer | None,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    checkpoint: Callable[[], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> MeaningStats:
    """Compute the meaning layer for `dirty` (after `carry_over`)."""
    stats = MeaningStats(cached=len(nodes) - len(dirty))
    lock = threading.Lock()

    def left_undone(node: NodeV1) -> None:
        if node.summary_state == "stale":
            stats.stale += 1
        else:
            stats.pending += 1

    if summarizer is None:
        for node in dirty:
            left_undone(node)
        return stats

    by_file: dict[str, list[NodeV1]] = {}
    for node in dirty:
        if node.path in sources:
            by_file.setdefault(node.path, []).append(node)
        else:
            left_undone(node)
    gate = FailureGate()
    files = sorted(by_file)
    done = 0
    last_checkpoint = time.monotonic()

    def run(path: str) -> None:
        nonlocal done, last_checkpoint
        file_nodes = by_file[path]
        source = sources[path]
        lines = source.split("\n")
        if gate.stopped:
            gate.skip()
            with lock:
                for node in file_nodes:
                    left_undone(node)
        else:
            targets = [
                Target(
                    n.id, n.kind, *span_lines(n.span, len(lines)), n.signature
                )
                for n in file_nodes
            ]
            notes, error, quality = _describe_all(
                summarizer, path, source, targets
            )
            with lock:
                missing = sum(
                    not (notes.get(n.id) and notes[n.id].summary)
                    for n in file_nodes
                )
                if missing and not error:
                    # A partial reply is a failed file: it must not pass
                    # as a complete meaning layer (`--allow-partial` says so).
                    error = (
                        f"model left {missing} of {len(file_nodes)} "
                        "definition(s) unsummarized"
                    )
                    quality = True
                for node in file_nodes:
                    note = notes.get(node.id)
                    if note is None or not note.summary:
                        left_undone(node)
                        continue
                    node.summary = note.summary
                    node.crux = cut_crux(note, node, lines)
                    node.summary_state = "ready"
                    stats.computed += 1
                if error:
                    stats.errors.append(f"{path}: {error}")
            if error:
                gate.record(error, quality=quality)
            else:
                gate.succeeded()
        with lock:
            done += 1
            if progress:
                progress(done, len(files), path)
            due = time.monotonic() - last_checkpoint >= CHECKPOINT_SECONDS
            if checkpoint and due:
                last_checkpoint = time.monotonic()
                checkpoint()

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(run, files))
    stats.failed_files = gate.failed
    stats.skipped_files = gate.skipped
    stats.fatal = gate.fatal
    return stats
