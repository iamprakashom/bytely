"""The concept pass: architecture nodes synthesized from file summaries.

1. Summarize each indexed file (one call per file, several at a time,
   cached by content hash in `cache/summaries.json`, flushed as it goes so
   an interrupted run resumes without paying twice).
2. Pack the summaries into batches under a size budget and synthesize
   curated nodes from each (one call per batch, cached by the batch's
   files and hashes; an empty result is never cached, so it is retried).
3. Merge nodes by slug, resolve links by name, and give concept nodes
   with no sources of their own the sources of the nodes they link to, so
   every node goes stale when its subject changes.
4. Save the result to `cache/concepts.json`; the markdown writer renders
   it as `concepts/<slug>.md`, and links file cards up to their concepts.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bytely.ai.failure import FailureGate
from bytely.util.id import content_hash

if TYPE_CHECKING:
    from collections.abc import Callable

    from bytely.ai.summarize import FileSummarizer
    from bytely.ai.synthesize import Synthesizer, SynthNode

CONCEPTS_FILE = Path("cache") / "concepts.json"
SUMMARY_CACHE_FILE = Path("cache") / "summaries.json"
BATCH_CHAR_BUDGET = 48_000
FLUSH_SECONDS = 15.0
CONCEPTS_VERSION = 1


@dataclass
class ConceptStats:
    """What the concept pass did."""

    files: int = 0
    summarized: int = 0
    cached: int = 0
    batches: int = 0
    nodes: int = 0
    links: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    fatal: str | None = None
    errors: list[str] = field(default_factory=list)
    # A failed run left the previous concept graph in place.
    kept_previous: bool = False


def slugify(name: str) -> str:
    """A file-name-safe slug: lowercase ASCII words joined by hyphens."""
    ascii_name = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or "node"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), "utf-8")
    os.replace(temp, path)


def read_concepts(context_dir: str | Path) -> dict[str, Any] | None:
    """The saved concept graph, if a `--deep` build made one."""
    data = _read_json(Path(context_dir) / CONCEPTS_FILE)
    return data if isinstance(data.get("nodes"), list) else None


def _batches(
    files: list[tuple[str, str]], budget: int
) -> list[list[tuple[str, str]]]:
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    size = 0
    for path, summary in files:
        length = len(path) + len(summary) + 8
        if current and size + length > budget:
            batches.append(current)
            current, size = [], 0
        current.append((path, summary))
        size += length
    if current:
        batches.append(current)
    return batches


def _batch_key(batch: list[tuple[str, str]], hashes: dict[str, str]) -> str:
    return content_hash(
        "\n".join(sorted(f"{path}:{hashes.get(path, '')}" for path, _ in batch))
    )


def _nodes_from_cache(raw: object) -> list[SynthNode]:
    from bytely.ai.synthesize import clean_nodes

    return clean_nodes(raw)


def build_concepts(
    root: Path,
    context_dir: Path,
    files: list[str],
    summarizer: FileSummarizer,
    synthesizer: Synthesizer,
    *,
    model: str = "",
    concurrency: int = 8,
    progress: Callable[[str, int, int, str], None] | None = None,
) -> ConceptStats:
    """Run the concept pass over `files` (absolute paths) and save it."""
    stats = ConceptStats()
    cache_path = context_dir / SUMMARY_CACHE_FILE
    cache = _read_json(cache_path)
    summaries: dict[str, Any] = cache.get("summaries") or {}
    synth: dict[str, Any] = cache.get("synth") or {}
    lock = threading.Lock()
    gate = FailureGate()
    last_flush = time.monotonic()
    done = 0

    def flush() -> None:
        _write_json(cache_path, {"summaries": summaries, "synth": synth})

    def summarize(file: str) -> tuple[str, str, str | None] | None:
        nonlocal last_flush, done
        rel = Path(file).resolve().relative_to(root).as_posix()
        try:
            code = Path(file).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        digest = content_hash(code)
        result: tuple[str, str, str | None]
        with lock:
            hit = summaries.get(rel)
        if isinstance(hit, dict) and hit.get("hash") == digest:
            with lock:
                stats.cached += 1
            result = (rel, digest, str(hit.get("summary") or ""))
        elif gate.stopped:
            gate.skip()
            result = (rel, digest, None)
        else:
            try:
                text = summarizer.summarize(rel, code)
                if not text:
                    raise ValueError("model returned an empty summary")
            except Exception as error:  # noqa: BLE001 - reported per file
                message = str(error) or type(error).__name__
                with lock:
                    stats.errors.append(f"{rel}: {message}")
                gate.record(message, quality=isinstance(error, ValueError))
                result = (rel, digest, None)
            else:
                gate.succeeded()
                with lock:
                    summaries[rel] = {"hash": digest, "summary": text}
                    stats.summarized += 1
                    if time.monotonic() - last_flush >= FLUSH_SECONDS:
                        last_flush = time.monotonic()
                        flush()
                result = (rel, digest, text)
        with lock:
            done += 1
            if progress:
                progress("summarize", done, len(files), rel)
        return result

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        work = [w for w in pool.map(summarize, files) if w is not None]
    stats.failed_files = gate.failed
    stats.skipped_files = gate.skipped
    stats.fatal = gate.fatal
    stats.files = len(work)
    flush()

    hashes = {rel: digest for rel, digest, _ in work}
    summarized = sorted((rel, text) for rel, _, text in work if text)
    batches = _batches(summarized, BATCH_CHAR_BUDGET)
    stats.batches = len(batches)
    synthesized: list[SynthNode] = []
    kept: dict[str, Any] = {}
    for index, batch in enumerate(batches, start=1):
        if progress:
            progress("synthesize", index, len(batches), f"batch {index}")
        key = _batch_key(batch, hashes)
        nodes = _nodes_from_cache(synth.get(key))
        if not nodes and gate.stopped:
            stats.errors.append(
                f"synthesis batch {index}: skipped (the pass had stopped)"
            )
        elif not nodes:
            try:
                nodes = synthesizer.synthesize(batch)
            except Exception as error:  # noqa: BLE001 - reported per batch
                message = str(error) or type(error).__name__
                stats.errors.append(f"synthesis batch {index}: {message}")
                gate.record(message)
                stats.fatal = gate.fatal or stats.fatal
                nodes = []
        if nodes:
            kept[key] = [node.to_dict() for node in nodes]
        synthesized.extend(nodes)

    complete = not stats.errors and stats.fatal is None
    target = context_dir / CONCEPTS_FILE
    if not complete and read_concepts(context_dir) is not None:
        # A partial run must not replace a complete concept graph (and,
        # through it, delete concept pages): keep the previous one, and
        # every cached batch, so the next run resumes from all of it.
        synth.update(kept)
        flush()
        stats.kept_previous = True
        previous = read_concepts(context_dir) or {}
        stats.nodes = len(previous.get("nodes", []))
        stats.links = sum(
            len(node.get("links", [])) for node in previous.get("nodes", [])
        )
        return stats
    # Only batches still produced are kept, so the cache cannot grow forever.
    if complete:
        synth.clear()
    synth.update(kept)
    flush()

    concept_nodes = merge_nodes(synthesized, hashes)
    stats.nodes = len(concept_nodes)
    stats.links = sum(len(node["links"]) for node in concept_nodes)
    _write_json(
        target,
        {
            "version": CONCEPTS_VERSION,
            "model": model,
            "files": [
                {"path": rel, "hash": hashes[rel]} for rel in sorted(hashes)
            ],
            "nodes": concept_nodes,
        },
    )
    return stats


def merge_nodes(
    synthesized: list[SynthNode], hashes: dict[str, str]
) -> list[dict[str, Any]]:
    """Merge nodes by slug; resolve links by name; inherit missing sources."""
    drafts: dict[str, dict[str, Any]] = {}
    by_name: dict[str, str] = {}
    for node in synthesized:
        slug = slugify(node.name)
        draft = drafts.setdefault(
            slug,
            {
                "name": node.name,
                "slug": slug,
                "type": node.type or "concept",
                "summary": node.summary,
                "sources": {},
                "links": {},
            },
        )
        if len(node.summary) > len(draft["summary"]):
            draft["summary"] = node.summary
        if node.type and draft["type"] == "concept":
            draft["type"] = node.type
        for source in node.sources:
            if source in hashes:
                draft["sources"][source] = hashes[source]
        by_name.setdefault(node.name.strip().lower(), slug)
    for node in synthesized:
        origin = drafts[slugify(node.name)]
        for link in node.links:
            target = by_name.get(link.to.strip().lower())
            if target is None or target == origin["slug"]:
                continue
            origin["links"].setdefault(
                (target, link.relation),
                {"to": target, "relation": link.relation}
                | (
                    {"description": link.description}
                    if link.description
                    else {}
                ),
            )
    # A node with no sources of its own inherits those of the nodes it
    # links to, else of the nodes linking to it, so it still goes stale
    # when its subject changes.
    for draft in drafts.values():
        if draft["sources"]:
            continue
        for key in draft["links"]:
            draft["sources"].update(drafts[key[0]]["sources"])
        if draft["sources"]:
            continue
        for other in drafts.values():
            if any(key[0] == draft["slug"] for key in other["links"]):
                draft["sources"].update(other["sources"])
    return [
        {
            "name": draft["name"],
            "slug": draft["slug"],
            "type": draft["type"],
            "summary": draft["summary"],
            "sources": [
                {"path": path, "hash": digest}
                for path, digest in sorted(draft["sources"].items())
            ],
            "links": sorted(
                draft["links"].values(),
                key=lambda link: (link["to"], link["relation"]),
            ),
        }
        for draft in sorted(drafts.values(), key=lambda d: d["slug"])
    ]
