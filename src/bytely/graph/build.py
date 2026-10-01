"""Build the structural code graph."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from bytely.graph.container import is_container_config_file
from bytely.graph.enrich import (
    DEFAULT_CONCURRENCY,
    MeaningStats,
    carry_over,
    enrich,
)
from bytely.graph.extract import RawEdge, depth_extensions, extract_file
from bytely.graph.extract_cache import ExtractionCache
from bytely.graph.invariants import validate_graph
from bytely.graph.outputs import (
    DEFAULT_FORMATS,
    INDEX_FILE,
    outputs_intact,
    write_outputs,
)
from bytely.graph.refresh import (
    build_lock,
    load_build_state,
    save_tree_fingerprint,
    tree_fingerprint,
)
from bytely.graph.resolve import resolve_edges
from bytely.graph.scopes import discover_scopes
from bytely.graph.source_files import filter_source_files
from bytely.graph.types import GraphV1, NodeV1, ScopeV1
from bytely.graph.write import graph_path, read_graph, write_graph
from bytely.ingest.fs import walk_dir
from bytely.util.id import content_hash
from bytely.util.paths import rel_posix
from bytely.util.phases import PhaseTimer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from bytely.ai.crux import CruxSummarizer


RACY_WINDOW_NS = 2_000_000_000

# Modules past extraction whose code shapes a build's output (extraction
# itself is covered by `current_extractor_fingerprint`).
PIPELINE_MODULES = (
    "bytely.ai.concepts",
    "bytely.graph.build",
    "bytely.graph.enrich",
    "bytely.graph.invariants",
    "bytely.graph.outputs",
    "bytely.graph.resolve",
    "bytely.graph.scopes",
    "bytely.graph.types",
    "bytely.graph.write",
)


@dataclass
class GraphBuildResult:
    """Paths and counts produced by a graph build."""

    context_dir: str
    graph_json: str
    files: int
    nodes: int
    edges: int
    cache_hits: int = 0
    cache_misses: int = 0
    meaning: MeaningStats | None = None


@dataclass
class DeepOptions:
    """The `--deep` meaning pass: who summarizes, and how."""

    summarizer: CruxSummarizer
    concurrency: int = DEFAULT_CONCURRENCY
    progress: Callable[[int, int, str], None] | None = None


def list_source_files(
    root_path: Path,
    include_patterns: Iterable[str] = (),
    exclude_patterns: Iterable[str] = (),
) -> list[str]:
    """Absolute paths of the files a build indexes, in build order."""
    return filter_source_files(
        root_path,
        walk_dir(
            str(root_path),
            depth_extensions(),
            include_file=is_container_config_file,
        ),
        include_patterns,
        exclude_patterns,
    )


@cache
def pipeline_fingerprint() -> str:
    """Fingerprint of the build code past extraction, installed right now.

    The tree snapshot covers the extractor; this covers resolution, scopes,
    the meaning carry-over, and rendering, so upgrading Bytely makes the
    next build redo its work instead of keeping the old outputs.
    """
    import bytely

    digest = hashlib.sha256(f"bytely:{bytely.__version__}".encode())
    for name in PIPELINE_MODULES:
        spec = importlib.util.find_spec(name)
        origin = spec.origin if spec else None
        # No readable source (a frozen or zipped install): the version above
        # still changes on every release.
        with contextlib.suppress(OSError):
            digest.update(Path(origin).read_bytes() if origin else b"")
    return digest.hexdigest()


def _scopes_key(scopes: list[ScopeV1]) -> str:
    data = [[s.prefix, s.label, list(s.markers)] for s in scopes]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def _meaning_counts(nodes: list[NodeV1]) -> tuple[int, int, int]:
    """(ready, pending, stale): what a non-deep rebuild of these reports.

    Mirrors `carry_over` on an unchanged tree: a node without a summary is
    pending, a ready one is carried over, and any other one is stale.
    """
    ready = pending = stale = 0
    for node in nodes:
        if not node.summary:
            pending += 1
        elif node.summary_state == "ready":
            ready += 1
        else:
            stale += 1
    return ready, pending, stale


def _outputs_stamp(
    ctx_dir: str, formats: tuple[str, ...], scopes: list[ScopeV1]
) -> str:
    """Stamp the outputs a build left, and their inputs besides the sources.

    The build code itself, the discovered scopes (from manifests and
    `.gitignore`), the saved concepts the cards render, and the graph file
    and index (which can be edited or truncated by hand) are all part of
    it, so a change to any of them makes the next build do the work.
    """
    from bytely.ai.concepts import CONCEPTS_FILE

    files = [graph_path(ctx_dir), Path(ctx_dir) / CONCEPTS_FILE]
    if "markdown" in formats:
        files.append(Path(ctx_dir) / INDEX_FILE)
    parts = [",".join(formats), pipeline_fingerprint(), _scopes_key(scopes)]
    for file in files:
        try:
            stat = os.stat(file)
            parts.append(f"{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            parts.append("none")
    return "|".join(parts)


def build_graph(
    root: str,
    context_dir: str | None = None,
    *,
    cache_dir: str | None = None,
    persist_cache: bool = True,
    include_patterns: Iterable[str] = (),
    exclude_patterns: Iterable[str] = (),
    output_formats: Iterable[str] = DEFAULT_FORMATS,
    deep: DeepOptions | None = None,
) -> GraphBuildResult:
    """Build the graph file, and the requested outputs, from the repository.

    The meaning layer (summaries and crux) of the saved graph is carried
    over for unchanged definitions; `deep` computes it for the rest.

    Holds the graph's build lock throughout, so two processes (a query, the
    MCP server, a background sync, `bytely build`) never write the same
    graph files at once.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(
            f"Repository directory does not exist: {root_path}"
        )
    ctx_dir = context_dir if context_dir else str(root_path / "bytely")
    with build_lock(ctx_dir):
        return _build_graph(
            root_path,
            ctx_dir,
            cache_dir=cache_dir,
            persist_cache=persist_cache,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            output_formats=output_formats,
            deep=deep,
        )


def _build_graph(
    root_path: Path,
    ctx_dir: str,
    *,
    cache_dir: str | None,
    persist_cache: bool,
    include_patterns: Iterable[str],
    exclude_patterns: Iterable[str],
    output_formats: Iterable[str],
    deep: DeepOptions | None,
) -> GraphBuildResult:
    # 1. Walk repo for source files, and snapshot them before reading any,
    # so an edit made during the build leaves the graph marked stale.
    timer = PhaseTimer()
    includes, excludes = list(include_patterns), list(exclude_patterns)
    formats = tuple(sorted(output_formats))
    abs_paths = list_source_files(root_path, includes, excludes)
    timer.mark("discover")
    scopes = discover_scopes(str(root_path))
    timer.mark("scopes")
    snapshot = tree_fingerprint(root_path, abs_paths, includes, excludes)
    outputs = _outputs_stamp(ctx_dir, formats, scopes)

    # Nothing changed since the last build wrote these same outputs: the
    # graph on disk is already this build's result.
    state = load_build_state(ctx_dir)
    if (
        deep is None
        and state.fingerprint == snapshot
        and state.outputs == outputs
        and state.counts is not None
        and state.meaning is not None
        and outputs_intact(Path(ctx_dir), formats)
    ):
        files, node_count, edge_count = state.counts
        ready, pending, stale = state.meaning
        timer.mark("fingerprint")
        timer.finish(files=files, noop=True)
        return GraphBuildResult(
            context_dir=ctx_dir,
            graph_json=str(graph_path(ctx_dir)),
            files=files,
            nodes=node_count,
            edges=edge_count,
            cache_hits=files,
            # What a full non-deep rebuild of this unchanged graph reports.
            meaning=MeaningStats(cached=ready, pending=pending, stale=stale),
        )

    timer.mark("fingerprint")
    extraction_cache = ExtractionCache.load(cache_dir or ctx_dir)
    timer.mark("cache_load")
    # Only a build that keeps its cache may skip reading files by stat;
    # `check` (persist_cache=False) compares against a rebuild that reads
    # every file, so a stale stat can never make a stale graph look fresh.
    trust_stat = deep is None and persist_cache
    # A file modified this recently may still change within its mtime's
    # resolution (2 s on FAT), so its stat isn't recorded as trustworthy;
    # the next build hashes it again (git's "racily clean" rule).
    racy_after_ns = time.time_ns() - RACY_WINDOW_NS
    racy = False

    # 2. Extract nodes & raw edges from each file
    nodes: list[NodeV1] = []
    raw_edges: list[RawEdge] = []
    cached_paths: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    processed_files = 0
    sources: dict[str, str] = {}

    for path in abs_paths:
        rel = rel_posix(str(root_path), path)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        # Without the meaning pass no source text is needed, so a file whose
        # size and mtime match its cache entry isn't read at all.
        if trust_stat:
            res = extraction_cache.get_unchanged(
                rel, stat.st_size, stat.st_mtime_ns
            )
            if res is not None:
                processed_files += 1
                cached_paths.add(rel)
                cache_hits += 1
                nodes.extend(res.nodes)
                raw_edges.extend(res.raw_edges)
                continue
        try:
            # Decode the raw bytes: read_text's universal newlines would turn
            # CRLF into LF, so body_hash would no longer match the file (or
            # the reference implementation, which hashes the source as read).
            source = Path(path).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        processed_files += 1
        cached_paths.add(rel)
        if deep is not None:
            sources[rel] = source
        source_hash = content_hash(source)
        # The stat taken before reading: if the file changes after it, the
        # next build sees a different stat and reads it again.
        file_stat = (
            (stat.st_size, stat.st_mtime_ns)
            if stat.st_mtime_ns < racy_after_ns
            else None
        )
        racy = racy or file_stat is None
        res = extraction_cache.get(rel, source_hash)
        if res is None:
            res = extract_file(rel, source)
            extraction_cache.set(rel, source_hash, res, file_stat)
            cache_misses += 1
        else:
            extraction_cache.set_stat(rel, file_stat)
            cache_hits += 1
        nodes.extend(res.nodes)
        raw_edges.extend(res.raw_edges)

    timer.mark("extract")

    # 3. Resolve cross-file edges
    edges = resolve_edges(nodes, raw_edges)
    timer.mark("resolve")

    # 4. Construct GraphV1 and save
    graph = GraphV1(
        version=1,
        nodes=nodes,
        edges=edges,
        scopes=scopes,
    )

    validate_graph(graph)
    dirty = carry_over(nodes, read_graph(ctx_dir))
    meaning = enrich(
        nodes,
        dirty,
        sources,
        deep.summarizer if deep else None,
        concurrency=deep.concurrency if deep else DEFAULT_CONCURRENCY,
        checkpoint=lambda: write_graph(graph, ctx_dir),
        progress=deep.progress if deep else None,
    )
    timer.mark("enrich")
    if persist_cache:
        extraction_cache.retain(cached_paths)
        extraction_cache.save()
    timer.mark("cache_save")
    write_graph(graph, ctx_dir)
    timer.mark("write_graph")
    # `formats`, not `output_formats`: sorting it may have consumed an
    # iterator.
    write_outputs(graph, ctx_dir, formats)
    timer.mark("outputs")
    save_tree_fingerprint(
        ctx_dir,
        snapshot,
        includes,
        excludes,
        # Stamped after writing, so it covers the files just written.
        outputs=_outputs_stamp(ctx_dir, formats, scopes),
        # Without counts the next build can't take the no-change shortcut:
        # a file too fresh to trust by stat must be read again.
        counts=None if racy else (processed_files, len(nodes), len(edges)),
        meaning=_meaning_counts(nodes),
    )
    timer.mark("state")
    timer.finish(
        files=processed_files,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        noop=False,
    )

    return GraphBuildResult(
        context_dir=ctx_dir,
        graph_json=str(graph_path(ctx_dir)),
        files=processed_files,
        nodes=len(nodes),
        edges=len(edges),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        meaning=meaning,
    )
