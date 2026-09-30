"""Build the structural code graph."""

from __future__ import annotations

from dataclasses import dataclass
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
from bytely.graph.outputs import DEFAULT_FORMATS, write_outputs
from bytely.graph.refresh import (
    build_lock,
    save_tree_fingerprint,
    tree_fingerprint,
)
from bytely.graph.resolve import resolve_edges
from bytely.graph.scopes import discover_scopes
from bytely.graph.source_files import filter_source_files
from bytely.graph.types import GraphV1, NodeV1
from bytely.graph.write import graph_path, read_graph, write_graph
from bytely.ingest.fs import walk_dir
from bytely.util.id import content_hash
from bytely.util.paths import rel_posix

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from bytely.ai.crux import CruxSummarizer


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
    extraction_cache = ExtractionCache.load(cache_dir or ctx_dir)

    # 1. Walk repo for source files, and snapshot them before reading any,
    # so an edit made during the build leaves the graph marked stale.
    includes, excludes = list(include_patterns), list(exclude_patterns)
    abs_paths = list_source_files(root_path, includes, excludes)
    snapshot = tree_fingerprint(root_path, abs_paths, includes, excludes)

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
        res = extraction_cache.get(rel, source_hash)
        if res is None:
            res = extract_file(rel, source)
            extraction_cache.set(rel, source_hash, res)
            cache_misses += 1
        else:
            cache_hits += 1
        nodes.extend(res.nodes)
        raw_edges.extend(res.raw_edges)

    # 3. Resolve cross-file edges
    edges = resolve_edges(nodes, raw_edges)

    # 4. Construct GraphV1 and save
    graph = GraphV1(
        version=1,
        nodes=nodes,
        edges=edges,
        scopes=discover_scopes(str(root_path)),
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
    if persist_cache:
        extraction_cache.retain(cached_paths)
        extraction_cache.save()
    write_graph(graph, ctx_dir)
    write_outputs(graph, ctx_dir, output_formats)
    save_tree_fingerprint(ctx_dir, snapshot, includes, excludes)

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
