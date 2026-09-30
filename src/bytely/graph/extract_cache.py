"""Content-hash cache for deterministic per-file graph extraction."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import asdict
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from bytely.graph.extract import GRAMMAR_LOADERS, ExtractResult, RawEdge
from bytely.graph.types import NodeV1

if TYPE_CHECKING:
    # `set` is shadowed by `ExtractionCache.set` inside the class body.
    from collections.abc import Container, Mapping

# Version of the cache file's layout. Extraction changes no longer need a
# bump: the extractor fingerprint below invalidates the cache on its own.
CACHE_VERSION = 7
CACHE_FILENAME = "extractions-v1.json"

# Modules whose code determines what a file extracts to (or how a result is
# stored). Editing any of them invalidates every cached extraction.
EXTRACTOR_MODULES = (
    "bytely.graph.extract",
    "bytely.graph.generic",
    "bytely.graph.container",
    "bytely.graph.types",
    "bytely.util.id",
    __name__,
)

# Grammar packages parse the source, so their versions shape the output too.
# `tree_sitter_javascript` is imported lazily in `extract._parse_javascript`,
# outside GRAMMAR_LOADERS; a test checks that every `tree_sitter_*` module
# named in extract.py is covered here.
GRAMMAR_MODULES = tuple(
    sorted(
        {module for module, _ in GRAMMAR_LOADERS.values()}
        | {"tree_sitter", "tree_sitter_javascript"}
    )
)


def extractor_fingerprint(
    module_sources: Mapping[str, bytes], package_versions: Mapping[str, str]
) -> str:
    """Hash extractor sources and parser versions into one cache key."""
    parts = [f"layout:{CACHE_VERSION}"]
    parts.extend(
        f"module:{name}:{hashlib.sha256(module_sources[name]).hexdigest()}"
        for name in sorted(module_sources)
    )
    parts.extend(
        f"package:{name}:{package_versions[name]}"
        for name in sorted(package_versions)
    )
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


@cache
def current_extractor_fingerprint() -> str:
    """Fingerprint of the extractor code and grammars installed right now."""
    sources: dict[str, bytes] = {}
    for name in EXTRACTOR_MODULES:
        spec = importlib.util.find_spec(name)
        origin = spec.origin if spec else None
        try:
            sources[name] = Path(origin).read_bytes() if origin else b""
        except OSError:
            # No readable source (a frozen or zipped install): fall back to
            # the package version, which still changes on every release.
            sources[name] = _package_version("bytely").encode()
    versions = {
        name: _package_version(name.replace("_", "-"))
        for name in GRAMMAR_MODULES
    }
    return extractor_fingerprint(sources, versions)


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "missing"


class ExtractionCache:
    """Persist and retrieve content-hash keyed extraction results."""

    def __init__(self, context_dir: str) -> None:
        """Create a cache backed by the given graph directory."""
        self.path = Path(context_dir) / "cache" / CACHE_FILENAME
        self.entries: dict[str, dict[str, Any]] = {}
        # Whether entries differ from the file on disk. A warm build with
        # only cache hits has nothing to write, so it skips the rewrite.
        self.dirty = False

    @classmethod
    def load(cls, context_dir: str) -> ExtractionCache:
        """Load a compatible cache, returning empty when unavailable."""
        cache = cls(context_dir)
        try:
            data = orjson.loads(cache.path.read_bytes())
            if (
                data.get("version") == CACHE_VERSION
                and data.get("extractor") == current_extractor_fingerprint()
                and isinstance(data.get("entries"), dict)
            ):
                cache.entries = data["entries"]
        except (OSError, orjson.JSONDecodeError, AttributeError):
            pass
        return cache

    def get(self, path: str, source_hash: str) -> ExtractResult | None:
        """Return the cached result when its source hash matches."""
        entry = self.entries.get(path)
        if (
            not isinstance(entry, dict)
            or entry.get("source_hash") != source_hash
        ):
            return None
        try:
            return ExtractResult(
                nodes=[NodeV1.from_dict(node) for node in entry["nodes"]],
                raw_edges=[RawEdge(**edge) for edge in entry["raw_edges"]],
            )
        except (KeyError, TypeError, ValueError):
            self.entries.pop(path, None)
            self.dirty = True
            return None

    def set(self, path: str, source_hash: str, result: ExtractResult) -> None:
        """Store an extraction result under its path and hash."""
        self.entries[path] = {
            "source_hash": source_hash,
            "nodes": [node.to_dict() for node in result.nodes],
            "raw_edges": [asdict(edge) for edge in result.raw_edges],
        }
        self.dirty = True

    def retain(self, paths: Container[str]) -> None:
        """Discard entries for paths absent from the current build."""
        retained = {
            path: entry for path, entry in self.entries.items() if path in paths
        }
        if len(retained) != len(self.entries):
            self.entries = retained
            self.dirty = True

    def save(self) -> None:
        """Atomically write cache entries to disk if they changed."""
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "version": CACHE_VERSION,
            "extractor": current_extractor_fingerprint(),
            "entries": self.entries,
        }
        temporary_path.write_bytes(orjson.dumps(payload))
        os.replace(temporary_path, self.path)
        self.dirty = False
