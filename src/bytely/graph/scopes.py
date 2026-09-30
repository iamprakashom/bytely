"""Project boundary discovery for monorepos."""

from __future__ import annotations

import os
from pathlib import Path

import pathspec

from bytely.graph.types import ScopeV1

PROJECT_MARKERS = (
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "composer.json",
    "Gemfile",
    "mix.exs",
)
IGNORED_DIRECTORIES = {".git", "node_modules", "__pycache__", ".venv", "venv"}


def discover_scopes(root: str) -> list[ScopeV1]:
    """Return deterministic project scopes marked by common build manifests."""
    root_path = Path(root).resolve()
    scopes: list[ScopeV1] = []
    gitignore_path = root_path / ".gitignore"
    try:
        patterns = (
            gitignore_path.read_text("utf-8").splitlines()
            if gitignore_path.exists()
            else []
        )
    except OSError:
        patterns = []
    patterns.extend(f"{name}/" for name in IGNORED_DIRECTORIES)
    ignore_spec = pathspec.PathSpec.from_lines("gitignore", patterns)

    for current_dir, directory_names, file_names in os.walk(root_path):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not ignore_spec.match_file(
                f"{Path(current_dir).relative_to(root_path).as_posix()}/{name}/"
                if Path(current_dir) != root_path
                else f"{name}/"
            )
        )
        current_path = Path(current_dir)
        relative = current_path.relative_to(root_path).as_posix()
        marker_names = sorted(
            name
            for name in set(file_names).intersection(PROJECT_MARKERS)
            if not ignore_spec.match_file(
                f"{relative}/{name}" if relative != "." else name
            )
        )
        if not marker_names:
            continue

        prefix = "" if relative == "." else f"{relative}/"
        label = root_path.name if not prefix else current_path.name
        scopes.append(ScopeV1(prefix=prefix, label=label, markers=marker_names))

    return sorted(scopes, key=lambda scope: scope.prefix)
