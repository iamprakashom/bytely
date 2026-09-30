"""Explicit source-file selection for graph builds."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pathspec

if TYPE_CHECKING:
    from collections.abc import Iterable


def filter_source_files(
    root: str | Path,
    paths: Iterable[str],
    include_patterns: Iterable[str] = (),
    exclude_patterns: Iterable[str] = (),
) -> list[str]:
    """Filter paths with repository-relative gitignore-style patterns."""
    includes = tuple(include_patterns)
    excludes = tuple(exclude_patterns)
    if not includes and not excludes:
        return list(paths)
    root_path = Path(root).resolve()
    include_spec = (
        pathspec.PathSpec.from_lines("gitignore", includes)
        if includes
        else None
    )
    exclude_spec = (
        pathspec.PathSpec.from_lines("gitignore", excludes)
        if excludes
        else None
    )

    selected = []
    for path in paths:
        # Walked paths are already under the resolved root; resolving each
        # one is a system call per file, so only fall back to it if needed.
        try:
            relative_path = Path(path).relative_to(root_path).as_posix()
        except ValueError:
            relative_path = (
                Path(path).resolve().relative_to(root_path).as_posix()
            )
        if include_spec and not include_spec.match_file(relative_path):
            continue
        if exclude_spec and exclude_spec.match_file(relative_path):
            continue
        selected.append(path)
    return selected
