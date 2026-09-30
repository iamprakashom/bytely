"""Walk repository files while respecting ignore rules.

Every query refreshes the graph, and a refresh walks the whole tree, so the
walk works on plain strings: a `Path` object per file costs more than the
directory listing itself.
"""

import os
from collections.abc import Callable
from pathlib import Path

from pathspec import GitIgnoreSpec

_BUILT_IN_IGNORES = [
    ".git/",
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "venv/",
]

# (directory the rules are relative to, rules), outermost first.
_Rules = list[tuple[str, GitIgnoreSpec]]


def _ignore_spec(directory: str, *, root: bool) -> GitIgnoreSpec | None:
    try:
        with open(
            os.path.join(directory, ".gitignore"), encoding="utf-8"
        ) as handle:
            patterns = handle.read().splitlines()
    except OSError:
        patterns = []
    if root:
        patterns.extend(_BUILT_IN_IGNORES)
    return GitIgnoreSpec.from_lines(patterns) if patterns else None


def _is_ignored(path: str, rules: _Rules, *, is_directory: bool) -> bool:
    ignored = False
    for base, spec in rules:
        relative = path[len(base.rstrip(os.sep)) + 1 :].replace(os.sep, "/")
        if is_directory:
            relative += "/"
        result = spec.check_file(relative)
        if result.index is not None:
            ignored = bool(result.include)
    return ignored


def walk_dir(
    root: str,
    extensions: list[str],
    *,
    include_file: Callable[[Path], bool] | None = None,
) -> list[str]:
    """Return matching absolute POSIX paths below a directory.

    The walk respects nested ``.gitignore`` files and built-in exclusions.
    """
    root_dir = str(Path(root).resolve())
    ext_set = set(extensions)
    results = []
    rules_by_directory: dict[str, _Rules] = {}

    for current_dir, directory_names, file_names in os.walk(root_dir):
        prefix = current_dir.rstrip(os.sep) + os.sep
        rules = list(rules_by_directory.get(os.path.dirname(current_dir), []))
        local_spec = _ignore_spec(current_dir, root=current_dir == root_dir)
        if local_spec:
            rules.append((current_dir, local_spec))
        rules_by_directory[current_dir] = rules

        directory_names[:] = sorted(
            name
            for name in directory_names
            if not _is_ignored(prefix + name, rules, is_directory=True)
        )

        for name in sorted(file_names):
            path = prefix + name
            if os.path.splitext(name)[1] not in ext_set and not (
                include_file and include_file(Path(path))
            ):
                continue
            if not _is_ignored(path, rules, is_directory=False):
                results.append(path.replace(os.sep, "/"))

    return sorted(results)
