"""Normalize repository paths and test path prefixes."""

from pathlib import Path, PurePosixPath


def rel_posix(root: str, abs_path: str) -> str:
    """Return the POSIX-style relative path from root to abs_path."""
    try:
        # pathlib.relative_to will fail if abs_path is not under root.
        # But we only use this when we know it is.
        rel = Path(abs_path).relative_to(Path(root))
        return PurePosixPath(rel).as_posix()
    except ValueError:
        return abs_path.replace("\\", "/")


def normalize_path_prefix(prefix: str) -> str:
    """Normalize a directory prefix string for matching.

    '' -> ''
    'foo' -> 'foo/'
    'foo/' -> 'foo/'
    """
    if not prefix:
        return ""
    p = prefix.replace("\\", "/")
    return p if p.endswith("/") else p + "/"


def dir_prefix_match(prefix: str, path: str) -> bool:
    """Check if the path is strictly under the directory prefix."""
    if not prefix:
        return True
    return path.replace("\\", "/").startswith(normalize_path_prefix(prefix))
