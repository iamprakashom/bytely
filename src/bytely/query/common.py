"""Helpers shared by the query commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bytely.graph.types import GraphV1, NodeV1

# Relations that describe structure rather than use: a query about who uses
# a symbol skips them.
STRUCTURAL_RELATIONS = frozenset({"contains", "imports"})


def span_lines(span: str) -> tuple[int, int]:
    """`"L12-L40"` → `(12, 40)`."""
    start, _, end = span.partition("-")
    return int(start.lstrip("L")), int((end or start).lstrip("L"))


def location(node: NodeV1) -> str:
    """`path:Lstart-Lend`."""
    return f"{node.path}:{node.span}"


def describe(node: NodeV1) -> str:
    """`name · kind · path:span`."""
    return f"{node.name} · {node.kind} · {location(node)}"


def qualified_name(node: NodeV1) -> str:
    """The node's ID without its path: `Widget.render`, `helper~2`."""
    return node.id.split("#", 1)[1] if "#" in node.id else node.id


def find_symbols(graph: GraphV1, symbol: str) -> list[NodeV1]:
    """Nodes a user-typed symbol names, best matches only.

    Tried in order, and the first tier with any match wins: an exact node ID
    (`src/app.py#main`, or a file path); an exact qualified name
    (`Widget.render`); a qualified-name suffix (`Inner.render` matches
    `Widget.Inner.render`); then a bare name (`render`). `~N` collision
    suffixes are ignored when matching.
    """
    nodes = graph.nodes
    exact = [node for node in nodes if node.id == symbol]
    if exact:
        return exact

    def base(name: str) -> str:
        return name.split("~", 1)[0]

    qualified = [
        node for node in nodes if base(qualified_name(node)) == symbol
    ]
    if qualified:
        return sorted(qualified, key=lambda node: node.id)
    suffix = [
        node
        for node in nodes
        if node.kind != "file"
        and base(qualified_name(node)).endswith("." + symbol)
    ]
    if suffix:
        return sorted(suffix, key=lambda node: node.id)
    return sorted(
        (node for node in nodes if node.name == symbol),
        key=lambda node: node.id,
    )


def find_file(graph: GraphV1, root: Path, file: str) -> NodeV1 | None:
    """The file node a user-typed path names.

    Accepts a path relative to the repository root, an absolute path (or one
    relative to the current directory that exists), or a unique path suffix
    (`graph/build.py`). A path that leaves the tree (`../x.py`) matches
    nothing.
    """
    files = {node.path: node for node in graph.nodes if node.kind == "file"}
    candidate = file.replace("\\", "/")
    on_disk = Path(file)
    if on_disk.is_absolute() or on_disk.exists():
        try:
            resolved = on_disk.resolve()
        except OSError:
            resolved = None
        if resolved is not None and resolved.is_relative_to(root):
            candidate = resolved.relative_to(root).as_posix()
    # Drop only literal `./` prefixes: `lstrip("./")` would also eat the dot
    # of `.github/…` and turn `../foo.py` into `foo.py`.
    while candidate.startswith("./"):
        candidate = candidate[2:]
    if candidate.startswith("../") or candidate == "..":
        return None
    if candidate in files:
        return files[candidate]
    matches = [path for path in files if path.endswith("/" + candidate)]
    return files[matches[0]] if len(matches) == 1 else None


@dataclass
class SourceReader:
    """Reads source lines relative to a root, caching each file."""

    root: Path
    unreadable: set[str] = field(default_factory=set)
    _cache: dict[str, list[str]] = field(default_factory=dict)

    def lines(self, path: str) -> list[str]:
        """All lines of a file, without line endings (empty if unreadable)."""
        if path not in self._cache:
            try:
                data = (self.root / path).read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                data = ""
                self.unreadable.add(path)
            self._cache[path] = data.splitlines()
        return self._cache[path]

    def span(self, node: NodeV1, limit: int | None = None) -> list[str]:
        """The node's source lines, optionally only the first `limit`."""
        start, end = span_lines(node.span)
        lines = self.lines(node.path)[start - 1 : end]
        return lines if limit is None else lines[:limit]


def numbered(lines: list[str], first_line: int, indent: str = "  ") -> str:
    """Lines prefixed with their line numbers."""
    width = len(str(first_line + max(len(lines) - 1, 0)))
    return "\n".join(
        f"{indent}{first_line + offset:>{width}}  {line}"
        for offset, line in enumerate(lines)
    )
