"""`bytely grep`: every match in the indexed files, grouped by symbol.

Unlike `ask`, which ranks and returns the top few answers, grep is
exhaustive over the files the graph indexes. Each hit is attributed to the
innermost definition whose span contains it (or to the file when it is at
module level), and groups are ordered by how often that definition is
referenced, so the most connected code comes first.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from bytely.graph.map import reference_counts
from bytely.query.common import SourceReader, location, span_lines

if TYPE_CHECKING:
    from bytely.graph.types import GraphV1, NodeV1

MAX_GROUPS = 60
MAX_LINES_PER_GROUP = 8
MAX_LINE_CHARS = 160


def render_grep(
    graph: GraphV1,
    reader: SourceReader,
    pattern: str,
    *,
    fixed: bool = False,
    ignore_case: bool = False,
    scope: str | None = None,
) -> str:
    """Matches grouped by enclosing symbol, most-referenced groups first."""
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(re.escape(pattern) if fixed else pattern, flags)
    except re.error as error:
        raise ValueError(f"Invalid pattern {pattern!r}: {error}") from error

    prefix = scope.replace("\\", "/").strip("/") + "/" if scope else ""
    files = sorted(
        node.path
        for node in graph.nodes
        if node.kind == "file" and node.path.startswith(prefix)
    )
    definitions: dict[str, list[tuple[int, int, NodeV1]]] = {}
    file_nodes: dict[str, NodeV1] = {}
    for node in graph.nodes:
        if node.kind == "file":
            file_nodes[node.path] = node
        else:
            start, end = span_lines(node.span)
            definitions.setdefault(node.path, []).append((start, end, node))

    groups: dict[str, tuple[NodeV1, list[tuple[int, str]]]] = {}
    for path in files:
        lines = reader.lines(path)
        spans = definitions.get(path, [])
        for number, line in enumerate(lines, start=1):
            if not regex.search(line):
                continue
            enclosing = min(
                (
                    (end - start, node)
                    for start, end, node in spans
                    if start <= number <= end
                ),
                default=(0, file_nodes[path]),
                key=lambda item: (item[0], item[1].id),
            )[1]
            groups.setdefault(enclosing.id, (enclosing, []))[1].append(
                (number, line.strip())
            )

    counts = reference_counts(graph)
    ordered = sorted(
        groups.values(),
        key=lambda group: (
            -counts[group[0].id],
            group[0].path,
            span_lines(group[0].span),
            group[0].id,
        ),
    )
    hits = sum(len(matches) for _, matches in ordered)
    matched_files = len({node.path for node, _ in ordered})
    header = (
        f"{pattern!r} — {hits} hits in {len(ordered)} symbols across "
        f"{matched_files} files (searched {len(files)} indexed files)"
    )
    out = [header]
    for node, matches in ordered[:MAX_GROUPS]:
        label = (
            f"{node.path} (module level)"
            if node.kind == "file"
            else f"{node.name} · {node.kind} · {location(node)}"
        )
        out.append("")
        out.append(f"{label} · {counts[node.id]} in-edges")
        for number, text in matches[:MAX_LINES_PER_GROUP]:
            if len(text) > MAX_LINE_CHARS:
                text = text[: MAX_LINE_CHARS - 1] + "…"
            out.append(f"  L{number}: {text}")
        if len(matches) > MAX_LINES_PER_GROUP:
            out.append(f"  … {len(matches) - MAX_LINES_PER_GROUP} more")
    if len(ordered) > MAX_GROUPS:
        out.append("")
        out.append(
            f"… {len(ordered) - MAX_GROUPS} more symbols not shown; "
            "narrow the pattern or use --in"
        )
    unreadable = sorted(reader.unreadable.intersection(files))
    if unreadable:
        out.append("")
        out.append(f"could not read: {', '.join(unreadable[:10])}")
    return "\n".join(out) + "\n"
