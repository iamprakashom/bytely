"""Human- and agent-readable outputs rendered from the graph.

`.graph/wiring.json` is the source of truth; every output here is derived
from it and can be regenerated at any time. Each format is an `OutputWriter`
registered in
`WRITERS`, so a new format is one class plus one registry entry. The default
is `markdown`: one card per source file, mirroring the source tree, plus an
`INDEX.md` at the root of the output directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bytely.graph.types import GraphV1, NodeV1

DEFAULT_FORMATS: tuple[str, ...] = ("markdown",)
MAX_SIGNATURE_CHARS = 200
# The root index; no card may take its name (see `card_paths`).
INDEX_FILE = "INDEX.md"
# Cards this writer created, so a rebuild deletes only its own stale files and
# never a user's markdown when the output directory is shared (`--dir`).
CARD_MANIFEST = Path("cache") / "cards.json"
# Each written file's content hash, size, and mtime: a file whose content
# and stat are unchanged is not read back to compare.
CARD_STAMPS = Path("cache") / "card-stamps.json"


class OutputWriter(Protocol):
    """Renders a graph into files under an output directory."""

    def write(self, graph: GraphV1, output_dir: Path) -> None:
        """Write this format's files for `graph` into `output_dir`."""
        ...


class MarkdownWriter:
    """One markdown card per source file, plus `INDEX.md`."""

    def write(self, graph: GraphV1, output_dir: Path) -> None:
        """Write changed cards and the index; delete cards of removed files."""
        by_path: dict[str, list[NodeV1]] = {}
        file_nodes: dict[str, NodeV1] = {}
        languages: Counter[str] = Counter()
        for node in graph.nodes:
            if node.kind == "file":
                by_path.setdefault(node.path, [])
                file_nodes[node.path] = node
                languages[PurePosixPath(node.path).suffix or node.name] += 1
            else:
                by_path.setdefault(node.path, []).append(node)

        concepts = _concept_nodes(output_dir)
        cards = card_paths(by_path)
        concept_files = concept_paths(concepts, set(cards.values()))
        uplinks: dict[str, list[str]] = {}
        for concept in concepts:
            for source in concept["sources"]:
                uplinks.setdefault(source["path"], []).append(concept["slug"])
        files = {
            card: render_card(
                source,
                by_path[source],
                file_nodes.get(source),
                uplinks.get(source, []),
            )
            for source, card in cards.items()
        }
        for concept in concepts:
            name = concept_files[concept["slug"]]
            files[name] = render_concept(
                concept, by_path, _existing(output_dir / name)
            )
        previous = _read_manifest(output_dir)
        for name in sorted(previous - set(files)):
            if name.startswith(f"{CONCEPT_DIR}/"):
                orphan = render_orphan(name, _existing(output_dir / name))
                if orphan is not None:
                    files[name] = orphan
        files[INDEX_FILE] = render_index(
            len(cards), graph, languages, concepts, concept_files
        )

        conflicts = sorted(
            name
            for name, text in files.items()
            if name not in previous
            and (output_dir / name).exists()
            and (output_dir / name).read_bytes() != text.encode("utf-8")
        )
        if conflicts:
            shown = ", ".join(conflicts[:5])
            extra = len(conflicts) - 5
            more = f" and {extra} more" if extra > 0 else ""
            raise FileExistsError(
                f"Not overwriting files bytely did not create in {output_dir}: "
                f"{shown}{more}. Use an empty or dedicated output directory."
            )

        stamps = _read_stamps(output_dir)
        written: dict[str, list[Any]] = {}
        for name, text in files.items():
            written[name] = _write_stamped(
                output_dir / name, text, stamps.get(name)
            )
        _remove_stale_cards(output_dir, previous, set(files))
        if written != stamps:
            _write_if_changed(
                output_dir / CARD_STAMPS, json.dumps(written, sort_keys=True)
            )


WRITERS: dict[str, OutputWriter] = {"markdown": MarkdownWriter()}


def write_outputs(
    graph: GraphV1, output_dir: str, formats: Iterable[str] = DEFAULT_FORMATS
) -> None:
    """Render `graph` in each requested format."""
    for name in formats:
        writer = WRITERS.get(name)
        if writer is None:
            raise ValueError(
                f"Unknown output format {name!r}; "
                f"available: {', '.join(sorted(WRITERS))}"
            )
        writer.write(graph, Path(output_dir))


def card_paths(sources: Iterable[str]) -> dict[str, str]:
    """Map each source path to its card path, relative to the output root.

    `src/app.py` gets `src/app.md`, as a card mirrors its source file. When
    two sources would share a card (`a.js` and `a.ts`), both keep their
    extension instead (`a.js.md`, `a.ts.md`), so the mapping stays one to
    one and does not depend on processing order.

    Names are compared case-insensitively, since on Windows and macOS
    `Util.md` and `util.md` are one file, and the root `INDEX.md` is taken:
    a root `index.js` gets `index.js.md` rather than overwriting the index.
    """
    paths = sorted(sources)
    stems = Counter(_stem_card(path).casefold() for path in paths)
    stems[INDEX_FILE.casefold()] += 1
    return {
        path: _stem_card(path)
        if stems[_stem_card(path).casefold()] == 1
        else f"{path}.md"
        for path in paths
    }


def one_liner(node: NodeV1) -> str:
    """A node's meaning in one line: its summary if any, else its signature."""
    if node.summary and node.summary.strip():
        return node.summary.strip().split("\n")[0].strip()
    if not node.signature:
        return ""
    signature = " ".join(node.signature.split())
    if len(signature) > MAX_SIGNATURE_CHARS:
        signature = signature[: MAX_SIGNATURE_CHARS - 1] + "…"
    return signature


def render_card(
    source: str,
    nodes: list[NodeV1],
    file_node: NodeV1 | None = None,
    concepts: list[str] | None = None,
) -> str:
    """A file's card.

    Its path (and the concepts that cite it), its summary, then one line
    per definition in source order.
    """
    links = " ".join(f"[[{slug}]]" for slug in sorted(set(concepts or [])))
    lines = [f"# {source} · {links}" if links else f"# {source}", ""]
    if file_node is not None and file_node.summary:
        lines += [one_liner(file_node), ""]
    for node in nodes:
        line = f"- {node.name} · {node.kind} · {node.span}"
        meaning = one_liner(node)
        if meaning:
            line += f" — {meaning}"
        lines.append(line)
    return "\n".join(lines) + "\n"


CONCEPT_DIR = "concepts"
GENERATED_START = "<!-- bytely:generated:start -->"
GENERATED_END = "<!-- bytely:generated:end -->"
DEFAULT_NOTES = (
    "\n## Notes\n\n_Anything written below the generated block is kept "
    "when the graph is rebuilt._\n"
)


def concept_paths(
    concepts: list[dict[str, Any]], taken: set[str]
) -> dict[str, str]:
    """Each concept's file, never clashing with a file card."""
    out = {}
    for concept in concepts:
        name = f"{CONCEPT_DIR}/{concept['slug']}.md"
        if name in taken:
            name = f"{CONCEPT_DIR}/{concept['slug']}.concept.md"
        out[concept["slug"]] = name
    return out


def _scalar(value: object) -> str:
    if isinstance(value, list):
        return "[]"
    if isinstance(value, dict):
        return "{}"
    return json.dumps(value, ensure_ascii=False)


def _yaml(value: object, indent: str = "") -> list[str]:
    """Minimal YAML for frontmatter (strings as JSON, which YAML reads)."""
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{indent}{key}:")
                lines += _yaml(item, indent + "  ")
            else:
                lines.append(f"{indent}{key}: {_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            nested = _yaml(item, indent + "  ")
            lines.append(f"{indent}- {nested[0].lstrip()}")
            lines += nested[1:]
        return lines
    return [f"{indent}{_scalar(value)}"]


def render_concept(
    concept: dict[str, Any],
    by_path: dict[str, list[NodeV1]],
    existing: str | None,
) -> str:
    """A concept node.

    Frontmatter (with the symbols it covers, at exact spans), the generated
    summary and links, then the user's notes, kept verbatim.
    """
    covers = [
        {
            "symbol": node.name,
            "kind": node.kind,
            "at": f"{node.path}:{node.span}",
        }
        for path in sorted({s["path"] for s in concept["sources"]})
        for node in by_path.get(path, [])
    ]
    front = {
        "name": concept["name"],
        "slug": concept["slug"],
        "type": concept["type"],
        "sources": concept["sources"],
        "links": concept["links"],
        "covers": covers,
    }
    summary = str(concept.get("summary") or "").strip() or "_(no summary)_"
    body = ["## Summary", "", summary, ""]
    if concept["links"]:
        body += ["## Related", ""]
        for link in concept["links"]:
            description = link.get("description")
            tail = f" — {description}" if description else ""
            relation = link["relation"].replace("_", " ")
            body.append(f"- {relation} [[{link['to']}]]{tail}")
        body.append("")
    notes = DEFAULT_NOTES
    if existing is not None and GENERATED_END in existing:
        notes = existing.split(GENERATED_END, 1)[1].removeprefix("\n")
    return (
        "---\n"
        + "\n".join(_yaml(front))
        + "\n---\n"
        + f"{GENERATED_START}\n"
        + "\n".join(body)
        + f"{GENERATED_END}\n"
        + notes
    )


ORPHAN_NOTE = (
    "_This concept is no longer in the graph. The page is kept for the "
    "notes below; delete it once they are no longer needed._"
)


def render_orphan(name: str, existing: str | None) -> str | None:
    """A removed concept's page, kept only if the user wrote notes on it.

    Returns None (delete the page) when its notes are still the default.
    """
    if existing is None or GENERATED_END not in existing:
        return None
    notes = existing.split(GENERATED_END, 1)[1].removeprefix("\n")
    if notes.replace("\r\n", "\n").strip() == DEFAULT_NOTES.strip():
        return None
    slug = PurePosixPath(name).stem.removesuffix(".concept")
    front = {"slug": slug, "orphaned": True}
    return (
        "---\n"
        + "\n".join(_yaml(front))
        + "\n---\n"
        + f"{GENERATED_START}\n## {slug} (orphaned)\n\n{ORPHAN_NOTE}\n"
        + f"{GENERATED_END}\n"
        + notes
    )


def _concept_nodes(output_dir: Path) -> list[dict[str, Any]]:
    from bytely.ai.concepts import read_concepts

    data = read_concepts(output_dir)
    return [
        node
        for node in (data or {}).get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("slug"), str)
    ]


def _existing(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def render_index(
    card_count: int,
    graph: GraphV1,
    languages: Counter[str],
    concepts: list[dict[str, Any]] | None = None,
    concept_files: dict[str, str] | None = None,
) -> str:
    """The root index: what the cards are and how to find one."""
    symbols = sum(node.kind != "file" for node in graph.nodes)
    by_extension = ", ".join(
        f"{extension} {count}" for extension, count in sorted(languages.items())
    )
    concept_lines = ""
    if concepts:
        listed = []
        for concept in concepts:
            sources = ", ".join(s["path"] for s in concept["sources"])
            where = (concept_files or {}).get(
                concept["slug"], f"{CONCEPT_DIR}/{concept['slug']}.md"
            )
            listed.append(
                f"- [{concept['slug']}]({where}) — {concept['name']}"
                + (f" · {sources}" if sources else "")
            )
        concept_lines = "\n## Concepts\n\n" + "\n".join(listed) + "\n"
    return (
        "# bytely — repo map\n"
        "\n"
        "One markdown card per source file, mirroring the source tree: "
        "`src/app.py` is\n"
        "described by `src/app.md` here. Each card lists the file's "
        "definitions in\n"
        "source order as `name · kind · span — signature`, with exact line "
        "spans.\n"
        "Search this folder for a symbol or file name to find its card; open "
        "the\n"
        "source only at the span you need.\n"
        "\n"
        "Relationships (imports, calls, inheritance) are in "
        "`.graph/wiring.json`,\n"
        "not in the cards.\n"
        "\n"
        "## Contents\n"
        "\n"
        f"- {card_count} cards, {symbols} definitions, "
        f"{len(graph.edges)} relationships\n"
        f"- Files by extension: {by_extension or 'none'}\n" + concept_lines
    )


def _stem_card(path: str) -> str:
    pure = PurePosixPath(path)
    return str(pure.with_suffix(".md")) if pure.suffix else f"{path}.md"


def _write_if_changed(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    try:
        if path.read_bytes() == data:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_stamped(path: Path, text: str, stamp: Any) -> list[Any]:
    """Write `text` unless the stamp shows the file already holds it.

    Returns the file's new stamp: [content hash, size, mtime_ns]. A file
    edited by hand since its stamp has a different stat, so it is read and
    rewritten as before.
    """
    data = text.encode("utf-8")
    digest = hashlib.sha1(data, usedforsecurity=False).hexdigest()
    try:
        stat = os.stat(path)
        if stamp == [digest, stat.st_size, stat.st_mtime_ns]:
            return stamp  # type: ignore[no-any-return]
    except OSError:
        pass
    _write_if_changed(path, text)
    stat = os.stat(path)
    return [digest, stat.st_size, stat.st_mtime_ns]


def outputs_intact(output_dir: Path, formats: Iterable[str]) -> bool:
    """Whether every file the last write produced is still as written.

    Compares each file's size and mtime with its stamp, without reading it,
    so a card deleted or edited by hand makes the next build rewrite it.
    """
    if "markdown" not in formats:
        return True
    stamps = _read_stamps(output_dir)
    if not stamps:
        return False
    for name, stamp in stamps.items():
        try:
            stat = os.stat(output_dir / name)
        except OSError:
            return False
        if not isinstance(stamp, list) or stamp[1:] != [
            stat.st_size,
            stat.st_mtime_ns,
        ]:
            return False
    return True


def _read_stamps(output_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads((output_dir / CARD_STAMPS).read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_manifest(output_dir: Path) -> set[str]:
    try:
        return set(json.loads((output_dir / CARD_MANIFEST).read_text("utf-8")))
    except (OSError, ValueError, TypeError):
        return set()


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _remove_stale_cards(
    output_dir: Path, previous: set[str], current: set[str]
) -> None:
    manifest = output_dir / CARD_MANIFEST
    by_folded = {name.casefold(): name for name in current}
    for card in sorted(previous - current):
        path = output_dir / card
        # On a case-insensitive filesystem a stale `index.md` may be the
        # very file just written as `INDEX.md` (or `util.md` as `Util.md`);
        # deleting it would delete the current one.
        twin = by_folded.get(card.casefold())
        if twin is not None and _same_file(path, output_dir / twin):
            continue
        try:
            path.unlink()
        except OSError:
            continue
        # Drop directories this left empty, up to the output root.
        parent = path.parent
        while parent != output_dir and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_if_changed(manifest, json.dumps(sorted(current), indent=0) + "\n")
