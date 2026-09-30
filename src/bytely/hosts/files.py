"""Edits bytely makes to other tools' files, and how to undo each one.

Two invariants hold for every function here:

1. Never touch what bytely did not write. In a file the user owns, only the
   marker-fenced block, the `bytely` key, or the `[mcp_servers.bytely]` table
   is changed; everything else is preserved (markdown and TOML byte
   for byte; JSON content exactly, though the file is re-indented). A
   file that cannot be parsed is reported and left alone.
2. Never leave an empty shell. A file that held nothing but bytely's
   contribution is deleted on removal, not truncated to `{}` or a blank file.

Every function takes `apply`: with `False` it only reports what it would do,
which is how `--dry-run` works.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

Action = Literal[
    "created",
    "updated",
    "unchanged",
    "removed",
    "deleted",
    "absent",
    "skipped-unparseable",
    "skipped-foreign",
]

SERVER_KEY = "bytely"
START = "<!-- bytely:start -->"
END = "<!-- bytely:end -->"
TOML_HEADER = f"[mcp_servers.{SERVER_KEY}]"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _eol(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def remove_file(
    path: Path, apply: bool, *, prune: bool | Path = True
) -> Action:
    """Delete a file bytely owns.

    `prune=True` also removes directories the deletion leaves empty. A
    directory for `prune` removes them only below it: outside the
    repository that is the tool's own folder (`~/.gemini`), since an empty
    `~/.gemini/config` may be how a tool is detected. `False` removes none.
    """
    if not path.is_file():
        return "absent"
    if apply:
        path.unlink()
        if prune is not False:
            _prune_empty_dirs(path.parent, None if prune is True else prune)
    return "deleted"


def _prune_empty_dirs(directory: Path, stop: Path | None) -> None:
    for _ in range(6):
        if stop is not None and (
            directory == stop or stop not in directory.parents
        ):
            return
        try:
            directory.rmdir()  # only succeeds when empty
        except OSError:
            return
        directory = directory.parent


def write_owned(path: Path, content: str, apply: bool) -> Action:
    """Write a file bytely owns outright (rewritten on every run)."""
    if path.is_file() and _read(path) == content:
        return "unchanged"
    action: Action = "updated" if path.exists() else "created"
    if apply:
        _write(path, content)
    return action


def _block(body: str, eol: str) -> str:
    text = f"{START}\n{body.replace(chr(13), '').rstrip()}\n{END}"
    return text if eol == "\n" else text.replace("\n", eol)


def upsert_section(path: Path, body: str, apply: bool) -> Action:
    """Put `body` between bytely's markers in a file the user owns.

    The block is replaced in place if present, else appended after one blank
    line; the file's line endings are kept.
    """
    if not path.exists():
        if apply:
            _write(path, _block(body, "\n") + "\n")
        return "created"
    text = _read(path)
    eol = _eol(text)
    lines = text.replace("\r\n", "\n").split("\n")
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == START), -1
    )
    end = next(
        (
            i
            for i, line in enumerate(lines)
            if i > start and start != -1 and line.strip() == END
        ),
        -1,
    )
    if start != -1 and end != -1:
        current = "\n".join(lines[start : end + 1])
        if current == _block(body, "\n"):
            return "unchanged"
        block = _block(body, "\n").split("\n")
        updated = lines[:start] + block + lines[end + 1 :]
        if apply:
            _write(path, eol.join(updated))
        return "updated"
    separator = (
        "" if text.endswith(eol * 2) else eol if text.endswith(eol) else eol * 2
    )
    if apply:
        _write(path, f"{text}{separator}{_block(body, eol)}{eol}")
    return "updated"


def strip_section(path: Path, apply: bool) -> Action:
    """Remove bytely's fenced block; delete the file if nothing else is left."""
    if not path.is_file():
        return "absent"
    text = _read(path)
    if START not in text:
        return "absent"
    eol = _eol(text)
    kept: list[str] = []
    inside = False
    found = False
    for line in text.replace("\r\n", "\n").split("\n"):
        if not inside and line.strip() == START:
            inside = found = True
            continue
        if inside:
            if line.strip() == END:
                inside = False
            continue
        kept.append(line)
    if not found:
        return "absent"
    rest = "\n".join(kept)
    while "\n\n\n" in rest:
        rest = rest.replace("\n\n\n", "\n\n")
    rest = rest.strip("\n")
    if not rest.strip():
        return remove_file(path, apply)
    if apply:
        _write(path, (rest + "\n").replace("\n", eol))
    return "removed"


def _load_json(path: Path) -> dict[str, Any] | None | Literal["unparseable"]:
    if not path.exists():
        return None
    try:
        data = json.loads(_read(path))
    except (OSError, UnicodeDecodeError, ValueError):
        return "unparseable"
    return data if isinstance(data, dict) else "unparseable"


def _dump_json(path: Path, data: dict[str, Any]) -> None:
    _write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def merge_json_key(
    path: Path, top_key: str, entry: dict[str, Any], apply: bool
) -> Action:
    """Set `<top_key>.bytely` in a JSON config, keeping every other entry."""
    loaded = _load_json(path)
    if loaded == "unparseable":
        return "skipped-unparseable"
    data: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    bucket = data.setdefault(top_key, {})
    if not isinstance(bucket, dict):
        return "skipped-unparseable"
    if bucket.get(SERVER_KEY) == entry:
        return "unchanged"
    action: Action = "updated" if loaded is not None else "created"
    bucket[SERVER_KEY] = entry
    if apply:
        _dump_json(path, data)
    return action


def remove_json_key(
    path: Path, top_key: str, apply: bool, *, prune: bool | Path = True
) -> Action:
    """Delete `<top_key>.bytely`, and the file if that was all it held."""
    loaded = _load_json(path)
    if loaded is None:
        return "absent"
    if loaded == "unparseable":
        return "skipped-unparseable"
    bucket = loaded.get(top_key)
    if not isinstance(bucket, dict) or SERVER_KEY not in bucket:
        return "absent"
    del bucket[SERVER_KEY]
    if not bucket:
        del loaded[top_key]
    if not loaded:
        return remove_file(path, apply, prune=prune)
    if apply:
        _dump_json(path, loaded)
    return "removed"


def edit_json(
    path: Path,
    change: Callable[[dict[str, Any]], Action | None],
    apply: bool,
    *,
    removing: bool = False,
    prune: bool | Path = True,
) -> Action:
    """Apply `change` to a JSON config in place, keeping everything else.

    `change` edits the object it is given, or returns an action to stop
    with (e.g. `skipped-foreign`) and leave the file alone. When removing,
    a file left empty is deleted rather than kept as `{}`.
    """
    loaded = _load_json(path)
    if loaded == "unparseable":
        return "skipped-unparseable"
    if loaded is None and removing:
        return "absent"
    original: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    data = copy.deepcopy(original)
    verdict = change(data)
    if verdict is not None:
        return verdict
    if data == original:
        return "absent" if removing else "unchanged"
    if removing:
        if not data:
            return remove_file(path, apply, prune=prune)
        if apply:
            _dump_json(path, data)
        return "removed"
    if apply:
        _dump_json(path, data)
    return "updated" if loaded is not None else "created"


def _strip_toml(text: str) -> tuple[str, bool]:
    """The TOML text without bytely's table (header to the next header)."""
    lines = text.replace("\r\n", "\n").split("\n")
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == TOML_HEADER), -1
    )
    if start == -1:
        return text.replace("\r\n", "\n"), False
    end = start + 1
    while end < len(lines) and not lines[end].lstrip().startswith("["):
        end += 1
    rest = "\n".join(lines[:start] + lines[end:])
    while "\n\n\n" in rest:
        rest = rest.replace("\n\n\n", "\n\n")
    return rest.lstrip("\n"), True


def upsert_toml_server(
    path: Path, command: str, args: list[str], apply: bool
) -> Action:
    """Replace or append `[mcp_servers.bytely]`, keeping every other table.

    Line-based on purpose: parsing and re-serialising would reformat the
    user's whole file.
    """
    existed = path.exists()
    text = _read(path) if existed else ""
    eol = _eol(text)
    section = (
        f"{TOML_HEADER}\ncommand = {json.dumps(command)}\n"
        f"args = [{', '.join(json.dumps(arg) for arg in args)}]\n"
    )
    rest, _ = _strip_toml(text)
    if rest.strip():
        if rest.endswith("\n\n"):
            separator = ""
        elif rest.endswith("\n"):
            separator = "\n"
        else:
            separator = "\n\n"
        updated = f"{rest}{separator}{section}"
    else:
        updated = section
    if text.replace("\r\n", "\n") == updated:
        return "unchanged"
    if apply:
        _write(path, updated.replace("\n", eol))
    return "updated" if existed else "created"


def remove_toml_server(
    path: Path, apply: bool, *, prune: bool | Path = True
) -> Action:
    """Delete `[mcp_servers.bytely]`, and the file if nothing else is left."""
    if not path.is_file():
        return "absent"
    original = _read(path)
    rest, found = _strip_toml(original)
    if not found:
        return "absent"
    if not rest.strip():
        return remove_file(path, apply, prune=prune)
    rest = rest if rest.endswith("\n") else rest + "\n"
    if apply:
        _write(path, rest.replace("\n", _eol(original)))
    return "removed"


IGNORE_COMMENT = "# bytely: local, regenerable code graph"
IGNORE_ENTRY = "/bytely/"


def add_ignore_entry(path: Path, apply: bool) -> Action:
    """Git-ignore the `bytely/` output folder, once."""
    existed = path.exists()
    text = _read(path) if existed else ""
    entries = {line.strip() for line in text.splitlines()}
    if entries & {IGNORE_ENTRY, "bytely/", "/bytely", "bytely"}:
        return "unchanged"
    eol = _eol(text)
    separator = "" if not text or text.endswith("\n") else eol
    if apply:
        _write(
            path, f"{text}{separator}{IGNORE_COMMENT}{eol}{IGNORE_ENTRY}{eol}"
        )
    return "updated" if existed else "created"


def remove_ignore_entry(path: Path, apply: bool) -> Action:
    """Remove the lines `add_ignore_entry` wrote."""
    if not path.is_file():
        return "absent"
    text = _read(path)
    lines = text.splitlines()
    kept = [
        line
        for line in lines
        if line.strip() not in (IGNORE_COMMENT, IGNORE_ENTRY)
    ]
    if len(kept) == len(lines):
        return "absent"
    rest = "\n".join(kept).strip("\n")
    if not rest.strip():
        return remove_file(path, apply)
    if apply:
        _write(path, (rest + "\n").replace("\n", _eol(text)))
    return "removed"
