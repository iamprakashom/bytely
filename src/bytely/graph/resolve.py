"""Resolve extracted edges to graph node identifiers."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, TypeVar

from bytely.graph.extract import language_of
from bytely.graph.types import Confidence, EdgeV1, NodeV1

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bytely.graph.extract import RawEdge


_Bound = TypeVar("_Bound", NodeV1, str)

# Languages whose definitions can reach one another by bare name. A language not
# listed here is its own family (as the reference implementation groups them).
FAMILIES: tuple[tuple[str, ...], ...] = (
    ("typescript", "tsx"),
    ("java", "kotlin", "scala", "clojure"),
    ("c", "cpp"),
)
FAMILY_OF = {lang: group[0] for group in FAMILIES for lang in group}

# What a bare name, an import binding, or a module member can refer to. Never a
# method: calling one needs a receiver (`self.m()`, `obj.m()`), so a bare
# `open()` is the builtin, not some class's `open` method. The reference
# implementation likewise resolves bare calls to functions, with class
# constructors as a fallback.
NAME_BINDABLE_KINDS = frozenset({"function", "class", "struct"})

# What an import binding may name: anything a bare name can, plus the types
# a class can only inherit from (`implements Props`, a Rust trait). Calls
# still filter their bindings to NAME_BINDABLE_KINDS.
IMPORT_BINDABLE_KINDS = NAME_BINDABLE_KINDS | {"interface", "trait"}


def _family_of(path: str) -> str | None:
    lang = language_of(path)
    if lang is None:
        return None
    return FAMILY_OF.get(lang, lang)


def _reachable(file: str, candidate_path: str) -> bool:
    """Could a bare-name reference in `file` reach `candidate_path`?"""
    source_family = _family_of(file)
    if source_family is None:
        return True
    target_family = _family_of(candidate_path)
    return target_family is None or source_family == target_family


def _rust_crate_key(path: str) -> str:
    """Approximate which Rust crate a file belongs to, from Cargo's layout.

    `src/` (and everything under it, including `src/tests/`) is the
    package's library/binary crate; `src/bin/<name>` is its own binary
    crate; each file or directory directly under `tests/`, `examples/`, or
    `benches/` is its own crate. Any other file groups with its directory,
    so a flat layout (`[lib] path = "lib.rs"`) stays one crate while
    `build.rs` stays apart from `src/`.
    """
    parts = path.split("/")
    if "src" in parts:
        src = len(parts) - 1 - parts[::-1].index("src")
        if len(parts) > src + 2 and parts[src + 1] == "bin":
            return "/".join(parts[: src + 3]).removesuffix(".rs")
        return "/".join(parts[: src + 1])
    for index, part in enumerate(parts[:-1]):
        if part in RUST_CRATE_DIRS:
            return "/".join(parts[: index + 2]).removesuffix(".rs")
    return path.rpartition("/")[0]


RUST_CRATE_DIRS = ("tests", "examples", "benches")


def _rust_crates(nodes: Iterable[NodeV1]) -> dict[str, frozenset[str]]:
    """Map each Rust file to the crates it is compiled into.

    Usually one. A module directory under `tests/` (or `examples/`,
    `benches/`) that is not itself a crate root, such as `tests/common/`,
    is compiled into every crate root beside it that declares
    `mod common;`, so it belongs to all of them.
    """
    nodes = list(nodes)
    declared_by: dict[str, set[str]] = {}
    for node in nodes:
        if node.kind != "module" or node.id != f"{node.path}#{node.name}":
            continue
        root_key = _rust_crate_key(node.path)
        parts = node.path.split("/")
        is_crate_root = (
            len(parts) >= 2
            and parts[-2] in RUST_CRATE_DIRS
            and root_key == node.path.removesuffix(".rs")
        )
        if is_crate_root:
            module_dir = f"{'/'.join(parts[:-1])}/{node.name}"
            declared_by.setdefault(module_dir, set()).add(root_key)
    crates: dict[str, frozenset[str]] = {}
    for node in nodes:
        if node.kind == "file" and node.path.endswith(".rs"):
            key = _rust_crate_key(node.path)
            crates[node.path] = frozenset({key, *declared_by.get(key, ())})
    return crates


def _same_rust_crate(
    file: str, candidate_path: str, crates: dict[str, frozenset[str]]
) -> bool:
    """Whether a bare name in `file` could name an item in `candidate_path`.

    In Rust another crate's items are only visible through `use` or a path,
    and those resolve as import bindings. A bare-name guess across crates
    is therefore never right: in Serde it bound 617 `Ok`/`Err`/`Some` calls
    to the unit structs `test_gen.rs` declares to test macro hygiene (P12).
    Non-Rust files are unaffected.
    """
    if not (file.endswith(".rs") and candidate_path.endswith(".rs")):
        return True
    file_crates = crates.get(file, frozenset({_rust_crate_key(file)}))
    candidate_crates = crates.get(
        candidate_path, frozenset({_rust_crate_key(candidate_path)})
    )
    return not file_crates.isdisjoint(candidate_crates)


@dataclass
class NameIdx:
    """Index for resolving bare names (functions, classes, interfaces)."""

    by_name: dict[str, list[NodeV1]]
    by_id: dict[str, NodeV1]
    by_path_name: dict[tuple[str, str], list[NodeV1]]


@dataclass
class MethodIdx:
    """Index for resolving methods on owners."""

    by_path_owner_method: dict[tuple[str, str, str], list[NodeV1]]


def build_indices(nodes: Iterable[NodeV1]) -> tuple[NameIdx, MethodIdx]:
    """Build name and method indexes used during edge resolution."""
    name_idx = NameIdx(by_name={}, by_id={}, by_path_name={})
    method_idx = MethodIdx(by_path_owner_method={})

    for n in nodes:
        name_idx.by_id[n.id] = n

        name_idx.by_name.setdefault(n.name, []).append(n)
        name_idx.by_path_name.setdefault((n.path, n.name), []).append(n)

        if n.owner and n.kind == "method":
            key = (n.path, n.owner, n.name)
            method_idx.by_path_owner_method.setdefault(key, []).append(n)

    return name_idx, method_idx


def resolve_edges(
    nodes: list[NodeV1], raw_edges: list[RawEdge]
) -> list[EdgeV1]:
    """Resolve raw references and deduplicate the resulting edges."""
    name_idx, method_idx = build_indices(nodes)
    edges: list[EdgeV1] = []
    edge_keys: set[tuple[str, str, str]] = set()
    file_ids = {node.path: node.id for node in nodes if node.kind == "file"}
    # Built once per build: import resolution must not rescan every file for
    # every import, which made builds quadratic in repository size (A14).
    file_paths = set(file_ids)
    suffix_index = _proper_suffix_index(file_paths)
    rust_crates = _rust_crates(nodes)
    import_memo: dict[tuple[str, bool, str], list[str]] = {}

    def resolve_import(importer: str, specifier: str) -> list[str]:
        """Resolve one import, memoized across importers where sound.

        Outside Rust the result depends only on the importer's directory,
        whether it is Python, and the specifier, so `import os` in hundreds
        of files resolves once. Rust is not memoized: its module targets are
        still being collected while this loop runs.
        """
        if importer.endswith(".rs"):
            return _resolve_import_paths(
                importer,
                specifier,
                file_paths,
                rust_module_targets,
                suffix_index,
            )
        key = (
            importer.rpartition("/")[0],
            importer.endswith((".py", ".pyi")),
            specifier,
        )
        if key not in import_memo:
            import_memo[key] = _resolve_import_paths(
                importer,
                specifier,
                file_paths,
                rust_module_targets,
                suffix_index,
            )
        return import_memo[key]

    # Import bindings per (file, local name): the definition the import sits
    # in (None at file level) and the symbol or module path it binds.
    imported_symbols: dict[
        tuple[str, str], list[tuple[str | None, NodeV1]]
    ] = {}
    # Local names bound by an import known to name a module outside the index
    # (see `_binds_unindexed_module`): `from functools import lru_cache`, or
    # `from ._pb_generated import Message`. Such a name is not a same-named
    # type or function in another file, so it is never guessed there. A
    # same-file definition still wins — that is the conditional-import
    # pattern (`try: from ._speedups import X` beside a pure-Python `X`),
    # where both bindings are live.
    external_bindings: dict[tuple[str, str], list[tuple[str | None, str]]] = {}
    tree_names = _tree_names(file_paths)
    imported_modules: dict[tuple[str, str], list[tuple[str | None, str]]] = {}
    rust_module_targets: dict[tuple[str, str], str] = {}
    parent_of = {
        raw.target_id: raw.source
        for raw in raw_edges
        if raw.relation == "contains" and raw.target_id
    }
    # Methods by the type that contains them, for inherited self-calls.
    methods_of: dict[tuple[str, str], list[NodeV1]] = {}
    for node in nodes:
        if node.kind == "method" and node.id in parent_of:
            methods_of.setdefault((parent_of[node.id], node.name), []).append(
                node
            )
    # Definitions nested directly in a function or method are lexically local: a
    # same-named parameter or local elsewhere must not bind to them, and they
    # are never importable or reachable by a cross-file name match. The
    # reference implementation indexes them like top-level names; this is a
    # deliberate divergence.
    function_scoped = {
        node_id
        for node_id, parent_id in parent_of.items()
        if (parent := name_idx.by_id.get(parent_id)) is not None
        and parent.kind in {"function", "method"}
    }

    def visible_from(target: NodeV1, source_id: str) -> bool:
        """Whether a call inside `source_id` can see `target` by bare name."""
        if target.id not in function_scoped:
            return True
        return _encloses(parent_of[target.id], source_id, parent_of)

    def append_edge(edge: EdgeV1) -> None:
        key = (edge.source, edge.target, edge.relation)
        if key not in edge_keys:
            edge_keys.add(key)
            edges.append(edge)

    for raw in raw_edges:
        if raw.relation != "imports":
            continue
        if raw.rust_module_name and raw.rust_module_target:
            rust_module_targets[(raw.file, raw.rust_module_name)] = (
                raw.rust_module_target
            )
            target_id = file_ids.get(raw.rust_module_target)
            if target_id:
                append_edge(
                    EdgeV1(
                        source=raw.source,
                        target=target_id,
                        relation="imports",
                    )
                )
            continue
        if not raw.specifier:
            continue
        target_paths = resolve_import(raw.file, raw.specifier)
        if not target_paths and not raw.file.endswith(".rs"):
            # Unresolved (external package, stdlib, or a path outside the tree):
            # keep the edge to the raw specifier, as in the reference
            # implementation (P4). Not for Rust: the reference implementation's
            # generic Rust tier emits no `use` edges, and a `use` path need not
            # name a module (`use E::*` imports an enum's variants).
            append_edge(
                EdgeV1(
                    source=raw.source, target=raw.specifier, relation="imports"
                )
            )
        if (
            not target_paths
            and raw.name
            and _binds_unindexed_module(raw.file, raw.specifier, tree_names)
        ):
            external_bindings.setdefault((raw.file, raw.name), []).append(
                (raw.scope_id, raw.name)
            )
        for target_path in target_paths:
            target_id = file_ids[target_path]
            append_edge(
                EdgeV1(source=raw.source, target=target_id, relation="imports")
            )
            if raw.name and raw.imported_name == "*":
                imported_modules.setdefault((raw.file, raw.name), []).append(
                    (raw.scope_id, target_path)
                )
            elif raw.name and raw.imported_name:
                targets = [
                    node
                    for node in name_idx.by_path_name.get(
                        (target_path, raw.imported_name), []
                    )
                    if node.kind in IMPORT_BINDABLE_KINDS
                    and node.id not in function_scoped
                ]
                imported_symbols.setdefault((raw.file, raw.name), []).extend(
                    (raw.scope_id, target) for target in targets
                )

    # Heritage first: a parent class can live in a file processed later,
    # and self-calls below walk the resolved `extends` chain.
    class_parents: dict[str, list[str]] = {}
    for raw in raw_edges:
        if raw.relation not in ("extends", "implements") or not raw.name:
            continue
        # The reference implementation resolves a base by name: `extends` to a
        # class or interface, `implements` to an interface or trait. An import
        # binding is tried first (as for calls), then a unique visible same-file
        # type, then a unique reachable one. An unresolved or ambiguous base
        # (usually external) keeps an edge to its bare name.
        source_node = name_idx.by_id.get(raw.source)
        if source_node is None:
            continue
        base_kinds = (
            {"interface", "trait"}
            if raw.relation == "implements"
            else {"class", "interface"}
        )
        bound = [
            target
            for target in _visible_bindings(
                imported_symbols.get((source_node.path, raw.name), []),
                raw.source,
                parent_of,
            )
            if target.kind in base_kinds and target.id != raw.source
        ]
        # An external import names the base: a same-named type in another
        # file is not it (`from django.views import View`; `class
        # MyView(View)` must not extend a test stub `View`).
        externally_bound = not bound and bool(
            _visible_bindings(
                external_bindings.get((source_node.path, raw.name), []),
                raw.source,
                parent_of,
            )
        )
        same_file = [
            target
            for target in name_idx.by_path_name.get(
                (source_node.path, raw.name), []
            )
            if target.kind in base_kinds
            and target.id != raw.source
            and visible_from(target, raw.source)
        ]
        reachable = [
            target
            for target in name_idx.by_name.get(raw.name, [])
            if target.kind in base_kinds
            and target.id != raw.source
            and target.id not in function_scoped
            and _reachable(source_node.path, target.path)
            and _same_rust_crate(source_node.path, target.path, rust_crates)
        ]
        attempts: list[tuple[list[NodeV1], Confidence]] = [
            (bound, "extracted"),
            (same_file, "extracted"),
            ([] if externally_bound else reachable, "inferred"),
        ]
        resolved = False
        for candidates, confidence in attempts:
            if candidates:
                resolved = len(candidates) == 1
                if resolved:
                    append_edge(
                        EdgeV1(
                            source=raw.source,
                            target=candidates[0].id,
                            relation=raw.relation,
                            confidence=confidence,
                        )
                    )
                    if raw.relation == "extends":
                        class_parents.setdefault(raw.source, []).append(
                            candidates[0].id
                        )
                break
        if not resolved:
            # Unknown or ambiguous base (usually an external type such as
            # `Exception`): an edge to the bare name, as in the reference
            # implementation (P4).
            append_edge(
                EdgeV1(
                    source=raw.source,
                    target=raw.name,
                    relation=raw.relation,
                    confidence="inferred",
                )
            )

    for raw in raw_edges:
        if raw.relation == "contains" and raw.target_id:
            append_edge(
                EdgeV1(
                    source=raw.source, target=raw.target_id, relation="contains"
                )
            )
            continue

        if raw.relation == "imports":
            continue

        if raw.relation in ("extends", "implements"):
            continue  # resolved in the heritage pass above

        if raw.relation == "calls" and raw.name:
            if raw.recv_type:
                # A call on a named type (Rust `Cache::new()`) or on a
                # variable whose type is known (`formatter: HelpFormatter`,
                # `x = Foo()`): resolve the type to a class, then the method
                # on it or up its bases (P13).
                source_node = name_idx.by_id.get(raw.source)
                type_node, type_bound = (
                    _resolve_type(
                        raw.recv_type,
                        source_node,
                        raw.source,
                        name_idx,
                        imported_symbols,
                        parent_of,
                    )
                    if source_node is not None
                    else (None, False)
                )
                if type_node is not None:
                    direct = methods_of.get((type_node.id, raw.name), [])
                    if len(direct) > 1 and type_node.path.endswith(
                        (".py", ".pyi")
                    ):
                        # `@overload` stubs precede the implementation, which
                        # is the last definition of the name in the class.
                        direct = [max(direct, key=_start_line)]
                    method = (
                        direct[0]
                        if len(direct) == 1
                        else _inherited_method(
                            type_node.id, raw.name, class_parents, methods_of
                        )
                        if not direct
                        else None
                    )
                    if method is not None:
                        append_edge(
                            EdgeV1(
                                source=raw.source,
                                target=method.id,
                                relation="calls",
                                confidence=(
                                    "extracted" if type_bound else "inferred"
                                ),
                            )
                        )
                        continue
                if source_node is not None:
                    methods = [
                        target
                        for target in name_idx.by_name.get(raw.name, [])
                        if target.kind == "method"
                        and target.owner == raw.recv_type
                        and _reachable(source_node.path, target.path)
                    ]
                    local = [
                        target
                        for target in methods
                        if target.path == source_node.path
                    ]
                    chosen = local or methods
                    if len(chosen) == 1:
                        append_edge(
                            EdgeV1(
                                source=raw.source,
                                target=chosen[0].id,
                                relation="calls",
                                confidence=(
                                    "extracted" if local else "inferred"
                                ),
                            )
                        )
                continue
            if raw.via_member and raw.specifier:
                if raw.specifier in ("this", "self"):
                    source_node = name_idx.by_id.get(raw.source)
                    if source_node and source_node.owner:
                        key = (source_node.path, source_node.owner, raw.name)
                        targets = method_idx.by_path_owner_method.get(key, [])
                        if len(targets) == 1:
                            append_edge(
                                EdgeV1(
                                    source=raw.source,
                                    target=targets[0].id,
                                    relation="calls",
                                )
                            )
                        elif not targets:
                            inherited = _inherited_method(
                                parent_of.get(raw.source),
                                raw.name,
                                class_parents,
                                methods_of,
                            )
                            if inherited is not None:
                                append_edge(
                                    EdgeV1(
                                        source=raw.source,
                                        target=inherited.id,
                                        relation="calls",
                                        confidence=(
                                            "extracted"
                                            if inherited.path
                                            == source_node.path
                                            else "inferred"
                                        ),
                                    )
                                )
                else:
                    source_node = name_idx.by_id.get(raw.source)
                    module_paths = (
                        _visible_bindings(
                            imported_modules.get(
                                (source_node.path, raw.specifier), []
                            ),
                            raw.source,
                            parent_of,
                        )
                        if source_node
                        else []
                    )
                    targets = [
                        target
                        for module_path in module_paths
                        for target in name_idx.by_path_name.get(
                            (module_path, raw.name), []
                        )
                        if target.kind in NAME_BINDABLE_KINDS
                        and target.id not in function_scoped
                    ]
                    if len(targets) == 1:
                        append_edge(
                            EdgeV1(
                                source=raw.source,
                                target=targets[0].id,
                                relation="calls",
                            )
                        )
                continue

            source_node = name_idx.by_id.get(raw.source)
            if source_node is None:
                continue
            callable_kinds = NAME_BINDABLE_KINDS
            bound_targets = [
                target
                for target in _visible_bindings(
                    imported_symbols.get((source_node.path, raw.name), []),
                    raw.source,
                    parent_of,
                )
                if target.kind in callable_kinds
            ]
            if len(bound_targets) == 1:
                append_edge(
                    EdgeV1(
                        source=raw.source,
                        target=bound_targets[0].id,
                        relation="calls",
                    )
                )
                continue
            if bound_targets:
                continue
            same_file_targets = [
                target
                for target in name_idx.by_path_name.get(
                    (source_node.path, raw.name), []
                )
                if target.kind in callable_kinds
            ]
            local_targets = [
                target
                for target in same_file_targets
                if visible_from(target, raw.source)
            ]
            if len(local_targets) == 1:
                append_edge(
                    EdgeV1(
                        source=raw.source,
                        target=local_targets[0].id,
                        relation="calls",
                    )
                )
                continue
            # Any same-file definition of the name, even one out of scope,
            # means this call is not safely guessable across files.
            if same_file_targets:
                continue
            # An external import binds this name (`from functools import
            # lru_cache`): it is that function, not a same-named one in another
            # file. The reference implementation guesses here; this is a
            # divergence.
            if _visible_bindings(
                external_bindings.get((source_node.path, raw.name), []),
                raw.source,
                parent_of,
            ):
                continue

            # Function-local definitions still count toward ambiguity here (a
            # shared name is a hint the call means something else, such as a
            # local variable), but are never picked as the target.
            global_targets = [
                target
                for target in name_idx.by_name.get(raw.name, [])
                if target.kind in callable_kinds
                and _reachable(source_node.path, target.path)
                and _same_rust_crate(source_node.path, target.path, rust_crates)
            ]
            if (
                len(global_targets) == 1
                and global_targets[0].id not in function_scoped
            ):
                append_edge(
                    EdgeV1(
                        source=raw.source,
                        target=global_targets[0].id,
                        relation="calls",
                        confidence="inferred",
                    )
                )

    return edges


TYPE_KINDS = frozenset({"class", "struct", "enum", "trait", "interface"})


def _start_line(node: NodeV1) -> int:
    return int(node.span.split("-", 1)[0].lstrip("L"))


def _resolve_type(
    type_name: str,
    source_node: NodeV1,
    source_id: str,
    name_idx: NameIdx,
    imported_symbols: dict[tuple[str, str], list[tuple[str | None, NodeV1]]],
    parent_of: dict[str, str],
) -> tuple[NodeV1 | None, bool]:
    """The type a receiver names, and whether it was bound, not guessed.

    Same file first, then the caller's import of that name; otherwise a type
    of that name that is unique among reachable files, marked as a guess.
    """
    candidates = [
        node
        for node in name_idx.by_name.get(type_name, [])
        if node.kind in TYPE_KINDS
    ]
    local = [node for node in candidates if node.path == source_node.path]
    if len(local) == 1:
        return local[0], True
    if local:
        return None, False
    imported = [
        node
        for node in _visible_bindings(
            imported_symbols.get((source_node.path, type_name), []),
            source_id,
            parent_of,
        )
        if node.kind in TYPE_KINDS
    ]
    if len(imported) == 1:
        return imported[0], True
    if imported:
        return None, False
    reachable = [
        node for node in candidates if _reachable(source_node.path, node.path)
    ]
    return (reachable[0], False) if len(reachable) == 1 else (None, False)


def _inherited_method(
    class_id: str | None,
    name: str,
    class_parents: dict[str, list[str]],
    methods_of: dict[tuple[str, str], list[NodeV1]],
) -> NodeV1 | None:
    """Find `name` up a class's resolved `extends` chain (`self.m()`).

    The reference implementation walks the chain by class name, up to three
    levels; this walks the resolved parent IDs, so a same-named class
    elsewhere cannot answer.
    The first level with a match decides: exactly one method, or nothing.
    """
    if class_id is None:
        return None
    seen = {class_id}
    frontier = list(class_parents.get(class_id, []))
    for _ in range(3):
        found = [
            method
            for parent in frontier
            for method in methods_of.get((parent, name), [])
        ]
        if found:
            return found[0] if len(found) == 1 else None
        seen.update(frontier)
        frontier = [
            grandparent
            for parent in frontier
            for grandparent in class_parents.get(parent, [])
            if grandparent not in seen
        ]
    return None


def _tree_names(file_paths: Iterable[str]) -> frozenset[str]:
    """Every directory name and file stem in the indexed tree."""
    names: set[str] = set()
    for path in file_paths:
        *directories, filename = path.split("/")
        names.update(directories)
        names.add(filename.split(".", 1)[0])
    return frozenset(names)


# JS/TS specifiers that are path aliases into the project (tsconfig
# `paths`, bundler aliases, `package.json` `imports`), not npm packages.
JS_ALIAS_PREFIXES = ("@/", "~/", "~", "#")


def _binds_unindexed_module(
    importer: str, specifier: str, tree_names: frozenset[str]
) -> bool:
    """Whether an unresolved import is known to name a module outside the tree.

    Unresolvable is not the same as external. It is known when:
    - the import is relative: it names one specific module that is not
      indexed (a compiled `._http_parser`, a generated `._pb2`);
    - the top-level package matches nothing in the tree: Python
      `functools` (no `functools.py` or `functools/`), JS `react` or
      `node:fs`. A JS path alias (`@/utils`) or a workspace package whose
      name matches a directory (`@myorg/shared` beside `packages/shared/`)
      is internal, only unresolvable without tsconfig or package.json.
    Rust never counts: its unresolved paths are usually internal modules
    this resolver cannot follow yet (`use self::content::x` into an inline
    module).
    """
    if importer.endswith(".rs"):
        return False
    if specifier.startswith((".", "/")):
        return True
    if importer.endswith((".py", ".pyi")):
        return specifier.split(".", 1)[0] not in tree_names
    if specifier.startswith("node:"):
        return True
    if specifier.startswith(JS_ALIAS_PREFIXES):
        return False
    parts = specifier.split("/")
    package = (
        parts[1] if specifier.startswith("@") and len(parts) > 1 else parts[0]
    )
    return package not in tree_names


def _encloses(scope_id: str, source_id: str, parent_of: dict[str, str]) -> bool:
    """Whether `source_id` is `scope_id` or nested somewhere inside it."""
    current: str | None = source_id
    while current is not None:
        if current == scope_id:
            return True
        current = parent_of.get(current)
    return False


def _visible_bindings(
    entries: list[tuple[str | None, _Bound]],
    source_id: str,
    parent_of: dict[str, str],
) -> list[_Bound]:
    """Distinct import bindings in scope at `source_id`.

    An import inside a function binds only there, and shadows a file-level
    import of the same name. Importing one symbol twice (say, under
    `TYPE_CHECKING` and again locally) is one binding, not an ambiguity.
    """
    local = [
        bound
        for scope_id, bound in entries
        if scope_id is not None and _encloses(scope_id, source_id, parent_of)
    ]
    visible = local or [
        bound for scope_id, bound in entries if scope_id is None
    ]
    distinct = {
        bound.id if isinstance(bound, NodeV1) else bound: bound
        for bound in visible
    }
    return list(distinct.values())


def _resolve_import_paths(
    importer: str,
    specifier: str,
    file_paths: set[str],
    rust_module_targets: dict[tuple[str, str], str],
    suffix_index: dict[str, list[str]],
) -> list[str]:
    importer_path = PurePosixPath(importer)
    python_absolute = False
    if importer.endswith(".rs") and (
        specifier in {"crate", "self", "super"}
        or specifier.startswith(("crate::", "self::", "super::"))
    ):
        target_path = _resolve_rust_module_path(
            importer_path,
            specifier,
            file_paths,
            rust_module_targets,
        )
        return [target_path] if target_path else []
    if specifier.startswith("."):
        base = importer_path.parent
        if importer.endswith((".py", ".pyi")):
            dot_count = len(specifier) - len(specifier.lstrip("."))
            for _ in range(max(0, dot_count - 1)):
                base = base.parent
            module_name = specifier[dot_count:]
            module_path = base / module_name.replace(".", "/")
        else:
            # `require('../..')`: collapse `..` segments, or the joined
            # path (`examples/auth/../..`) can never match a file (P4
            # exposed this in Express). Escaping the tree is unresolvable.
            normalized = posixpath.normpath(str(base / specifier))
            if normalized == ".." or normalized.startswith("../"):
                return []
            module_path = PurePosixPath(normalized)
    elif importer.endswith((".py", ".pyi")):
        python_absolute = True
        module_path = PurePosixPath(specifier.replace(".", "/"))
    else:
        return []

    stem = str(module_path)
    # A path normalized to the repository root (`.`) has its index files
    # at the top level: `index.js`, not `./index.js`.
    directory = "" if stem == "." else f"{stem}/"
    candidates = [stem]
    if not PurePosixPath(stem).suffix:
        candidates.extend(
            f"{stem}{extension}"
            for extension in (
                ".ts",
                ".tsx",
                ".mts",
                ".cts",
                ".js",
                ".jsx",
                ".mjs",
                ".cjs",
                ".py",
                ".pyi",
            )
        )
        candidates.extend(
            f"{directory}index{extension}"
            for extension in (
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
                ".mjs",
                ".cjs",
            )
        )
        candidates.append(f"{directory}__init__.py")
    elif PurePosixPath(stem).suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        source_stem = str(PurePosixPath(stem).with_suffix(""))
        candidates.extend(
            f"{source_stem}{extension}"
            for extension in (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx")
        )
    for candidate in dict.fromkeys(candidates):
        if candidate in file_paths:
            return [candidate]

    if python_absolute:
        # Equivalent to scanning for `path.endswith("/" + candidate)`, but a
        # dictionary lookup per candidate instead of a pass over every file.
        matches = {
            path
            for candidate in candidates
            for path in suffix_index.get(candidate, ())
        }
        if len(matches) == 1:
            return list(matches)
    return []


def _proper_suffix_index(file_paths: Iterable[str]) -> dict[str, list[str]]:
    """Map each proper path suffix to the files ending in it.

    `pkg/sub/mod.py` is indexed under `sub/mod.py` and `mod.py` (not under
    itself): exactly the keys `k` for which `path.endswith("/" + k)`.
    """
    index: dict[str, list[str]] = {}
    for path in file_paths:
        parts = path.split("/")
        for start in range(1, len(parts)):
            index.setdefault("/".join(parts[start:]), []).append(path)
    return index


def _resolve_rust_module_path(
    importer_path: PurePosixPath,
    specifier: str,
    file_paths: set[str],
    module_targets: dict[tuple[str, str], str],
) -> str | None:
    crate_root = _rust_crate_root(importer_path, file_paths)
    if crate_root is None:
        return None

    prefix, _, suffix = specifier.partition("::")
    suffix_parts: list[str] = suffix.split("::") if suffix else []
    if prefix == "crate":
        module_dir = crate_root
        module_file = _rust_module_source(module_dir, crate_root, file_paths)
    elif prefix == "self":
        module_dir = _rust_module_path(importer_path, crate_root)
        module_file = str(importer_path)
    else:
        module_dir = _rust_module_path(importer_path, crate_root)
        if module_dir == crate_root:
            return None
        module_dir = module_dir.parent
        module_file = _rust_module_source(module_dir, crate_root, file_paths)
        while suffix_parts and suffix_parts[0] == "super":
            if module_dir == crate_root:
                return None
            module_dir = module_dir.parent
            module_file = _rust_module_source(
                module_dir, crate_root, file_paths
            )
            suffix_parts.pop(0)

    for module_name in suffix_parts:
        target_path = (
            module_targets.get((module_file, module_name))
            if module_file is not None
            else None
        )
        if target_path not in file_paths:
            target_path = _rust_module_source(
                module_dir / module_name, crate_root, file_paths
            )
        if target_path is None:
            return None
        module_file = target_path
        module_dir = _rust_module_path(PurePosixPath(target_path), crate_root)
    return module_file


def _rust_module_source(
    module_dir: PurePosixPath,
    crate_root: PurePosixPath,
    file_paths: set[str],
) -> str | None:
    if module_dir == crate_root:
        candidates = (module_dir / "lib.rs", module_dir / "main.rs")
    else:
        candidates = (module_dir.with_suffix(".rs"), module_dir / "mod.rs")
    return next(
        (
            str(candidate)
            for candidate in candidates
            if str(candidate) in file_paths
        ),
        None,
    )


def _rust_crate_root(
    importer_path: PurePosixPath, file_paths: set[str]
) -> PurePosixPath | None:
    for parent in importer_path.parents:
        has_crate_root = any(
            str(parent / root) in file_paths for root in ("lib.rs", "main.rs")
        )
        if has_crate_root:
            return parent
    return None


def _rust_module_path(
    importer_path: PurePosixPath, crate_root: PurePosixPath
) -> PurePosixPath:
    if importer_path.name in {"lib.rs", "main.rs"}:
        return crate_root
    if importer_path.name == "mod.rs":
        return importer_path.parent
    return importer_path.parent / importer_path.stem
