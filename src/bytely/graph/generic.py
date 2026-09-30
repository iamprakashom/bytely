"""Generic declaration extraction for supported languages."""

from typing import Literal

import tree_sitter

GenericKind = Literal["constant", "variable", "function"]

# A JS/TS declarator whose value is one of these is a function definition:
# `const f = () => {}` / `const f = function () {}`, as in the reference
# implementation.
FUNCTION_VALUE_TYPES = frozenset(
    {
        "arrow_function",
        "function",
        "function_expression",
        "generator_function",
    }
)


def generic_declarations(
    root: tree_sitter.Node,
    source: bytes,
    language: str,
) -> list[tuple[tree_sitter.Node, GenericKind]]:
    """Return top-level constant, variable, and function-valued declarations."""
    if language == "python":
        return _python_declarations(root, source)
    if language in {"typescript", "tsx"}:
        return _typescript_declarations(root)
    if language == "go":
        return _go_declarations(root)
    if language == "rust":
        return _rust_declarations(root)
    if language == "kotlin":
        return _kotlin_declarations(root)
    if language in {"c", "cpp"}:
        return _c_declarations(root, source)
    if language == "swift":
        return _swift_declarations(root, source)
    if language == "c_sharp":
        return _c_sharp_declarations(root, source)
    return []


def _python_declarations(
    root: tree_sitter.Node, source: bytes
) -> list[tuple[tree_sitter.Node, GenericKind]]:
    declarations: list[tuple[tree_sitter.Node, GenericKind]] = []
    for statement in root.named_children:
        assignments = (
            statement.named_children
            if statement.type == "expression_statement"
            else [statement]
        )
        for assignment in assignments:
            if assignment.type not in {"assignment", "type"}:
                continue
            name_node = assignment.child_by_field_name("left")
            if name_node is None or name_node.type != "identifier":
                continue
            name = source[name_node.start_byte : name_node.end_byte].decode(
                "utf-8"
            )
            kind: GenericKind = "constant" if name.isupper() else "variable"
            declarations.append((assignment, kind))
    return declarations


def _typescript_declarations(
    root: tree_sitter.Node,
) -> list[tuple[tree_sitter.Node, GenericKind]]:
    declarations: list[tuple[tree_sitter.Node, GenericKind]] = []
    for statement in root.named_children:
        candidates = (
            statement.named_children
            if statement.type == "export_statement"
            else [statement]
        )
        for candidate in candidates:
            if candidate.type not in {
                "lexical_declaration",
                "variable_declaration",
            }:
                continue
            declaration_kind: GenericKind = (
                "constant"
                if candidate.children and candidate.children[0].type == "const"
                else "variable"
            )
            for declarator in candidate.named_children:
                if declarator.type != "variable_declarator":
                    continue
                name = declarator.child_by_field_name("name")
                if name is None or name.type != "identifier":
                    continue
                value = declarator.child_by_field_name("value")
                if value is not None and value.type in FUNCTION_VALUE_TYPES:
                    declarations.append((declarator, "function"))
                else:
                    declarations.append((declarator, declaration_kind))
    return declarations


def _go_declarations(
    root: tree_sitter.Node,
) -> list[tuple[tree_sitter.Node, GenericKind]]:
    declarations: list[tuple[tree_sitter.Node, GenericKind]] = []
    declaration_kinds: dict[str, GenericKind] = {
        "const_declaration": "constant",
        "var_declaration": "variable",
    }
    spec_kinds: dict[str, GenericKind] = {
        "const_spec": "constant",
        "var_spec": "variable",
    }
    for declaration in root.named_children:
        kind = declaration_kinds.get(declaration.type)
        if kind is None:
            continue
        declarations.extend(
            (spec, kind)
            for spec in declaration.named_children
            if spec_kinds.get(spec.type) == kind
        )
    return declarations


def _rust_declarations(
    root: tree_sitter.Node,
) -> list[tuple[tree_sitter.Node, GenericKind]]:
    declaration_kinds: dict[str, GenericKind] = {
        "const_item": "constant",
        "static_item": "variable",
    }
    return [
        (declaration, kind)
        for declaration in root.named_children
        if (kind := declaration_kinds.get(declaration.type)) is not None
        and declaration.child_by_field_name("name") is not None
    ]


def _kotlin_declarations(
    root: tree_sitter.Node,
) -> list[tuple[tree_sitter.Node, GenericKind]]:
    declarations: list[tuple[tree_sitter.Node, GenericKind]] = []
    for declaration in root.named_children:
        if declaration.type != "property_declaration":
            continue
        keyword_types = {child.type for child in declaration.children}
        if "val" in keyword_types:
            declarations.append((declaration, "constant"))
        elif "var" in keyword_types:
            declarations.append((declaration, "variable"))
    return declarations


def _c_declarations(
    root: tree_sitter.Node, source: bytes
) -> list[tuple[tree_sitter.Node, GenericKind]]:
    declarations: list[tuple[tree_sitter.Node, GenericKind]] = []
    for statement in root.named_children:
        if statement.type != "declaration":
            continue
        for declarator in statement.named_children:
            if declarator.type != "init_declarator":
                continue
            name = declarator.child_by_field_name("declarator")
            if name is None:
                continue
            prefix = source[statement.start_byte : declarator.start_byte]
            kind: GenericKind = (
                "constant"
                if {"const", "constexpr"}.intersection(
                    prefix.decode("utf-8").split()
                )
                else "variable"
            )
            declarations.append((declarator, kind))
    return declarations


def _swift_declarations(
    root: tree_sitter.Node, source: bytes
) -> list[tuple[tree_sitter.Node, GenericKind]]:
    declarations: list[tuple[tree_sitter.Node, GenericKind]] = []
    for declaration in root.named_children:
        if declaration.type != "property_declaration":
            continue
        tokens = (
            source[declaration.start_byte : declaration.end_byte]
            .decode("utf-8")
            .split()
        )
        binding = next(
            (token for token in tokens if token in {"let", "var"}), None
        )
        if binding:
            kind: GenericKind = "constant" if binding == "let" else "variable"
            declarations.append((declaration, kind))
    return declarations


def _c_sharp_declarations(
    root: tree_sitter.Node, source: bytes
) -> list[tuple[tree_sitter.Node, GenericKind]]:
    declarations: list[tuple[tree_sitter.Node, GenericKind]] = []
    for statement in root.named_children:
        if statement.type != "global_statement":
            continue
        local_declaration = next(
            (
                child
                for child in statement.named_children
                if child.type == "local_declaration_statement"
            ),
            None,
        )
        if local_declaration is None:
            continue
        variable_declaration = next(
            (
                child
                for child in local_declaration.named_children
                if child.type == "variable_declaration"
            ),
            None,
        )
        if variable_declaration is None:
            continue
        modifiers = (
            source[local_declaration.start_byte : local_declaration.end_byte]
            .decode("utf-8")
            .split()
        )
        kind: GenericKind = "constant" if "const" in modifiers else "variable"
        declarations.extend(
            (declarator, kind)
            for declarator in variable_declaration.named_children
            if declarator.type == "variable_declarator"
            and declarator.child_by_field_name("name") is not None
        )
    return declarations
