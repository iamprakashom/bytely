"""Tier-1 extraction: source file -> NodeV1[] + raw edges, via tree-sitter.

Deterministic and dependency-only (no LLM, no network).
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal

import tree_sitter

from bytely.graph.container import is_container_config_file
from bytely.graph.generic import FUNCTION_VALUE_TYPES, generic_declarations
from bytely.graph.types import Kind, NodeV1, Relation
from bytely.util.id import content_hash

Language = Literal[
    "typescript",
    "tsx",
    "python",
    "go",
    "java",
    "kotlin",
    "swift",
    "php",
    "r",
    "rust",
    "c",
    "cpp",
    "c_sharp",
]

SUPPORTED_GRAMMARS = frozenset(
    {
        "typescript",
        "tsx",
        "python",
        "go",
        "java",
        "kotlin",
        "swift",
        "php",
        "rust",
        "c",
        "cpp",
        "c_sharp",
    }
)

EXTENSIONS: list[dict[str, str]] = [
    {"ext": ".tsx", "grammar": "tsx", "label": "tsx"},
    {"ext": ".jsx", "grammar": "tsx", "label": "jsx"},
    {"ext": ".mts", "grammar": "typescript", "label": "typescript"},
    {"ext": ".cts", "grammar": "typescript", "label": "typescript"},
    {"ext": ".ts", "grammar": "typescript", "label": "typescript"},
    {"ext": ".mjs", "grammar": "typescript", "label": "javascript"},
    {"ext": ".cjs", "grammar": "typescript", "label": "javascript"},
    {"ext": ".js", "grammar": "typescript", "label": "javascript"},
    {"ext": ".pyi", "grammar": "python", "label": "python"},
    {"ext": ".py", "grammar": "python", "label": "python"},
    {"ext": ".go", "grammar": "go", "label": "go"},
    {"ext": ".java", "grammar": "java", "label": "java"},
    {"ext": ".kt", "grammar": "kotlin", "label": "kotlin"},
    {"ext": ".kts", "grammar": "kotlin", "label": "kotlin"},
    {"ext": ".swift", "grammar": "swift", "label": "swift"},
    {"ext": ".php", "grammar": "php", "label": "php"},
    {"ext": ".r", "grammar": "r", "label": "r"},
    {"ext": ".rs", "grammar": "rust", "label": "rust"},
    {"ext": ".c", "grammar": "c", "label": "c"},
    {"ext": ".h", "grammar": "c", "label": "c"},
    {"ext": ".cc", "grammar": "cpp", "label": "cpp"},
    {"ext": ".cpp", "grammar": "cpp", "label": "cpp"},
    {"ext": ".cxx", "grammar": "cpp", "label": "cpp"},
    {"ext": ".hh", "grammar": "cpp", "label": "cpp"},
    {"ext": ".hpp", "grammar": "cpp", "label": "cpp"},
    {"ext": ".hxx", "grammar": "cpp", "label": "cpp"},
    {"ext": ".cs", "grammar": "c_sharp", "label": "c_sharp"},
]


def _entry_for(path: str) -> dict[str, str] | None:
    p = path.lower()
    for e in EXTENSIONS:
        if p.endswith(e["ext"]):
            return e
    return None


def depth_extensions() -> list[str]:
    """File extensions whose grammar is supported and installed.

    Grammars outside the core languages are optional extras
    (`pip install bytely[go]`), so a missing one skips that language's
    files instead of failing the build.
    """
    return [
        entry["ext"]
        for entry in EXTENSIONS
        if entry["grammar"] in SUPPORTED_GRAMMARS
        and grammar_installed(entry["grammar"])
    ]


@functools.cache
def grammar_installed(grammar: str) -> bool:
    """Whether the grammar package for `grammar` can be imported."""
    loader = GRAMMAR_LOADERS.get(grammar)  # type: ignore[call-overload]
    if loader is None:
        return False
    return importlib.util.find_spec(loader[0]) is not None


def language_of(path: str) -> Language | None:
    """Map a file path to a supported language, or None if unsupported."""
    entry = _entry_for(path)
    return entry["grammar"] if entry else None  # type: ignore


def language_label_of(path: str) -> str | None:
    """What to call the language of this file."""
    entry = _entry_for(path)
    return entry["label"] if entry else None


@dataclass(slots=True)
class RawEdge:
    """Unresolved relationship emitted while parsing a source file."""

    source: str
    relation: Relation
    file: str
    target_id: str | None = None
    specifier: str | None = None
    name: str | None = None
    imported_name: str | None = None
    via_member: bool | None = None
    recv_type: str | None = None
    kinds: list[Kind] | None = None
    arg_count: int | None = None
    implicit_self: bool | None = None
    is_wildcard: bool = False
    rust_module_name: str | None = None
    rust_module_target: str | None = None
    # For an import binding: the definition the import statement sits in
    # (None at file level). A function-local import binds only inside it.
    scope_id: str | None = None


@dataclass(slots=True)
class ExtractResult:
    """Nodes and unresolved relationships extracted from one file."""

    nodes: list[NodeV1]
    raw_edges: list[RawEdge]


GRAMMAR_LOADERS: dict[Language, tuple[str, str]] = {
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "python": ("tree_sitter_python", "language"),
    "go": ("tree_sitter_go", "language"),
    "java": ("tree_sitter_java", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "kotlin": ("tree_sitter_kotlin", "language"),
    "swift": ("tree_sitter_swift", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "c_sharp": ("tree_sitter_c_sharp", "language"),
}

PARSERS: dict[Language, tree_sitter.Parser] = {}


def get_parser(lang: Language) -> tree_sitter.Parser:
    """Return the cached tree-sitter parser for a supported language."""
    if lang not in PARSERS:
        loader = GRAMMAR_LOADERS.get(lang)
        if loader is None:
            raise NotImplementedError(
                f"No packaged tree-sitter grammar is configured for {lang}"
            )
        module_name, function_name = loader
        try:
            module = importlib.import_module(module_name)
            grammar = getattr(module, function_name)()
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                f"The {lang} parser is unavailable; install the project "
                "dependencies to enable it."
            ) from error

        parser = tree_sitter.Parser()
        parser.language = tree_sitter.Language(grammar)
        PARSERS[lang] = parser
    return PARSERS[lang]


_JAVASCRIPT_PARSER: tree_sitter.Parser | None = None

# Statement keywords that error recovery can misread as a method name
# (`if (x) {…}` inside a broken class body becomes a method `if`).
_JS_RECOVERY_NAMES = frozenset(
    {
        "if", "else", "for", "while", "do", "switch", "case", "return",
        "throw", "try", "catch", "finally", "break", "continue", "const",
        "let", "var", "function", "class", "import", "export", "default",
        "with", "debugger", "yield", "await",
    }
)  # fmt: skip


def _member_assigned_function(
    node: tree_sitter.Node, source_bytes: bytes
) -> tuple[tuple[str, ...], str] | None:
    """Owner path and name of a function assigned to a property, or None.

    `app.use = function () {}` → (("app",), "use"), a method of `app`;
    `Foo.prototype.bar = () => {}` → (("Foo",), "bar");
    `module.exports.x = …` and `exports.x = …` → ((), "x"), an exported
    function. Computed (`a[b] = …`) and `this.x = …` targets are skipped:
    their owner is not known statically. The reference implementation mints
    no node for any of these (decision #11).
    """
    right = node.child_by_field_name("right")
    left = node.child_by_field_name("left")
    if (
        right is None
        or right.type not in FUNCTION_VALUE_TYPES
        or left is None
        or left.type != "member_expression"
    ):
        return None
    parts: list[str] = []
    current: tree_sitter.Node | None = left
    while current is not None and current.type == "member_expression":
        prop = current.child_by_field_name("property")
        if prop is None or prop.type != "property_identifier":
            return None
        parts.append(source_bytes[prop.start_byte : prop.end_byte].decode())
        current = current.child_by_field_name("object")
    if current is None or current.type != "identifier":
        return None
    parts.append(source_bytes[current.start_byte : current.end_byte].decode())
    parts.reverse()
    if parts[-1] == "prototype":
        # `Foo.prototype = function () {}` replaces the prototype object;
        # it defines no method.
        return None
    if parts[:2] == ["module", "exports"]:
        exported = parts[2:]
        return ((), exported[0]) if len(exported) == 1 else None
    if parts[0] == "exports":
        return ((), parts[1]) if len(parts) == 2 else None
    owner = tuple(part for part in parts[:-1] if part != "prototype")
    return (owner, parts[-1]) if owner else None


def _is_recovery_artifact(
    lang: Language,
    node: tree_sitter.Node,
    name_node: tree_sitter.Node,
    source_bytes: bytes,
) -> bool:
    """Whether a definition is a parse-error artifact, not real code.

    JavaScript allows keywords as method names (`delete() {}`), so a keyword
    name alone is not enough: it must also sit inside an `ERROR` subtree.
    Definitions there with ordinary names are kept; they are usually real
    (Flow `export type` lines the grammar could not parse).
    """
    if lang not in ("typescript", "tsx"):
        return False
    name = source_bytes[name_node.start_byte : name_node.end_byte]
    if name.decode("utf-8") not in _JS_RECOVERY_NAMES:
        return False
    parent = node.parent
    while parent is not None:
        if parent.type == "ERROR":
            return True
        parent = parent.parent
    return False


def _error_count(tree: tree_sitter.Tree) -> int:
    count = 0
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        if node.type == "ERROR" or node.is_missing:
            count += 1
        pending.extend(node.children)
    return count


def _parse_javascript(source_bytes: bytes) -> tree_sitter.Tree:
    """Parse JavaScript, falling back to TSX for syntax it cannot read.

    `tree-sitter-javascript` reads JSX in `.js` files and valid JavaScript
    the TypeScript grammars reject (`interface` as an identifier, and
    `f(a < b, c > (d))`, which they misread as a generic call). It fails on
    Flow annotations, which TSX tolerates better, so a parse with errors is
    retried with TSX and the one with fewer errors kept (decision #15). Both
    grammars share the node types the TypeScript extraction tables use.
    """
    global _JAVASCRIPT_PARSER
    if _JAVASCRIPT_PARSER is None:
        import tree_sitter_javascript

        _JAVASCRIPT_PARSER = tree_sitter.Parser(
            tree_sitter.Language(tree_sitter_javascript.language())
        )
    tree = _JAVASCRIPT_PARSER.parse(source_bytes)
    if not tree.root_node.has_error:
        return tree
    fallback = get_parser("tsx").parse(source_bytes)
    return fallback if _error_count(fallback) < _error_count(tree) else tree


MAX_BODY_CHARS = 5000
MAX_FILE_BODY_CHARS = 16000


def search_body(text: str, max_len: int = MAX_BODY_CHARS) -> str:
    """Normalize and bound text for graph search."""
    norm = re.sub(r"\s+", " ", text).strip()
    return norm[:max_len] if len(norm) > max_len else norm


def file_residual(source: str, symbols: list[NodeV1]) -> str:
    """Return searchable file text outside the extracted symbol spans."""
    lines = source.split("\n")
    covered = bytearray(len(lines) + 2)
    for s in symbols:
        m = re.match(r"^L(\d+)-L(\d+)$", s.span)
        if not m:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        for r in range(start, min(end + 1, len(covered))):
            covered[r] = 1

    kept = []
    for i in range(len(lines)):
        if not covered[i + 1]:
            kept.append(lines[i])
    return search_body(" ".join(kept), MAX_FILE_BODY_CHARS)


TS_KINDS: dict[str, Kind] = {
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "method_definition": "method",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
}

PY_KINDS: dict[str, Kind] = {
    "class_definition": "class",
    "function_definition": "function",
}

GO_KINDS: dict[str, Kind] = {
    "function_declaration": "function",
    "method_declaration": "method",
}

R_KINDS: dict[str, Kind] = {}

JAVA_KINDS: dict[str, Kind] = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "struct",
    "annotation_type_declaration": "interface",
    "annotation_type_element_declaration": "method",
    "method_declaration": "method",
    "constructor_declaration": "method",
}

JAVA_TYPE_KINDS: set[Kind] = {"class", "interface", "enum", "struct"}

KOTLIN_KINDS: dict[str, Kind] = {
    "class_declaration": "class",
    "object_declaration": "class",
    "function_declaration": "function",
    "secondary_constructor": "method",
    "type_alias": "type",
    "property_declaration": "variable",
}

KOTLIN_TYPE_KINDS: set[Kind] = {"class", "interface", "enum"}

SWIFT_KINDS: dict[str, Kind] = {
    "class_declaration": "class",
    "protocol_declaration": "interface",
    "function_declaration": "function",
    "protocol_function_declaration": "method",
    "init_declaration": "method",
    "typealias_declaration": "type",
    "property_declaration": "variable",
}

SWIFT_TYPE_KINDS: set[Kind] = {"class", "struct", "enum", "interface", "module"}

PHP_KINDS: dict[str, Kind] = {
    "function_definition": "function",
    "method_declaration": "method",
    "class_declaration": "class",
    "interface_declaration": "interface",
    "trait_declaration": "trait",
    "enum_declaration": "enum",
}

RUST_KINDS: dict[str, Kind] = {
    "function_item": "function",
    "function_signature_item": "method",
    "struct_item": "struct",
    "enum_item": "enum",
    "trait_item": "trait",
    "type_item": "type",
    "mod_item": "module",
    "const_item": "constant",
    "static_item": "variable",
}

C_KINDS: dict[str, Kind] = {
    "function_definition": "function",
    "struct_specifier": "struct",
    "enum_specifier": "enum",
    "type_definition": "type",
}

CPP_KINDS: dict[str, Kind] = {
    "function_definition": "function",
    "class_specifier": "class",
    "struct_specifier": "struct",
    "enum_specifier": "enum",
    "namespace_definition": "module",
    "alias_declaration": "type",
}

C_SHARP_KINDS: dict[str, Kind] = {
    "class_declaration": "class",
    "struct_declaration": "struct",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "delegate_declaration": "type",
    "method_declaration": "method",
    "constructor_declaration": "method",
    "local_function_statement": "function",
}

KINDS_BY_LANG: dict[Language, dict[str, Kind]] = {
    "typescript": TS_KINDS,
    "tsx": TS_KINDS,
    "python": PY_KINDS,
    "go": GO_KINDS,
    "r": R_KINDS,
    "java": JAVA_KINDS,
    "kotlin": KOTLIN_KINDS,
    "swift": SWIFT_KINDS,
    "php": PHP_KINDS,
    "rust": RUST_KINDS,
    "c": C_KINDS,
    "cpp": CPP_KINDS,
    "c_sharp": C_SHARP_KINDS,
}

CALL_TYPES: dict[Language, set[str]] = {
    "typescript": {"call_expression"},
    "tsx": {"call_expression"},
    "python": {"call"},
    "go": {"call_expression"},
    "java": {"method_invocation", "object_creation_expression"},
    "kotlin": {"call_expression"},
    "swift": {"call_expression"},
    "php": {
        "function_call_expression",
        "member_call_expression",
        "nullsafe_member_call_expression",
        "scoped_call_expression",
    },
    "r": {"call"},
    "rust": {"call_expression"},
    "c": {"call_expression"},
    "cpp": {"call_expression"},
    "c_sharp": {"invocation_expression", "object_creation_expression"},
}


def _declaration_name(node: tree_sitter.Node) -> tree_sitter.Node | None:
    name = node.child_by_field_name("name")
    if name:
        return name

    declarator = node.child_by_field_name("declarator")
    pending = [declarator] if declarator else []
    while pending:
        declarator = pending.pop(0)
        if declarator.type in {
            "identifier",
            "field_identifier",
            "type_identifier",
        }:
            return declarator
        pending[0:0] = declarator.named_children
    return None


def _rust_impl_type_name(
    node: tree_sitter.Node, source_bytes: bytes
) -> str | None:
    type_node = node.child_by_field_name("type")
    if type_node is None:
        return None
    if type_node.type == "generic_type":
        type_node = type_node.child_by_field_name("type") or type_node
    if type_node.type == "scoped_type_identifier":
        type_node = type_node.child_by_field_name("name") or type_node
    if type_node.type not in {"identifier", "type_identifier"}:
        return None
    return source_bytes[type_node.start_byte : type_node.end_byte].decode(
        "utf-8"
    )


def _swift_property_name(node: tree_sitter.Node) -> tree_sitter.Node | None:
    pending = list(node.named_children)
    while pending:
        child = pending.pop(0)
        if child.type == "simple_identifier":
            return child
        pending[0:0] = child.named_children
    return None


def _import_specifier(
    node: tree_sitter.Node, source_bytes: bytes, lang: Language
) -> str | None:
    text = source_bytes[node.start_byte : node.end_byte].decode("utf-8")
    if lang in ("typescript", "tsx"):
        match = re.search(r"\bfrom\s*(['\"])([^'\"]+)\1", text)
        if not match:
            match = re.match(r"\s*import\s*(['\"])([^'\"]+)\1", text)
        return match.group(2) if match else None
    if lang == "python":
        match = re.match(r"\s*from\s+([.\w]+)\s+import\b", text)
        if not match:
            match = re.match(r"\s*import\s+([\w.]+)", text)
        return match.group(1) if match else None
    return None


def _imported_symbols(
    node: tree_sitter.Node, source_bytes: bytes, lang: Language
) -> list[tuple[str, str]]:
    text = source_bytes[node.start_byte : node.end_byte].decode("utf-8")
    if lang == "python":
        if not re.match(r"\s*from\s+", text):
            match = re.fullmatch(
                r"\s*import\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?:\s+as\s+([A-Za-z_]\w*))?\s*",
                text,
            )
            if not match:
                return []
            return [(match.group(2) or match.group(1), "*")]
        match = re.search(r"\bimport\s+(.+)$", text, re.DOTALL)
        if not match:
            return []
        bindings = match.group(1).strip().strip("()").replace("\n", " ")
        result = []
        for binding in bindings.split(","):
            parts = re.fullmatch(
                r"\s*([A-Za-z_]\w*)(?:\s+as\s+([A-Za-z_]\w*))?\s*", binding
            )
            if parts:
                imported = parts.group(1)
                result.append((parts.group(2) or imported, imported))
        return result
    if lang in ("typescript", "tsx"):
        match = re.search(
            r"\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s*(['\"])", text
        )
        if match:
            return [(match.group(1), "*")]
        match = re.search(r"\{([^{}]+)\}\s*from\s*(['\"])", text, re.DOTALL)
        if not match:
            return []
        result = []
        for binding in match.group(1).split(","):
            parts = re.fullmatch(
                r"\s*(?:type\s+)?([A-Za-z_$][\w$]*)(?:\s+as\s+([A-Za-z_$][\w$]*))?\s*",
                binding,
            )
            if parts:
                imported = parts.group(1)
                result.append((parts.group(2) or imported, imported))
        return result
    return []


def _commonjs_require(
    node: tree_sitter.Node, source_bytes: bytes
) -> tuple[str, list[tuple[str, str]]] | None:
    """Read `require('./x')` as a specifier plus its local bindings.

    `const x = require(...)` binds the module (`*`); a destructuring
    `const { a, b: c } = require(...)` binds exported names.
    """
    function = node.child_by_field_name("function")
    arguments = node.child_by_field_name("arguments")
    if (
        function is None
        or function.type != "identifier"
        or source_bytes[function.start_byte : function.end_byte] != b"require"
        or arguments is None
        or len(arguments.named_children) != 1
        or arguments.named_children[0].type != "string"
    ):
        return None
    literal = arguments.named_children[0]
    fragment = next(
        (c for c in literal.named_children if c.type == "string_fragment"),
        None,
    )
    if fragment is None:
        return None
    specifier = source_bytes[fragment.start_byte : fragment.end_byte].decode(
        "utf-8"
    )

    def text(child: tree_sitter.Node) -> str:
        return source_bytes[child.start_byte : child.end_byte].decode("utf-8")

    bindings: list[tuple[str, str]] = []
    parent = node.parent
    if parent is not None and parent.type == "variable_declarator":
        pattern = parent.child_by_field_name("name")
        if pattern is not None and pattern.type == "identifier":
            bindings.append((text(pattern), "*"))
        elif pattern is not None and pattern.type == "object_pattern":
            for prop in pattern.named_children:
                if prop.type == "shorthand_property_identifier_pattern":
                    bindings.append((text(prop), text(prop)))
                elif prop.type == "pair_pattern":
                    key = prop.child_by_field_name("key")
                    value = prop.child_by_field_name("value")
                    if (
                        key is not None
                        and value is not None
                        and key.type == "property_identifier"
                        and value.type == "identifier"
                    ):
                        bindings.append((text(value), text(key)))
    return specifier, bindings


def _rust_use_bindings(
    node: tree_sitter.Node, source_bytes: bytes
) -> list[tuple[str, str, str, bool]]:
    """Flatten a Rust use tree into module path, local name, and item name."""
    argument = node.child_by_field_name("argument")
    if argument is None:
        return []
    return _walk_rust_use_tree(argument, source_bytes)


def _rust_path_attribute(
    node: tree_sitter.Node, source_bytes: bytes
) -> str | None:
    parent = node.parent
    if parent is None:
        return None

    attributes: list[tree_sitter.Node] = []
    for sibling in parent.children:
        if sibling.start_byte >= node.start_byte:
            break
        if sibling.type == "attribute_item":
            attributes.append(sibling)
        elif sibling.type not in {"line_comment", "block_comment"}:
            attributes.clear()

    for attribute_item in reversed(attributes):
        if not attribute_item.named_children:
            continue
        attribute = attribute_item.named_children[0]
        if not attribute.named_children:
            continue
        name_node = attribute.named_children[0]
        name = source_bytes[name_node.start_byte : name_node.end_byte].decode(
            "utf-8"
        )
        if name != "path":
            continue
        value = attribute.child_by_field_name("value")
        if value is None or value.type != "string_literal":
            continue
        literal = source_bytes[value.start_byte : value.end_byte].decode(
            "utf-8"
        )
        try:
            path = ast.literal_eval(literal)
        except (SyntaxError, ValueError):
            return None
        return path if isinstance(path, str) else None
    return None


def _walk_rust_use_tree(
    node: tree_sitter.Node, source_bytes: bytes, prefix: str = ""
) -> list[tuple[str, str, str, bool]]:
    if node.type == "scoped_use_list":
        path = node.child_by_field_name("path")
        use_list = node.child_by_field_name("list")
        if path is None or use_list is None:
            return []
        path_text = source_bytes[path.start_byte : path.end_byte].decode(
            "utf-8"
        )
        current_prefix = f"{prefix}::{path_text}" if prefix else path_text
        return [
            binding
            for child in use_list.named_children
            for binding in _walk_rust_use_tree(
                child, source_bytes, current_prefix
            )
        ]

    if node.type == "use_list":
        return [
            binding
            for child in node.named_children
            for binding in _walk_rust_use_tree(child, source_bytes, prefix)
        ]

    if node.type == "use_wildcard":
        if not node.named_children:
            return []
        path = node.named_children[0]
        path_text = source_bytes[path.start_byte : path.end_byte].decode(
            "utf-8"
        )
        module_path = f"{prefix}::{path_text}" if prefix else path_text
        local = module_path.rsplit("::", 1)[-1]
        return [(module_path, local, "*", True)]

    if node.type == "self" and prefix:
        local = prefix.rsplit("::", 1)[-1]
        return [(prefix, local, "*", False)]

    alias = None
    path = node
    if node.type == "use_as_clause":
        path = node.child_by_field_name("path")
        alias = node.child_by_field_name("alias")
    if path is None or path.type not in {"identifier", "scoped_identifier"}:
        return []

    path_text = source_bytes[path.start_byte : path.end_byte].decode("utf-8")
    full_path = f"{prefix}::{path_text}" if prefix else path_text
    module_path, separator, imported = full_path.rpartition("::")
    if not separator:
        return []
    local = (
        source_bytes[alias.start_byte : alias.end_byte].decode("utf-8")
        if alias
        else imported
    )
    return [(module_path, local, imported, False)]


@dataclass(frozen=True, slots=True)
class _WalkCtx:
    """Lexical context for the definitions below a syntax node."""

    # Node that contains definitions found here (the file, or a definition).
    parent_id: str
    # Nearest enclosing type name: the owner of any method found here.
    owner_name: str | None = None
    # Enclosing definition names; they qualify nested IDs as the reference
    # implementation does (`cli.py#outer.inner`,
    # `models.py#Widget.Inner.method`).
    scope: tuple[str, ...] = ()
    # Kind of the nearest enclosing definition, or None at file level.
    enclosing_kind: Kind | None = None
    rust_impl: bool = False
    # Nearest enclosing function or method: an import found here binds only
    # inside it. None at file level (and in a top-level binding such as
    # `const m = require(...)`, which binds for the whole file).
    import_scope: str | None = None
    # Local variable → type name, where the function's own code says so
    # (`formatter: HelpFormatter`, `x = Foo()`, `let c = Cache::new()`). A
    # member call on such a variable is resolved against that type.
    receiver_types: dict[str, str] = field(default_factory=dict)


# Scopes whose locals are their own: a receiver scan stops at them.
_SCOPE_NODE_TYPES = frozenset(
    {
        "function_definition",
        "class_definition",
        "lambda",
        "function_item",
        "closure_expression",
        "impl_item",
        "function_declaration",
        "function_expression",
        "generator_function_declaration",
        "arrow_function",
        "class_declaration",
        "method_definition",
    }
)
# Wrappers whose methods callers usually reach through (`Box<T>` → `T`).
_RUST_WRAPPERS = frozenset({"Box", "Rc", "Arc"})
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def _annotation_type(text: str, lang: Language) -> str | None:
    """The class a type annotation names, or None if it is not one class.

    `HelpFormatter`, `"Context"`, `t.Optional[Context]`, `Context | None`,
    `&mut Parser<'a>`, `Box<Cache>`, and `crate::ser::Serializer` each name
    one class; containers (`list[Foo]`, `Vec<Foo>`) and unions do not.
    """
    text = text.strip().strip("\"'").strip()
    if lang == "python":
        parts = [part.strip() for part in text.split("|")]
        parts = [part for part in parts if part != "None"]
        if len(parts) != 1:
            return None
        text = parts[0]
        optional = re.fullmatch(r"(?:\w+\.)?Optional\[(.+)\]", text)
        if optional:
            text = optional.group(1).strip().strip("\"'")
        if "[" in text:
            return None
        name = text.rsplit(".", 1)[-1]
    elif lang == "rust":
        text = re.sub(r"^&\s*('\w+\s+)?(mut\s+)?", "", text)
        text = re.sub(r"^(dyn|impl)\s+", "", text)
        base, _, rest = text.partition("<")
        name = base.rsplit("::", 1)[-1].strip()
        if name in _RUST_WRAPPERS and rest:
            return _annotation_type(rest.rsplit(">", 1)[0], lang)
    else:
        return None
    return name if _IDENTIFIER.fullmatch(name) else None


def _constructed_type(
    value: tree_sitter.Node, lang: Language, source_bytes: bytes
) -> str | None:
    """The type an initializer constructs (`Foo()`, `Cache::new()`)."""

    def text(node: tree_sitter.Node) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8")

    if lang == "python" and value.type == "call":
        callee = value.child_by_field_name("function")
        if callee is not None and callee.type in ("identifier", "attribute"):
            name = text(callee).rsplit(".", 1)[-1]
            # Classes are capitalized by convention; `load()` is not a type.
            return name if name[:1].isupper() else None
    if lang == "rust":
        if value.type == "struct_expression":
            name_node = value.child_by_field_name("name")
            return (
                _annotation_type(text(name_node), lang) if name_node else None
            )
        if value.type == "call_expression":
            callee = value.child_by_field_name("function")
            if callee is not None and callee.type == "scoped_identifier":
                path_node = callee.child_by_field_name("path")
                if path_node is not None:
                    name = text(path_node).rsplit("::", 1)[-1]
                    return name if name[:1].isupper() else None
    if lang in ("typescript", "tsx") and value.type == "new_expression":
        constructor = value.child_by_field_name("constructor")
        if constructor is not None and constructor.type == "identifier":
            return text(constructor)
    return None


# Where a syntax node binds names, per language: node type -> the fields
# holding the bound pattern. `None` means the node itself is the pattern.
_BINDING_SITES: dict[str, dict[str, tuple[str | None, ...]]] = {
    "python": {
        "assignment": ("left",),
        "augmented_assignment": ("left",),
        "for_statement": ("left",),
        "for_in_clause": ("left",),
        "named_expression": ("name",),
        "as_pattern_target": (None,),
        "global_statement": (None,),
        "nonlocal_statement": (None,),
        "case_pattern": (None,),
        "aliased_import": ("alias",),
    },
    "rust": {
        "let_declaration": ("pattern",),
        "for_expression": ("pattern",),
        "let_condition": ("pattern",),
        "match_arm": ("pattern",),
        "assignment_expression": ("left",),
        "compound_assignment_expr": ("left",),
    },
    "typescript": {
        "variable_declarator": ("name",),
        "assignment_expression": ("left",),
        "augmented_assignment_expression": ("left",),
        "for_in_statement": ("left",),
        "catch_clause": ("parameter",),
    },
}
_BINDING_SITES["tsx"] = _BINDING_SITES["typescript"]
# The forms that can give a plain name a type, per language: node type ->
# (target field, annotation field, value field).
_TYPED_BINDINGS: dict[str, tuple[str, str, str, str]] = {
    "python": ("assignment", "left", "type", "right"),
    "rust": ("let_declaration", "pattern", "type", "value"),
    "typescript": ("variable_declarator", "name", "type", "value"),
    "tsx": ("variable_declarator", "name", "type", "value"),
}
# Parts of a pattern that name no local: annotations, default values, and
# member targets (`self.x = …`, `obj.y = …` assign attributes, not names).
_NON_BINDING_FIELDS = ("type", "value", "right", "default", "key")
_NON_BINDING_NODES = frozenset(
    {
        "attribute",
        "subscript",
        "member_expression",
        "subscript_expression",
        "field_expression",
        "index_expression",
        "scoped_identifier",
        "type_annotation",
    }
)
_NAME_NODES = frozenset({"identifier", "shorthand_property_identifier_pattern"})
_FUNCTION_NODE_TYPES = frozenset(
    {
        "function_definition",
        "function_item",
        "function_declaration",
        "generator_function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
    }
)


def _pattern_names(node: tree_sitter.Node, source_bytes: bytes) -> list[str]:
    """Every local name a binding pattern introduces."""
    names: list[str] = []
    pending = [node]
    while pending:
        current = pending.pop()
        if current.type in _NAME_NODES:
            names.append(
                source_bytes[current.start_byte : current.end_byte].decode()
            )
            continue
        if current.type in _NON_BINDING_NODES:
            continue
        skipped = {
            child.id
            for field_name in _NON_BINDING_FIELDS
            for child in current.children_by_field_name(field_name)
        }
        pending.extend(
            child for child in current.named_children if child.id not in skipped
        )
    return names


def _receiver_types(
    function: tree_sitter.Node,
    lang: Language,
    source_bytes: bytes,
    inherited: dict[str, str],
    owner_name: str | None,
) -> dict[str, str]:
    """Variable types a function establishes, on top of its closure's.

    Every binding of a name in the function counts: parameters, assignments,
    loop and `with`/`except` targets, destructuring, walrus, `global`, and
    match or `if let` patterns. A name keeps a type only if each of its
    bindings is a typed form that names the same type (an annotation or a
    constructor call); any other binding removes it, including a type from
    the enclosing function. Removing too eagerly only loses an edge; keeping
    a stale type would create a wrong one.
    """
    if function.type not in _FUNCTION_NODE_TYPES:
        return dict(inherited)
    found: dict[str, set[str | None]] = {}

    def text(node: tree_sitter.Node) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8")

    def bind(name: str, type_name: str | None) -> None:
        if lang == "rust" and type_name == "Self":
            type_name = owner_name
        found.setdefault(name, set()).add(type_name)

    def bind_pattern(pattern: tree_sitter.Node) -> None:
        for name in _pattern_names(pattern, source_bytes):
            bind(name, None)

    def bind_typed(
        target: tree_sitter.Node | None,
        annotation: tree_sitter.Node | None,
        value: tree_sitter.Node | None,
    ) -> bool:
        """Bind a plain name from its annotation or constructor call."""
        if target is None or target.type != "identifier":
            return False
        if annotation is not None:
            type_name = _annotation_type(text(annotation), lang)
        elif value is not None:
            type_name = _constructed_type(value, lang, source_bytes)
        else:
            type_name = None
        bind(text(target), type_name)
        return True

    parameters = function.child_by_field_name("parameters")
    single = function.child_by_field_name("parameter")
    if single is not None:
        bind_pattern(single)  # `c => …`: one parameter, no parentheses
    for parameter in parameters.named_children if parameters else ():
        if lang == "python" and parameter.type in (
            "typed_parameter",
            "typed_default_parameter",
        ):
            name_node = parameter.child_by_field_name("name") or next(
                (
                    child
                    for child in parameter.named_children
                    if child.type == "identifier"
                ),
                None,
            )
            if bind_typed(
                name_node, parameter.child_by_field_name("type"), None
            ):
                continue
        elif lang == "rust" and parameter.type == "parameter":
            if bind_typed(
                parameter.child_by_field_name("pattern"),
                parameter.child_by_field_name("type"),
                None,
            ):
                continue
        bind_pattern(parameter)

    sites = _BINDING_SITES.get(lang, {})
    typed_form = _TYPED_BINDINGS.get(lang)
    body = function.child_by_field_name("body")
    pending = list(body.named_children) if body is not None else []
    while pending:
        node = pending.pop()
        if node.type in _SCOPE_NODE_TYPES:
            continue
        pending.extend(node.named_children)
        fields = sites.get(node.type)
        if fields is None:
            continue
        if typed_form is not None and node.type == typed_form[0]:
            _, target_field, annotation_field, value_field = typed_form
            if bind_typed(
                node.child_by_field_name(target_field),
                node.child_by_field_name(annotation_field),
                node.child_by_field_name(value_field),
            ):
                continue
        for field_name in fields:
            targets = (
                [node]
                if field_name is None
                else node.children_by_field_name(field_name)
            )
            for target in targets:
                bind_pattern(target)

    types = dict(inherited)
    for name, kinds in found.items():
        only = next(iter(kinds)) if len(kinds) == 1 else None
        if only is None:
            types.pop(name, None)
        else:
            types[name] = only
    return types


CLASS_LIKE_KINDS: frozenset[Kind] = frozenset(
    {"class", "interface", "struct", "module", "enum", "trait"}
)


def _file_node(path: str, source: str, body: str) -> NodeV1:
    # The reference implementation counts `split("\n")`, so a trailing newline
    # adds a line.
    line_count = len(source.split("\n"))
    return NodeV1(
        id=path,
        name=path.rsplit("/", 1)[-1],
        kind="file",
        path=path,
        span=f"L1-L{line_count}",
        body_hash=content_hash(source),
        body=body,
        # The reference implementation reports JavaScript `string.length`:
        # UTF-16 code units.
        chars=len(source.encode("utf-16-le")) // 2,
    )


def _clean_signature(raw: str) -> str | None:
    """Collapse a definition header and drop its trailing `{ : = =>`.

    Mirrors the reference implementation: one trailing token is stripped,
    so `class Widget(Base):` becomes `class Widget(Base)`.
    """
    signature = re.sub(r"\s+", " ", raw).strip()
    signature = re.sub(r"(=>|[{:=])\s*$", "", signature).strip()
    return signature or None


def _signature(node: tree_sitter.Node, source_bytes: bytes) -> str | None:
    """The definition's header: its source up to where its body starts."""
    header_end = node.end_byte
    if node.type == "variable_declarator":
        # `const f = (a) => …`: the header runs to the function's body.
        value = node.child_by_field_name("value")
        if value is not None and value.type in FUNCTION_VALUE_TYPES:
            value_body = value.child_by_field_name("body")
            if value_body is not None:
                header_end = value_body.start_byte
    else:
        body = node.child_by_field_name("body")
        if body is not None:
            header_end = body.start_byte
    return _clean_signature(
        source_bytes[node.start_byte : header_end].decode("utf-8")
    )


def _heritage(
    node: tree_sitter.Node, lang: Language, source_bytes: bytes
) -> list[tuple[Relation, str]]:
    """Base types a class declaration names, as in the reference implementation.

    Only bare names count (`class W(Base)`, `extends Base`); a qualified,
    called, or subscripted base (`mod.Base`, `mixin(Base)`, `Generic[T]`)
    cannot be bound by name. JavaScript's grammar puts the base directly
    under `class_heritage`; the TypeScript grammars wrap it in an
    `extends_clause` or `implements_clause`.
    """

    def text(child: tree_sitter.Node) -> str:
        return source_bytes[child.start_byte : child.end_byte].decode("utf-8")

    if lang == "python":
        supers = node.child_by_field_name("superclasses")
        return [
            ("extends", text(child))
            for child in (supers.named_children if supers else [])
            if child.type == "identifier"
        ]
    if lang not in ("typescript", "tsx"):
        return []
    heritage = next(
        (c for c in node.named_children if c.type == "class_heritage"), None
    )
    if heritage is None:
        return []
    bases: list[tuple[Relation, str]] = []
    for clause in heritage.named_children:
        if clause.type in ("extends_clause", "implements_clause"):
            relation: Relation = (
                "implements"
                if clause.type == "implements_clause"
                else "extends"
            )
            names = clause.named_children
        else:
            relation, names = "extends", [clause]
        bases.extend(
            (relation, text(name))
            for name in names
            if name.type in ("identifier", "type_identifier")
        )
    return bases


def _is_exported(
    node: tree_sitter.Node,
    name: str,
    lang: Language,
    enclosing_kind: Kind | None,
) -> bool:
    """Whether a definition is visible outside its module.

    Python: no leading underscore; JavaScript: inside an `export` statement
    (both as in the reference implementation). Rust: a `pub` visibility
    modifier, or a member of a trait. The reference implementation indexes
    Rust in its generic tier, which marks every definition exported.
    Deferred languages default to exported.
    """
    if lang == "python":
        return not name.startswith("_")
    if lang in ("typescript", "tsx"):
        ancestor = node.parent
        while ancestor is not None:
            if ancestor.type == "export_statement":
                return True
            ancestor = ancestor.parent
        return False
    if lang == "rust":
        # Items of `impl Trait for Type` take no `pub`: they are as visible
        # as the trait and type. Only a bare `pub` exports; `pub(crate)`,
        # `pub(super)`, `pub(self)`, and `pub(in …)` stay inside the crate.
        impl = node.parent.parent if node.parent else None
        if (
            impl is not None
            and impl.type == "impl_item"
            and impl.child_by_field_name("trait") is not None
        ):
            return True
        return enclosing_kind == "trait" or any(
            child.type == "visibility_modifier" and child.text == b"pub"
            for child in node.children
        )
    return True


def extract_file(path: str, source: str) -> ExtractResult:
    """Parse a source file into graph nodes and unresolved relationships."""
    if is_container_config_file(path):
        return ExtractResult(
            nodes=[_file_node(path, source, search_body(source))], raw_edges=[]
        )

    lang = language_of(path)
    if not lang:
        return ExtractResult(nodes=[], raw_edges=[])

    if lang not in GRAMMAR_LOADERS:
        return ExtractResult(nodes=[], raw_edges=[])

    source_bytes = source.encode("utf-8")
    if language_label_of(path) in ("javascript", "jsx"):
        tree = _parse_javascript(source_bytes)
    else:
        tree = get_parser(lang).parse(source_bytes)
    line_starts = [0]
    line_starts.extend(
        index + 1
        for index, byte in enumerate(source_bytes)
        if byte == ord("\n")
    )

    def line_number(byte_offset: int) -> int:
        return bisect_right(line_starts, byte_offset)

    nodes: list[NodeV1] = []
    edges: list[RawEdge] = []
    node_ids: set[str] = set()
    generic_nodes: dict[tuple[int, int, str], NodeV1] = {}

    def mint_node_id(base_id: str) -> str:
        candidate = base_id
        suffix = 2
        while candidate in node_ids:
            candidate = f"{base_id}~{suffix}"
            suffix += 1
        node_ids.add(candidate)
        return candidate

    generic_items = generic_declarations(tree.root_node, source_bytes, lang)
    for declaration, kind in generic_items:
        name_node = (
            declaration.child_by_field_name("left")
            if lang == "python"
            else declaration.child_by_field_name("name")
        )
        if lang in {"c", "cpp"}:
            name_node = _declaration_name(declaration)
        if lang == "swift":
            name_node = _swift_property_name(declaration)
        if lang == "go" and declaration.named_children:
            name_node = declaration.named_children[0]
            if name_node.type == "identifier_list" and name_node.named_children:
                name_node = name_node.named_children[0]
        if lang == "kotlin":
            variable_declaration = next(
                (
                    child
                    for child in declaration.named_children
                    if child.type == "variable_declaration"
                ),
                None,
            )
            name_node = (
                variable_declaration.named_children[0]
                if variable_declaration and variable_declaration.named_children
                else None
            )
        if name_node is None:
            continue
        name = source_bytes[name_node.start_byte : name_node.end_byte].decode(
            "utf-8"
        )
        declaration_text = source_bytes[
            declaration.start_byte : declaration.end_byte
        ].decode("utf-8")
        node_id = mint_node_id(f"{path}#{name}")
        generic_node = NodeV1(
            id=node_id,
            name=name,
            kind=kind,
            path=path,
            span=(
                f"L{line_number(declaration.start_byte)}-"
                f"L{line_number(declaration.end_byte)}"
            ),
            body_hash=content_hash(declaration_text),
            signature=_signature(declaration, source_bytes),
            exported=_is_exported(declaration, name, lang, None),
            body=search_body(declaration_text),
        )
        generic_nodes[
            (declaration.start_byte, declaration.end_byte, declaration.type)
        ] = generic_node
        nodes.append(generic_node)
        edges.append(
            RawEdge(
                source=path, relation="contains", file=path, target_id=node_id
            )
        )

    kinds = KINDS_BY_LANG[lang]
    call_types = CALL_TYPES[lang]

    def walk(node: tree_sitter.Node, ctx: _WalkCtx) -> None:
        pending: list[tuple[tree_sitter.Node, _WalkCtx]] = [(node, ctx)]
        while pending:
            node, ctx = pending.pop()
            binding_scope = ctx.import_scope
            if lang == "rust" and node.type == "use_declaration":
                for (
                    rust_specifier,
                    local_name,
                    imported_name,
                    is_wildcard,
                ) in _rust_use_bindings(node, source_bytes):
                    edges.append(
                        RawEdge(
                            source=path,
                            relation="imports",
                            file=path,
                            specifier=rust_specifier,
                        )
                    )
                    edges.append(
                        RawEdge(
                            source=path,
                            relation="imports",
                            file=path,
                            specifier=rust_specifier,
                            name=local_name,
                            imported_name=imported_name,
                            is_wildcard=is_wildcard,
                            scope_id=binding_scope,
                        )
                    )
            elif node.type == "import_statement" or (
                lang == "python" and node.type == "import_from_statement"
            ):
                import_specifier = _import_specifier(node, source_bytes, lang)
                if import_specifier:
                    edges.append(
                        RawEdge(
                            source=path,
                            relation="imports",
                            file=path,
                            specifier=import_specifier,
                        )
                    )
                    for local_name, imported_name in _imported_symbols(
                        node, source_bytes, lang
                    ):
                        edges.append(
                            RawEdge(
                                source=path,
                                relation="imports",
                                file=path,
                                specifier=import_specifier,
                                name=local_name,
                                imported_name=imported_name,
                                scope_id=binding_scope,
                            )
                        )

            elif lang in ("typescript", "tsx") and node.type == (
                "call_expression"
            ):
                required = _commonjs_require(node, source_bytes)
                if required:
                    require_specifier, require_bindings = required
                    edges.append(
                        RawEdge(
                            source=path,
                            relation="imports",
                            file=path,
                            specifier=require_specifier,
                        )
                    )
                    edges.extend(
                        RawEdge(
                            source=path,
                            relation="imports",
                            file=path,
                            specifier=require_specifier,
                            name=local_name,
                            imported_name=imported_name,
                            scope_id=binding_scope,
                        )
                        for local_name, imported_name in require_bindings
                    )

            if lang == "rust" and node.type == "impl_item":
                impl_owner = _rust_impl_type_name(node, source_bytes)
                impl_ctx = ctx
                if impl_owner:
                    impl_scope = (*ctx.scope, impl_owner)
                    candidate_owner_id = f"{path}#{'.'.join(impl_scope)}"
                    impl_ctx = _WalkCtx(
                        parent_id=(
                            candidate_owner_id
                            if candidate_owner_id in node_ids
                            else ctx.parent_id
                        ),
                        owner_name=impl_owner,
                        scope=impl_scope,
                        enclosing_kind=ctx.enclosing_kind,
                        import_scope=ctx.import_scope,
                        rust_impl=True,
                    )
                pending.extend(
                    (child, impl_ctx) for child in reversed(node.children)
                )
                continue

            if lang == "rust" and node.type == "mod_item":
                module_name_node = _declaration_name(node)
                module_path = _rust_path_attribute(node, source_bytes)
                if module_name_node and module_path is not None:
                    module_name = source_bytes[
                        module_name_node.start_byte : module_name_node.end_byte
                    ].decode("utf-8")
                    target_path = str(
                        PurePosixPath(path).parent / PurePosixPath(module_path)
                    )
                    edges.append(
                        RawEdge(
                            source=path,
                            relation="imports",
                            file=path,
                            name=module_name,
                            rust_module_name=module_name,
                            rust_module_target=target_path,
                        )
                    )

            generic_node = generic_nodes.get(
                (node.start_byte, node.end_byte, node.type)
            )
            if generic_node:
                generic_ctx = _WalkCtx(
                    parent_id=generic_node.id,
                    owner_name=ctx.owner_name,
                    scope=(*ctx.scope, generic_node.name),
                    enclosing_kind=generic_node.kind,
                    import_scope=(
                        generic_node.id
                        if generic_node.kind == "function"
                        else ctx.import_scope
                    ),
                )
                pending.extend(
                    (child, generic_ctx) for child in reversed(node.children)
                )
                continue

            assigned = (
                _member_assigned_function(node, source_bytes)
                if lang in ("typescript", "tsx")
                and node.type == "assignment_expression"
                else None
            )
            if assigned is not None:
                owner_parts, name = assigned
                kind: Kind = "method" if owner_parts else "function"
                owner = ".".join(owner_parts) or None
                scope = (*ctx.scope, *owner_parts, name)
                node_id = mint_node_id(f"{path}#{'.'.join(scope)}")
                body_str = source_bytes[node.start_byte : node.end_byte].decode(
                    "utf-8"
                )
                nodes.append(
                    NodeV1(
                        id=node_id,
                        name=name,
                        kind=kind,
                        path=path,
                        span=(
                            f"L{line_number(node.start_byte)}-"
                            f"L{line_number(node.end_byte)}"
                        ),
                        body_hash=content_hash(body_str),
                        owner=owner,
                        signature=_signature(node, source_bytes),
                        exported=owner is None,
                        body=search_body(body_str),
                    )
                )
                edges.append(
                    RawEdge(
                        source=ctx.parent_id,
                        relation="contains",
                        file=path,
                        target_id=node_id,
                    )
                )
                child_ctx = _WalkCtx(
                    parent_id=node_id,
                    owner_name=owner or ctx.owner_name,
                    scope=scope,
                    enclosing_kind=kind,
                    import_scope=node_id,
                    receiver_types=_receiver_types(
                        node.child_by_field_name("right") or node,
                        lang,
                        source_bytes,
                        ctx.receiver_types,
                        owner or ctx.owner_name,
                    ),
                )
                pending.extend(
                    (child, child_ctx) for child in reversed(node.children)
                )
                continue

            node_kind = kinds.get(node.type)
            if (
                node_kind is None
                and lang in ("typescript", "tsx")
                and node.type == "variable_declarator"
            ):
                declared = node.child_by_field_name("name")
                value = node.child_by_field_name("value")
                if (
                    declared is not None
                    and declared.type == "identifier"
                    and value is not None
                    and value.type in FUNCTION_VALUE_TYPES
                ):
                    node_kind = "function"

            if node_kind is not None:
                kind = node_kind
                # Only a def directly in a class body is a method; one nested in
                # a method is a local function, as in the reference
                # implementation.
                if (
                    lang == "python"
                    and kind == "function"
                    and ctx.enclosing_kind == "class"
                ):
                    kind = "method"
                # Rust functions in an impl or a trait body (default methods)
                # are methods of that type or trait.
                if (
                    lang == "rust"
                    and kind == "function"
                    and (ctx.rust_impl or ctx.enclosing_kind == "trait")
                ):
                    kind = "method"
                name_node = _declaration_name(node)
                if name_node and not _is_recovery_artifact(
                    lang, node, name_node, source_bytes
                ):
                    name = source_bytes[
                        name_node.start_byte : name_node.end_byte
                    ].decode("utf-8")
                    scope = (*ctx.scope, name)
                    node_id = mint_node_id(f"{path}#{'.'.join(scope)}")
                    span = (
                        f"L{line_number(node.start_byte)}-"
                        f"L{line_number(node.end_byte)}"
                    )
                    body_bytes = source_bytes[node.start_byte : node.end_byte]
                    body_str = body_bytes.decode("utf-8")
                    body_hash = content_hash(body_str)
                    nodes.append(
                        NodeV1(
                            id=node_id,
                            name=name,
                            kind=kind,
                            path=path,
                            span=span,
                            body_hash=body_hash,
                            # As in the reference implementation, only methods
                            # carry an owner.
                            owner=ctx.owner_name if kind == "method" else None,
                            signature=_signature(node, source_bytes),
                            exported=_is_exported(
                                node, name, lang, ctx.enclosing_kind
                            ),
                            body=search_body(body_str),
                        )
                    )
                    edges.append(
                        RawEdge(
                            source=ctx.parent_id,
                            relation="contains",
                            file=path,
                            target_id=node_id,
                        )
                    )
                    if kind == "class":
                        edges.extend(
                            RawEdge(
                                source=node_id,
                                relation=relation,
                                file=path,
                                name=base,
                            )
                            for relation, base in _heritage(
                                node, lang, source_bytes
                            )
                        )

                    child_ctx = _WalkCtx(
                        parent_id=node_id,
                        owner_name=(
                            name if kind in CLASS_LIKE_KINDS else ctx.owner_name
                        ),
                        scope=scope,
                        enclosing_kind=kind,
                        import_scope=(
                            node_id
                            if kind in ("function", "method")
                            else ctx.import_scope
                        ),
                        receiver_types=(
                            _receiver_types(
                                node.child_by_field_name("value") or node
                                if node.type == "variable_declarator"
                                else node,
                                lang,
                                source_bytes,
                                ctx.receiver_types,
                                ctx.owner_name,
                            )
                            if kind in ("function", "method")
                            else ctx.receiver_types
                        ),
                    )
                    pending.extend(
                        (child, child_ctx) for child in reversed(node.children)
                    )
                    continue

            if node.type in call_types:
                func_node = node.child_by_field_name("function")
                if func_node:
                    member_fields = {
                        "member_expression": ("property", "object"),
                        "attribute": ("attribute", "object"),
                        "field_expression": ("field", "value"),
                        "selector_expression": ("field", "operand"),
                    }
                    if func_node.type in member_fields:
                        name_field, object_field = member_fields[func_node.type]
                        prop = func_node.child_by_field_name(name_field)
                        obj = func_node.child_by_field_name(object_field)
                        if prop and obj:
                            prop_name = source_bytes[
                                prop.start_byte : prop.end_byte
                            ].decode("utf-8")
                            obj_name = source_bytes[
                                obj.start_byte : obj.end_byte
                            ].decode("utf-8")
                            edges.append(
                                RawEdge(
                                    source=ctx.parent_id,
                                    relation="calls",
                                    file=path,
                                    name=prop_name,
                                    via_member=True,
                                    specifier=obj_name,
                                    recv_type=(
                                        ctx.receiver_types.get(obj_name)
                                        if obj.type == "identifier"
                                        else None
                                    ),
                                )
                            )
                    elif (
                        lang == "rust" and func_node.type == "scoped_identifier"
                    ):
                        # `Type::f()` / `Self::f()`: an associated function
                        # of a type, resolved against that type's methods.
                        path_node = func_node.child_by_field_name("path")
                        fn_node = func_node.child_by_field_name("name")
                        if path_node and fn_node:
                            type_name = (
                                source_bytes[
                                    path_node.start_byte : path_node.end_byte
                                ]
                                .decode("utf-8")
                                .rsplit("::", 1)[-1]
                            )
                            if type_name == "Self":
                                type_name = ctx.owner_name or ""
                            fn_name = source_bytes[
                                fn_node.start_byte : fn_node.end_byte
                            ].decode("utf-8")
                            if type_name:
                                edges.append(
                                    RawEdge(
                                        source=ctx.parent_id,
                                        relation="calls",
                                        file=path,
                                        name=fn_name,
                                        via_member=True,
                                        recv_type=type_name,
                                    )
                                )
                    elif func_node.type == "identifier":
                        name = source_bytes[
                            func_node.start_byte : func_node.end_byte
                        ].decode("utf-8")
                        edges.append(
                            RawEdge(
                                source=ctx.parent_id,
                                relation="calls",
                                file=path,
                                name=name,
                                via_member=False,
                            )
                        )

            pending.extend((child, ctx) for child in reversed(node.children))

    file_node = _file_node(path, source, "")
    node_ids.add(file_node.id)
    nodes.insert(0, file_node)
    walk(tree.root_node, _WalkCtx(parent_id=path))
    file_node.body = file_residual(source, nodes[1:])
    return ExtractResult(nodes=nodes, raw_edges=edges)
