"""The one instruction text agents get, wrapped in each host's format.

Change the wording here only; the renderers just add each host's
front-matter.
"""

from __future__ import annotations


def instruction_body() -> str:
    """Markdown telling an agent how to use bytely in this repository."""
    return """## bytely — code graph for this repository

This repository is indexed by bytely: a graph of every function, class, and
method, with exact `file:line` spans and who calls what. It refreshes itself
before every query, so answers include uncommitted edits.

Before grepping or opening source files, ask the graph. Call the MCP tools
(`bytely_*`) directly; the CLI forms are for hosts without MCP. Don't wrap
either in shell pipes: output is already capped.

- **Where is X / how does Y work** → `bytely_find_code` (MCP) or
  `bytely ask "<question>" --source`: ranked definitions with their code
  inlined. The top hit is usually the answer; open a file only at the span it
  names, and only when the shown lines are not enough.
- **Every occurrence of a pattern** → `bytely_find_all` or
  `bytely grep "<literal>"`: exhaustive, grouped by enclosing function. Use it
  instead of `ask` when you need all matches, not the best ones.
- **Who calls X / what does X call / what breaks if X changes** →
  `bytely_trace_calls` or `bytely callers <symbol>` (`--direction out` for
  callees, `--depth all` for the full blast radius). Run it before renaming,
  deleting, or changing a signature. A file path works as the symbol.
- **What is in this file** → `bytely_file_api` or `bytely skeleton <file>`:
  every definition's signature and span, far cheaper than reading the file.
- **New to the repository** → `bytely_repo_map` or `bytely map`: folders,
  the most-referenced symbols, and hotspots.

Reuse literal identifiers you already have (a symbol, an error message, a file
name) as the query. The per-file cards in `bytely/` (see `bytely/INDEX.md`)
list each file's definitions if you prefer to browse."""


def cursor_rule() -> str:
    """Cursor project rule (always applied)."""
    return (
        "---\n"
        "description: Use the bytely code graph before exploring source\n"
        "alwaysApply: true\n"
        "---\n"
        f"{instruction_body()}\n"
    )


def kiro_steering() -> str:
    """Kiro steering file (always included)."""
    return f"---\ninclusion: always\n---\n{instruction_body()}\n"


def plain_rule() -> str:
    """A rule file with no front-matter (Windsurf)."""
    return f"{instruction_body()}\n"


def skill() -> str:
    """A skill file (Claude Code, Grok, AdaL): loaded when relevant."""
    return (
        "---\n"
        "name: bytely\n"
        "description: This repository is indexed by bytely, a code graph with "
        "exact file:line spans and call edges. Use it for ANY task here — "
        "understanding how something works, finding where code lives, "
        "tracing callers, or scoping a change — before grepping or reading "
        "source files.\n"
        "---\n\n"
        f"{instruction_body()}\n"
    )
