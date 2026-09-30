"""Command-line interface for Bytely."""

import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from bytely.graph.build import build_graph
from bytely.graph.check import check_graph
from bytely.hooks.handlers import EVENTS, run_hook
from bytely.hooks.metrics import format_session_stats
from bytely.hooks.state import latest_session
from bytely.hosts.registry import HOST_IDS, HOSTS
from bytely.hosts.wiring import Change, run_init, run_uninstall
from bytely.mcp.server import serve
from bytely.query.service import (
    QueryError,
    Workspace,
    ask_text,
    callers_text,
    grep_text,
    map_text,
    repo_root,
    skeleton_text,
)


@click.group()
@click.version_option()
@click.option(
    "--dir",
    "context_dir",
    help="context graph directory (default: <repo>/bytely)",
)
@click.option(
    "--provider",
    help=(
        "LLM wire format: openai | anthropic | litellm | orcarouter "
        "(env BYTELY_PROVIDER)"
    ),
)
@click.option("--model", help="model id for the LLM pass (env BYTELY_MODEL)")
@click.option("--api-key", help="provider API key (env BYTELY_API_KEY)")
@click.option(
    "--base-url", help="OpenAI-compatible endpoint URL (env BYTELY_BASE_URL)"
)
@click.pass_context
def main(
    ctx: click.Context,
    context_dir: str | None,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Build a repository's context graph and keep it synchronized with code."""
    # Output includes `·` and `←`; Windows pipes default to a legacy code
    # page that cannot encode them, and agents read our output through pipes.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and (
            stream.encoding.lower().replace("-", "") != "utf8"
        ):
            stream.reconfigure(encoding="utf-8")
    ctx.ensure_object(dict)
    ctx.obj["context_dir"] = context_dir
    ctx.obj["provider"] = provider
    ctx.obj["model"] = model
    ctx.obj["api_key"] = api_key
    ctx.obj["base_url"] = base_url


@main.command()
@click.argument("directory", required=False)
@click.option("--deep", is_flag=True, help="Run LLM meaning pass")
@click.option(
    "--include",
    "include_patterns",
    multiple=True,
    help="Include matching source paths (repeatable)",
)
@click.option(
    "--exclude",
    "exclude_patterns",
    multiple=True,
    help="Exclude matching source paths (repeatable)",
)
@click.option(
    "-j",
    "--concurrency",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="With --deep: files summarized in parallel",
)
@click.option(
    "--allow-partial",
    is_flag=True,
    help="With --deep: exit 0 even when some summaries failed",
)
@click.pass_context
def build(
    ctx: click.Context,
    directory: str | None,
    deep: bool,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
    concurrency: int,
    allow_partial: bool,
) -> None:
    """Build or update the context graph for a repository.

    With --deep, an LLM also summarizes every definition (with its crux,
    the few lines that matter most) and synthesizes concept nodes
    (bytely/concepts/). Both resume from what earlier runs cached.
    """
    root = directory or "."
    context_dir = ctx.obj.get("context_dir")
    if deep:
        _build_deep(
            ctx,
            root,
            context_dir,
            include_patterns or None,
            exclude_patterns or None,
            concurrency,
            allow_partial,
        )
        return
    try:
        result = build_graph(
            root,
            context_dir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(f"Built graph for {root}")
    click.echo(
        f"Files: {result.files}; nodes: {result.nodes}; edges: {result.edges}; "
        f"extraction cache: {result.cache_hits} hits, "
        f"{result.cache_misses} misses"
    )
    click.echo(f"Output: {result.graph_json}")


def _build_deep(
    ctx: click.Context,
    root: str,
    context_dir: str | None,
    include_patterns: tuple[str, ...] | None,
    exclude_patterns: tuple[str, ...] | None,
    concurrency: int,
    allow_partial: bool,
) -> None:
    """`build --deep`: concepts, then the graph with its meaning layer."""
    from bytely.ai.deep import run_deep
    from bytely.ai.providers import ConfigError, needs_key, resolve_config

    try:
        config = resolve_config(
            ctx.obj.get("provider"),
            ctx.obj.get("model"),
            ctx.obj.get("api_key"),
            ctx.obj.get("base_url"),
        )
    except ConfigError as error:
        raise click.ClickException(str(error)) from error
    if needs_key(config):
        raise click.ClickException(
            f"--deep needs an API key for {config.provider}: set "
            "BYTELY_API_KEY (or pass --api-key), plus BYTELY_PROVIDER / "
            "BYTELY_BASE_URL / BYTELY_MODEL for your provider. Without "
            "--deep, `bytely build` needs no key."
        )
    click.echo(
        f"Deep build with {config.provider}:{config.model}"
        + (f" via {config.base_url}" if config.base_url else ""),
        err=True,
    )
    labels = {
        "summarize": "reading files",
        "synthesize": "synthesizing concepts",
        "enrich": "summarizing symbols",
    }

    def progress(phase: str, index: int, total: int, item: str) -> None:
        click.echo(
            f"\r{labels[phase]} {index}/{total}: {item[:48]:<48}",
            err=True,
            nl=False,
        )
        if index == total:
            click.echo("", err=True)

    try:
        result = run_deep(
            root,
            context_dir,
            config,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            concurrency=concurrency,
            progress=progress,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    c, g = result.concepts, result.graph
    click.echo(
        f"Concepts: {c.nodes} nodes, {c.links} links from {c.files} files "
        f"({c.summarized} summarized, {c.cached} cached)"
    )
    click.echo(f"Graph: {g.nodes} nodes, {g.edges} edges from {g.files} files")
    m = g.meaning
    if m is not None:
        click.echo(
            f"Meaning: {m.computed} computed, {m.cached} cached, "
            f"{m.stale} stale, {m.pending} pending"
        )
    click.echo(f"Output: {g.context_dir}")
    if not result.degraded:
        return
    for failure in [*c.errors, *(m.errors if m else [])][:20]:
        click.echo(f"  ✗ {failure}", err=True)
    total = (m.ready + m.stale + m.pending) if m else 0
    ready = m.ready if m else 0
    percent = round(ready / total * 100) if total else 0
    click.echo("", err=True)
    click.echo(
        "✗ The deep pass did not complete: the meaning layer is incomplete.",
        err=True,
    )
    if c.fatal:
        click.echo(f"  concepts: {c.fatal}", err=True)
    if c.kept_previous:
        click.echo(
            f"  concepts: kept the previous concept graph ({c.nodes} "
            "nodes) rather than replace it with a partial one.",
            err=True,
        )
    if m and m.fatal:
        click.echo(f"  summaries: {m.fatal}", err=True)
    if m and m.failed_files:
        skipped = (
            f", {m.skipped_files} never attempted" if m.skipped_files else ""
        )
        click.echo(
            f"  {m.failed_files} file(s) failed to summarize{skipped}.",
            err=True,
        )
    click.echo(
        f"  Meaning coverage: {ready}/{total} definitions ({percent}%).\n"
        "  Nothing computed was lost: run `bytely build --deep` again to "
        "resume.\n  --allow-partial accepts a partial meaning layer and "
        "exits 0.",
        err=True,
    )
    if not allow_partial:
        ctx.exit(1)


@main.command()
@click.argument("directory", required=False)
@click.option(
    "--include",
    "include_patterns",
    multiple=True,
    help="Include matching source paths (repeatable)",
)
@click.option(
    "--exclude",
    "exclude_patterns",
    multiple=True,
    help="Exclude matching source paths (repeatable)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Re-extract every file instead of trusting the extraction cache",
)
@click.pass_context
def check(
    ctx: click.Context,
    directory: str | None,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
    no_cache: bool,
) -> None:
    """Check whether the saved graph matches the repository."""
    try:
        fresh, message = check_graph(
            str(repo_root(directory)),
            ctx.obj.get("context_dir"),
            # Without patterns, compare with the selection the graph was
            # built from, so a scoped graph is not reported stale.
            include_patterns=include_patterns or None,
            exclude_patterns=exclude_patterns or None,
            use_cache=not no_cache,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(message)
    if not fresh:
        ctx.exit(1)


def _workspace(ctx: click.Context, root: str | None) -> Workspace:
    return Workspace(repo_root(root), ctx.obj.get("context_dir"))


def _run(query: Callable[[], str]) -> None:
    """Print a query's answer, or its error as a CLI error."""
    try:
        text = query()
    except QueryError as error:
        raise click.ClickException(str(error)) from error
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(text, nl=False)


_ROOT = click.option(
    "--root", "root", help="Repository root (default: the nearest with a graph)"
)
_IN = click.option("--in", "scope", help="Only code under this path")


@main.command(name="map")
@click.argument("directory", required=False)
@click.option(
    "--max-dirs",
    type=click.IntRange(min=1),
    default=12,
    show_default=True,
    help="Top-level folders to list",
)
@click.pass_context
def map_command(
    ctx: click.Context, directory: str | None, max_dirs: int
) -> None:
    """Print a short orientation: folders, hubs, and hotspots."""
    workspace = _workspace(ctx, directory)
    _run(lambda: map_text(workspace, max_dirs))


@main.command()
@click.argument("file")
@_ROOT
@click.pass_context
def skeleton(ctx: click.Context, file: str, root: str | None) -> None:
    """List a file's definitions with their spans and signatures."""
    workspace = _workspace(ctx, root)
    _run(lambda: skeleton_text(workspace, file))


@main.command()
@click.argument("symbol")
@click.option(
    "--direction",
    type=click.Choice(["in", "out"]),
    default="in",
    show_default=True,
    help="in: who uses SYMBOL; out: what SYMBOL uses",
)
@click.option(
    "--depth",
    default="1",
    show_default=True,
    help="Levels to walk, or 'all' for the whole connected closure",
)
@_IN
@_ROOT
@click.pass_context
def callers(
    ctx: click.Context,
    symbol: str,
    direction: str,
    depth: str,
    scope: str | None,
    root: str | None,
) -> None:
    """Show who calls SYMBOL (or what it calls), from exact graph edges.

    SYMBOL may be a file path, which stands for everything defined in it.
    """
    workspace = _workspace(ctx, root)
    _run(
        lambda: callers_text(
            workspace, symbol, direction=direction, depth=depth, scope=scope
        )
    )


@main.command()
@click.argument("pattern")
@click.option("--fixed", is_flag=True, help="Treat PATTERN as a literal")
@click.option("-i", "ignore_case", is_flag=True, help="Ignore case")
@_IN
@_ROOT
@click.pass_context
def grep(
    ctx: click.Context,
    pattern: str,
    fixed: bool,
    ignore_case: bool,
    scope: str | None,
    root: str | None,
) -> None:
    """Find every match in the indexed files, grouped by enclosing symbol."""
    workspace = _workspace(ctx, root)
    _run(
        lambda: grep_text(
            workspace,
            pattern,
            fixed=fixed,
            ignore_case=ignore_case,
            scope=scope,
        )
    )


@main.command()
@click.argument("question")
@click.option(
    "-n", "limit", type=click.IntRange(min=1), default=8, show_default=True
)
@click.option("--source", is_flag=True, help="Show each hit's first lines")
@click.option("--full", is_flag=True, help="Show each hit's whole definition")
@_IN
@_ROOT
@click.pass_context
def ask(
    ctx: click.Context,
    question: str,
    limit: int,
    source: bool,
    full: bool,
    scope: str | None,
    root: str | None,
) -> None:
    """Rank definitions for a question, with the code if asked."""
    workspace = _workspace(ctx, root)
    _run(
        lambda: ask_text(
            workspace,
            question,
            limit=limit,
            source=source,
            full=full,
            scope=scope,
        )
    )


@main.command()
@click.argument(
    "directory",
    required=False,
    type=click.Path(exists=True, file_okay=False),
)
@click.pass_context
def mcp(ctx: click.Context, directory: str | None) -> None:
    """Serve the graph's queries to agents over MCP (stdio JSON-RPC).

    The server never indexes a directory by itself: where there is no graph
    yet, it offers no tools until `bytely build` has run.
    """
    serve(
        Workspace(
            repo_root(directory), ctx.obj.get("context_dir"), create=False
        )
    )


def _stdin_json() -> dict[str, Any]:
    try:
        data = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@main.command()
@click.argument("event", type=click.Choice(EVENTS))
@click.option(
    "--user-level",
    is_flag=True,
    help="Installed in user settings: defer to the project's own hooks",
)
def hook(event: str, user_level: bool) -> None:
    """Handle an agent host's hook (reads the host's JSON on stdin).

    Hosts call this; it never fails the host's turn: errors are
    reported on stderr and the exit code is always 0.
    """
    try:
        out = run_hook(event, _stdin_json(), user_level=user_level)
    except Exception as error:  # noqa: BLE001 - a hook must not break a turn
        click.echo(f"bytely hook {event}: {error}", err=True)
        return
    if out:
        click.echo(out, nl=False)


@main.command()
def statusline() -> None:
    """Print Claude Code's status line (reads the host's JSON on stdin)."""
    from bytely.hooks.statusline import render

    try:
        click.echo(render(_stdin_json()), nl=False)
    except Exception:  # noqa: BLE001 - a broken status line helps no one
        click.echo("◆ bytely", nl=False)


@main.command()
@click.argument(
    "directory",
    required=False,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--json", "as_json", is_flag=True, help="Print raw JSON")
def stats(directory: str | None, as_json: bool) -> None:
    """Show the latest agent session's bytely reads and tokens saved."""
    session = latest_session(repo_root(directory))
    if as_json:
        click.echo(json.dumps(session, indent=2))
    else:
        click.echo(format_session_stats(session))


def _print_changes(changes: list[Change], root: Path, dry_run: bool) -> None:
    shown = [
        change
        for change in changes
        if change.action not in ("unchanged", "absent")
    ]
    if not shown:
        click.echo("Nothing to change; already up to date.")
        return
    verb = "would be " if dry_run else ""
    for change in shown:
        try:
            where = change.path.relative_to(root).as_posix()
        except ValueError:
            where = str(change.path)
        scope = " (global)" if change.scope == "global" else ""
        click.echo(
            f"  {verb}{change.action}: {where} — {change.what} "
            f"[{change.host}]{scope}"
        )


@main.command()
@click.argument(
    "directory",
    required=False,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--agents",
    help=f"Comma-separated hosts to wire (default: detected). "
    f"Known: {', '.join(HOST_IDS)}",
)
@click.option(
    "--all-agents",
    "--all",
    "all_hosts",
    is_flag=True,
    help="Wire every known host, detected or not",
)
@click.option(
    "--no-agents",
    is_flag=True,
    help="Claude Code only; skip the other hosts",
)
@click.option(
    "--list-agents", is_flag=True, help="List known host ids and exit"
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Accepted for compatibility (there is no interactive picker)",
)
@click.option("--no-mcp", is_flag=True, help="Skip MCP server registration")
@click.option(
    "--no-hooks",
    is_flag=True,
    help="Skip hooks (session orientation, prompt pointers, edit blast "
    "radius, background sync) and the status line",
)
@click.option(
    "--no-statusline",
    is_flag=True,
    help="Keep Claude Code's status line as it is "
    "(or set BYTELY_NO_STATUSLINE=1)",
)
@click.option(
    "--no-global",
    is_flag=True,
    help="Never write outside the repository (skips user-level configs: "
    "Claude Code, Codex, Antigravity)",
)
@click.option("--no-build", is_flag=True, help="Do not build a missing graph")
@click.option("--dry-run", is_flag=True, help="Show what would change")
def init(
    directory: str | None,
    agents: str | None,
    all_hosts: bool,
    no_agents: bool,
    list_agents: bool,
    yes: bool,
    no_mcp: bool,
    no_hooks: bool,
    no_statusline: bool,
    no_global: bool,
    no_build: bool,
    dry_run: bool,
) -> None:
    """Wire bytely into your AI coding tools: instructions and MCP server.

    Claude Code is always wired; other hosts are the ones detected on this
    machine or in the repository, unless --agents or --all-agents says
    otherwise.
    """
    if list_agents:
        for host in HOSTS:
            click.echo(f"{host.id:12} {host.name}")
        return
    root = Path(directory or Path.cwd()).resolve()
    report = run_init(
        root,
        Path.home(),
        agents=[agent.strip() for agent in agents.split(",") if agent.strip()]
        if agents
        else None,
        all_hosts=all_hosts,
        others=not no_agents,
        mcp=not no_mcp,
        hooks=not no_hooks,
        statusline=not no_statusline,
        global_scope=not no_global,
        build=not no_build,
        apply=not dry_run,
    )
    if report.unknown:
        raise click.ClickException(
            f"Unknown host(s): {', '.join(report.unknown)}. "
            f"Known: {', '.join(HOST_IDS)}"
        )
    if not report.selected:
        click.echo(
            "No AI coding tools detected. "
            "Name them with --agents, or use --all."
        )
    else:
        click.echo(f"Wiring: {', '.join(report.selected)}")
    _print_changes(report.changes, root, dry_run)
    if any(change.action == "skipped-foreign" for change in report.changes):
        click.echo(
            "Note: a status line is already configured, so it was left as "
            "it is. To use bytely's, set statusLine's command to "
            "`bytely statusline` in .claude/settings.json."
        )
    if report.built:
        click.echo("Built the graph (bytely/).")
    if not report.on_path and not no_mcp:
        click.echo(
            "Note: `bytely` is not on PATH, so hosts cannot start the MCP "
            "server yet; install it so the `bytely` command is available."
        )


@main.command()
@click.argument(
    "directory",
    required=False,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--no-global",
    is_flag=True,
    help="Leave user-level configs (Codex, Antigravity) alone",
)
@click.option(
    "--keep-cache",
    "--keep-graph",
    "keep_graph",
    is_flag=True,
    help="Keep the bytely/ graph folder and its .gitignore entry",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Actually remove (without it, only shows what would be removed)",
)
def uninstall(
    directory: str | None, no_global: bool, keep_graph: bool, yes: bool
) -> None:
    """Remove everything `bytely init` wrote, keeping your own content.

    Without --yes this only lists what would be removed.
    """
    root = Path(directory or Path.cwd()).resolve()
    changes = run_uninstall(
        root,
        Path.home(),
        global_scope=not no_global,
        keep_graph=keep_graph,
        apply=yes,
    )
    _print_changes(changes, root, dry_run=not yes)
    if not yes and any(
        change.action not in ("unchanged", "absent") for change in changes
    ):
        click.echo("Run again with --yes to remove these.")


if __name__ == "__main__":
    main()
