# Bytely

Bytely turns a code repository into a **context graph**: every function, class,
method, and module, with its exact `file:line` span, and the imports and calls
that connect them. AI coding agents can then find code, trace callers, and scope
changes from the graph instead of grepping and reading whole files.

> **Status: alpha.** Building and querying the graph work for Python, Rust,
> and JavaScript: `build`, `check`, `map`, `skeleton`, `callers`, `grep`, and
> `ask`, every build writes a markdown card per source file, `bytely mcp`
> serves the queries to agents over MCP, and `bytely init` wires it into
> Claude Code, Cursor, Codex, Gemini, and other AI coding tools, with
> session hooks, a status line, and token-savings accounting.
> `bytely build --deep` adds an LLM meaning layer: per-symbol summaries and
> crux lines, and concept nodes.

## The problem it solves

When an AI agent works in an unfamiliar codebase, most of its time and tokens
go to *finding* things: grepping for a name, opening files, and reading
definitions it doesn't need. A graph answers the same questions directly: where
`Cache.get` is defined (and on which lines), what it calls, and who calls it.

Bytely is pure Python on top of tree-sitter, whose bindings and grammars ship
as pre-built wheels. Installing it never compiles native code, so it installs
the same way on Windows, macOS, and Linux.


## How it works

`bytely build` runs a deterministic pipeline with no network access and no LLM:

1. **Walk the repository.** Source files are listed with root and nested
   `.gitignore` rules applied (plus `.git/`, `node_modules/`, `__pycache__/`,
   and virtualenvs), then narrowed by any `--include` / `--exclude` patterns.
   Dockerfiles and Compose files are indexed as file nodes too.
2. **Extract symbols with tree-sitter.** Each file becomes nodes (`file`,
   `class`, `function`, `method`, `struct`, `enum`, `trait`, `interface`,
   `module`, `constant`, `variable`) with a line span, a signature, and a
   SHA-256 hash of the definition's exact source. Nested definitions get
   scope-qualified IDs, such as `pkg/cli.py#outer.inner` or
   `models.py#Widget.Inner.method`, and same-named siblings get stable `~2`,
   `~3` suffixes. JavaScript functions assigned to properties
   (`app.use = function …`) are definitions too.
3. **Resolve relationships** into `contains`, `imports`, `calls`, `extends`,
   and `implements` edges. This covers relative, absolute, aliased, wildcard,
   and namespace imports, CommonJS `require()`, and Rust `use` trees and `mod`
   declarations, including `#[path]`. Method calls resolve through `self`,
   through base classes, and through variables whose type the code states: an
   annotated parameter (`formatter: HelpFormatter`, `c: &Cache`) or a local
   built by a constructor (`x = Foo()`, `let c = Cache::new()`,
   `const v = new View()`). Scope is respected: a function defined inside
   another is visible only there, an import inside a function binds only
   inside it, and a bare call such as `open()` never resolves to some class's
   `open` method. A call resolved only by a unique name match elsewhere in the
   repo is labelled `inferred`; everything else is `extracted`.
4. **Cache.** Per-file extraction results are keyed by the file's content hash
   and by a fingerprint of the extractor and installed grammars, so a rebuild
   re-parses only the files that changed, and an upgrade re-parses everything.
5. **Write and verify.** The graph is written to `bytely/.graph/wiring.json`,
   with one markdown card per source file (`src/app.py` →
   `bytely/src/app.md`) listing its definitions, spans, and signatures, plus
   `bytely/INDEX.md`. `bytely check` rebuilds the graph and compares. Output
   is byte-deterministic, so any difference means the graph is stale.

Commands that read the graph (`map` and the queries below) refresh it first: they
compare a snapshot of the source tree (paths, sizes, and modification times)
with the one saved by the last build, and rebuild only if it changed.

For a monorepo, project roots are detected from markers such as
`pyproject.toml`, `package.json`, `Cargo.toml`, and `go.mod`, and recorded as
scopes in the graph.

## Language support

A language counts as supported only after it passes a verification gate:
fixtures with explicit expected output, a pinned real repository that builds,
rebuilds from cache, and checks fresh, and a reviewed comparison against a
reference implementation. Three languages are in scope:

| Language | Extensions | Verified against |
|---|---|---|
| Python | `.py`, `.pyi` | [pallets/click](https://github.com/pallets/click) 8.5.0 |
| Rust | `.rs` | [serde-rs/serde](https://github.com/serde-rs/serde) |
| JavaScript | `.js`, `.mjs`, `.cjs`, `.jsx` | [expressjs/express](https://github.com/expressjs/express) |

Every pull request runs the test suite on Linux and Windows, plus integration
tests that build and query pinned Click, Express, and ripgrep checkouts. macOS
is not verified yet.

JavaScript files are parsed with `tree-sitter-javascript`, so JSX inside a
`.js` file works. A file that grammar can't parse cleanly, such as
Flow-typed code, is retried with the TSX grammar.

Parsers for TypeScript, Go, Java, Kotlin, Swift, PHP, C, C++, and C# also
exist. They are **deferred and unverified**, not a support claim, and their
grammars are optional (see below).

## Installation

Requires Python 3.11 or newer. Bytely is not on PyPI yet; the first release
will be an alpha. Until then, install it from a checkout (below). Once it is
published:

```bash
# Recommended when uv is installed
uv tool install --prerelease allow bytely
# Or install into the active environment
uv pip install --prerelease allow bytely
# Standard pip installation is also supported
python -m pip install --pre bytely
```

To install from a checkout, or for development:

```bash
git clone <this repository> bytely
cd bytely
pip install -e .              # Python, Rust, and JavaScript
pip install -e ".[dev]"       # plus test tools and every optional grammar
```

The core install needs only `click`, `orjson`, `pathspec`, `tree-sitter`, and
the grammars of the supported languages. Other grammars are extras:

| Extra | Adds |
|---|---|
| `go`, `java`, `kotlin`, `swift`, `php`, `c`, `cpp`, `csharp` | That language's grammar; without it, its files are skipped |
| `languages` | All of the above |
| `llm` | Nothing: `--deep` needs no extra packages (kept for compatibility) |
| `agents` | Dependencies of the planned agent integrations |
| `dev` | Test and lint tools, plus `languages` |

## Usage

```bash
bytely build [DIR]            # index DIR (default: the current directory)
bytely check [DIR]            # exit 0 if the graph is current, 1 if stale
bytely check --no-cache [DIR] # re-extract every file instead of trusting the cache
bytely map [DIR]              # folders, most-referenced symbols, hotspots
bytely skeleton FILE          # a file's definitions, spans, and signatures
bytely callers SYMBOL         # who calls SYMBOL (exact graph edges)
bytely callers SYMBOL --direction out --depth 2   # what it calls, 2 levels
bytely callers SYMBOL --depth all                 # the whole blast radius
bytely grep PATTERN [-i] [--fixed] [--in DIR]     # every match, by symbol
bytely ask "how is the cache invalidated" [--source | --full] [-n 8]
# queries take --root DIR (default: the nearest folder with a graph)
bytely mcp [DIR]              # MCP server on stdin/stdout, for agents
bytely stats [DIR] [--json]   # the latest agent session: reads, tokens saved
bytely build --include "src/**" --exclude "src/vendor/**"
bytely --dir /tmp/graph build .   # write the graph somewhere else
python -m bytely --help
```

With no directory, `bytely check`, `bytely map`, and the query commands look
upward for the nearest folder that contains `bytely/.graph/wiring.json`
(queries take `--root` to point elsewhere). The `bytely/` output
directory is a local, regenerable cache, so add it to `.gitignore`.

### The meaning layer (`--deep`)

```bash
export BYTELY_API_KEY=...            # or OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY
bytely build --deep                  # OpenAI by default
bytely --provider anthropic build --deep
bytely --provider litellm build --deep                    # a local LiteLLM proxy
BYTELY_BASE_URL=https://api.groq.com/openai/v1 BYTELY_MODEL=... bytely build --deep
bytely build --deep -j 8 --allow-partial
```

`--deep` adds what structure alone cannot say, using an LLM:

- **Summaries and crux.** Every definition gets a one-sentence summary of what
  it is for, and a *crux*: the few lines that carry its core logic. Cards
  show the summaries. `ask --source` shows the crux instead of the first 8
  lines, and summaries are searched too.
- **Concept nodes.** Files are summarized, then synthesized into curated
  system and concept nodes with typed links (`uses`, `part_of`,
  `validates`, …) in `bytely/concepts/<slug>.md`. Each file card links up to
  the concepts that cite it, and each concept lists the exact symbols it
  covers. Notes you write below a concept's generated block are kept.

It costs one call per file per pass, and only for what changed. Summaries are
cached by content hash and per-definition meaning by body hash, and ordinary
builds and query refreshes carry them over. A definition whose code changed
keeps its old summary, marked stale: `bytely check` reports it, and the next
`--deep` redoes only that file. An interrupted run resumes where it stopped.
If the provider stops working (a rejected key, a spent quota, five failures
in a row), the pass stops rather than keep calling. The build then exits 1
with the coverage reached, unless you pass `--allow-partial`.

Providers: `openai` (any OpenAI-compatible endpoint: OpenAI, OpenRouter,
Groq, Together, Fireworks, DeepSeek, a local server), `anthropic` (native),
`litellm`, and `orcarouter`. Set them with `--provider`, `--model`,
`--api-key`, and `--base-url`, or `BYTELY_PROVIDER`, `BYTELY_MODEL`,
`BYTELY_API_KEY`, and `BYTELY_BASE_URL`. Rate limits and server errors are
retried with backoff (`BYTELY_LLM_RETRIES`, default 4). No SDK is needed:
calls go over plain HTTPS. Servers are verified against the system's
certificates plus `certifi`'s when it is installed. That covers macOS
python.org builds without their "Install Certificates" step, and keeps
corporate roots in the system store working.

### Set up your AI coding tools

```bash
bytely init                   # Claude Code + every tool detected here
bytely init --agents claude,cursor --dry-run   # choose, and preview first
bytely init --list-agents     # claude, agents (Codex/opencode), cursor,
                              # gemini, copilot, kiro, windsurf, grok, …
bytely uninstall              # list what init wrote; add --yes to remove it
```

For each tool, `init` writes the usage instructions in that tool's own format
(a skill, a rule file, or a marked section of `AGENTS.md`, `GEMINI.md`, or
`.github/copilot-instructions.md`), registers the MCP server in the tool's
config, adds `/bytely/` to `.gitignore`, and builds the graph if it is
missing. It only ever edits its own marked section or `bytely` entry: your
text and other MCP servers stay as they are, and a config file it cannot
parse is reported and left alone. Running it again changes nothing, and
`uninstall --yes` restores the files. `--no-global` keeps it inside the
repository (Claude Code's user settings, Codex, and Antigravity live in your
home folder); `--no-mcp` skips MCP registration, `--no-hooks` skips hooks and
the status line, and `--no-statusline` (or `BYTELY_NO_STATUSLINE=1`) skips
only the status line.

### Hooks, status line, and savings

`init` also installs hooks, which run `bytely hook <event>` and never fail
the agent's turn:

| Event | What bytely does |
|---|---|
| Session start | Injects how to use bytely plus the start of `bytely/INDEX.md`; if edits left the graph stale, says so and syncs it in the background |
| Prompt | Injects pointers (`file:line`) to the definitions the prompt is about, only when the match is strong (a weak one gets at most two nudges a session) and never the same pointer twice |
| Edit | Marks the file stale and shows who depends on it (its blast radius) |
| Tool use | Counts bytely reads, plain Read/Grep/Glob reads, and tokens saved |
| Stop | Prices the turn from the transcript, notes whether the reply reported its savings, and re-syncs a stale graph in a detached process |

Where they go: Claude Code gets all five in `.claude/settings.json`, and
user-level copies in `~/.claude/settings.json` that step aside in projects
with their own. Codex gets the same in `~/.codex/hooks.json`, and Cursor
gets tool-use accounting in `.cursor/hooks.json`.

Every query (CLI or MCP) opens with its estimated saving:
`[bytely] tokens saved ≈ 9,900 (97%) — this output ≈ 300 tok vs reading the 2
file(s) it covers whole ≈ 10,200 tok (estimate)`. Once a turn has been
priced, it also gives the dollar value at the rate the session is actually
paying. Claude Code's status line (`bytely statusline`) shows the graph size,
whether it is synced, the session's running savings, the context window used,
and the last file edited. An existing status line is never replaced.

### Use from an AI agent (MCP)

`bytely mcp` is a [Model Context Protocol](https://modelcontextprotocol.io)
server over stdio (`bytely init` registers it for you). To register it by
hand with any MCP client, for example in a project's `.mcp.json`:

```json
{
  "mcpServers": {
    "bytely": { "command": "bytely", "args": ["mcp"] }
  }
}
```

It offers six tools, each the same query as the matching command:
`bytely_find_code` (`ask`), `bytely_find_all` (`grep`), `bytely_trace_calls`
(`callers`), `bytely_file_api` (`skeleton`), `bytely_repo_map` (`map`), and
`bytely_check_freshness` (`check`). Every call except the freshness check
refreshes the graph first. The server never indexes a directory by itself:
where no graph exists it offers no tools until `bytely build` has run, and a
graph built with `--include`/`--exclude` stays scoped.

### Example

Given:

```python
# pkg/helpers.py
def greet():
    return "hi"

# pkg/app.py
from .helpers import greet

def main():
    return greet()
```

`bytely build .` prints:

```text
Files: 3; nodes: 5; edges: 4; extraction cache: 0 hits, 3 misses
```

`bytely/pkg/app.md` is the file's card:

```text
# pkg/app.py

- main · function · L3-L4 — def main()
```

and `bytely/.graph/wiring.json` contains nodes and edges such as:

```json
{"id": "pkg/app.py#main", "name": "main", "kind": "function", "path": "pkg/app.py", "span": "L3-L4"}
{"source": "pkg/app.py#main", "target": "pkg/helpers.py#greet", "relation": "calls"}
```

## Project layout

```text
src/bytely/
  cli.py                  Click CLI: every command
  __main__.py             So `python -m bytely` runs the CLI
  engine.py               Library entry point (Bytely class)
  graph/
    build.py              Build pipeline: walk, extract, resolve, write
    extract.py            Tree-sitter extraction (nodes and raw edges)
    generic.py            Top-level constant/variable declarations
    resolve.py            Import, call, and inheritance resolution
    extract_cache.py      Content-hash extraction cache
    enrich.py             Carries meaning over by body hash; runs the crux pass
    refresh.py            Rebuild-if-changed before a query
    outputs.py            Markdown cards and INDEX.md
    check.py              Freshness check (`bytely check`)
    invariants.py         Graph validation (such as unique IDs)
    map.py                `bytely map`
    scopes.py             Monorepo scope discovery
    root.py               Nearest folder holding a graph
    source_files.py       `--include` / `--exclude` selection over a walked tree
    container.py          Recognizes Dockerfiles and Compose files
    types.py, write.py    Graph schema and serialization
  hosts/
    registry.py           Every file init writes, per AI coding tool
    wiring.py             `bytely init` / `bytely uninstall`
    files.py              Safe edits: marked sections, JSON/TOML entries
    hookconfig.py         Hook and status-line entries in hosts' settings
    instructions.py       The agent instructions, in each tool's format
  ai/                     The meaning layer (`build --deep`)
    deep.py               Runs the concept pass, then the graph with meaning
    concepts.py           File summaries → synthesized concept nodes
    crux.py               Per-file symbol summaries and crux spans
    summarize.py, synthesize.py   The other two LLM operations
    failure.py            When a pass stops calling a failing provider
    providers.py          Provider settings (flags, environment, defaults)
    llm/                  Transport: OpenAI-compatible and Anthropic over HTTPS
      http.py             JSON over HTTPS: retries, backoff, connection reuse
      openai.py           OpenAI-compatible transport (`/chat/completions`)
      anthropic.py        Native Anthropic transport (the Messages API)
      types.py            Chat request/response types and the model protocol
      recover.py          Recovers a forced tool call's arguments from reply text
  hooks/
    handlers.py           `bytely hook <event>` and the background sync
    format.py             Orientation, prompt pointers, blast radius, status line
    savings.py            Tokens-saved estimate and pricing
    metrics.py            Per-session reads and savings; `bytely stats`
    transcript.py         Turn cost and savings tally from a transcript
    state.py              Stats, sessions, and the sync lock (bytely/.cache/)
    statusline.py         `bytely statusline`
  mcp/
    server.py             MCP over stdio: JSON-RPC 2.0, one message per line
    tools.py              The six tools and their input schemas
    client.py             A minimal MCP client, for tests and benchmarks
  query/
    service.py            One implementation of each query, for the CLI and MCP
    common.py             Symbol and file lookup, source spans
    skeleton.py           `bytely skeleton`
    callers.py            `bytely callers` (edges in or out, to any depth)
    grep.py               `bytely grep` (grouped by enclosing symbol)
    ask.py                `bytely ask` (lexical ranking + PageRank)
  ingest/fs.py            .gitignore-aware file walker
  util/                   Shared helpers, with no bytely-specific knowledge
    id.py                 Content hashes and text normalization
    lock.py               Cross-process build lock: stale reclaim, heartbeat
    paths.py              Path normalization and prefix matching
    phases.py             Optional per-stage timings, for benchmarking a build
tests/                    Unit, per-language, and reference-parity tests
bench/                    Corpus fetching, build benchmarks, and comparison
```

## Development

```bash
pytest                         # all tests (parity tests: pytest -m parity)
ruff check --fix src tests bench .circleci   # lint; also sorts imports
ruff format src tests bench .circleci        # format (does not sort imports)
mypy src                       # strict
```

CI fails a pull request when `ruff check` or `ruff format --check` would
change anything, so run both before pushing.

The extraction cache invalidates itself when extractor code or a grammar
version changes, so no manual cache bump is needed. If you move extraction
logic into a new module, add it to `EXTRACTOR_MODULES` in
`src/bytely/graph/extract_cache.py`.

`tests/parity/` holds fixture projects, baselines of the expected graph for
each, and a reviewed manifest listing every accepted difference with a reason
code; `tests/test_reference_parity.py` fails on any new difference, and on any
listed one that no longer occurs. The tests only read the committed baselines.

CI (`.circleci/config.yml`) runs lint and type checks, the test suite on Linux
with Python 3.11–3.14 and on Windows with Python 3.13, and integration tests
that build and query pinned Click, Express, and ripgrep checkouts.

## Roadmap

1. Finish the language verification gate: Linux and macOS evidence.
2. Concept nodes in `ask` and MCP results, and a concept-drift report.
3. Performance on large repositories: a faster refresh, a prebuilt search
   index for `ask`, and calls through variables whose type the code never
   states.


## License

Licensed under the [Mozilla Public License 2.0](LICENSE) (`MPL-2.0`).
