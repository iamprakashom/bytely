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


## License

Licensed under the [Mozilla Public License 2.0](LICENSE) (`MPL-2.0`).
