"""Agent-host integration: hooks, status line, and token-savings accounting."""

# The `bytely hook` events, here rather than in `handlers` so the CLI can
# list them without importing the handlers.
EVENTS = (
    "session-start",
    "prompt",
    "post-edit",
    "tool-savings",
    "stop",
    "cursor-post-tool",
    "cursor-mcp",
    "cursor-session-end",
    "sync",
)
