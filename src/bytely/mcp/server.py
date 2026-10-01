"""A Model Context Protocol server over stdio.

The transport is newline-delimited JSON-RPC 2.0: one message per line on
stdin, one response per line on stdout. Stdout carries protocol messages
only; anything diagnostic goes to stderr. The server answers `initialize`,
`ping`, `tools/list`, and `tools/call`, ignores notifications, and replies
to anything else with a JSON-RPC error. A tool that cannot answer returns
its message as a result with `isError`, so the agent can see it and retry.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any

from bytely.graph.refresh import NoGraphError
from bytely.graph.write import graph_path
from bytely.mcp.tools import TOOLS, TOOLS_BY_NAME, ToolError

if TYPE_CHECKING:
    from bytely.mcp.tools import Workspace

# Newest first; the first is offered when the client asks for another.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
# JSON-RPC batches were removed from MCP in this version.
NO_BATCHES_FROM = "2025-06-18"
NO_GRAPH = "No graph found here; run `bytely build` in the repository first."

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

INSTRUCTIONS = (
    "This repository is indexed by bytely: a graph of every definition, "
    "its exact file:line span, and who calls what. Prefer these tools to "
    "grepping and reading files; one call usually replaces several reads. "
    'bytely_find_code answers "how does X work" / "where is Y" with '
    "ranked definitions (add source=true to see the code); "
    "bytely_find_all finds every occurrence of a pattern; "
    'bytely_trace_calls shows callers or callees, and with depth="all" '
    "the blast radius of a change; bytely_file_api lists a file's API; "
    "bytely_repo_map orients you in an unfamiliar repository. Results "
    "reflect uncommitted edits: the graph refreshes before each query."
)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("bytely")
    except Exception:  # noqa: BLE001 - metadata is optional (source checkout)
        return "0"


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


@dataclass
class Session:
    """One client connection: the repository, and the negotiated version."""

    workspace: Workspace
    protocol_version: str | None = None

    def has_graph(self) -> bool:
        """Whether there is a graph to answer from (or one may be built)."""
        if self.workspace.create:
            return True
        return graph_path(
            self.workspace.context_dir or str(self.workspace.root / "bytely")
        ).is_file()


def handle_message(session: Session, message: Any) -> dict[str, Any] | None:
    """The response to one decoded message, or None when none is due.

    Notifications and responses (a message with no `method`) get no reply,
    as JSON-RPC requires.
    """
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, INVALID_REQUEST, "Invalid JSON-RPC 2.0 request")
    method = message.get("method")
    if "method" not in message:
        return None  # a response (or a stray reply); never answered
    request_id = message.get("id")
    if "id" not in message:
        return None  # `notifications/initialized`, cancellations, …
    if not isinstance(method, str):
        return _error(request_id, INVALID_REQUEST, "method must be a string")
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return _error(request_id, INVALID_PARAMS, "params must be an object")

    if method == "initialize":
        requested = params.get("protocolVersion")
        version = (
            requested
            if requested in PROTOCOL_VERSIONS
            else PROTOCOL_VERSIONS[0]
        )
        session.protocol_version = version
        return _result(
            request_id,
            {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "bytely", "version": _version()},
                "instructions": INSTRUCTIONS,
            },
        )
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        # A directory that never had a graph gets no tools: each would only
        # answer "run bytely build", at the cost of their schemas in every
        # prompt.
        tools = (
            [tool.listing() for tool in TOOLS] if session.has_graph() else []
        )
        return _result(request_id, {"tools": tools})
    if method == "tools/call":
        return _call_tool(session, request_id, params)
    return _error(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")


def _call_tool(
    session: Session, request_id: Any, params: dict[str, Any]
) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    tool = TOOLS_BY_NAME.get(name) if isinstance(name, str) else None
    if tool is None:
        return _error(request_id, INVALID_PARAMS, f"Unknown tool: {name}")
    if not isinstance(arguments, dict):
        return _error(request_id, INVALID_PARAMS, "arguments must be an object")
    if not session.has_graph():
        return _tool_result(request_id, NO_GRAPH, is_error=True)
    try:
        text = tool.call(session.workspace, arguments)
    except ToolError as error:
        return _tool_result(request_id, str(error), is_error=True)
    except NoGraphError:
        return _tool_result(request_id, NO_GRAPH, is_error=True)
    except Exception as error:  # noqa: BLE001 - reported to the agent
        print(  # noqa: T201 - stderr; stdout carries the protocol
            f"bytely: {tool.name} failed: {error!r}", file=sys.stderr
        )
        return _tool_result(
            request_id, f"{tool.name} failed: {error}", is_error=True
        )
    return _tool_result(request_id, text)


def _tool_result(
    request_id: Any, text: str, *, is_error: bool = False
) -> dict[str, Any]:
    return _result(
        request_id,
        {"content": [{"type": "text", "text": text}], "isError": is_error},
    )


def handle_line(session: Session, line: bytes) -> Any:
    """The reply to one line of input: a message, a list, or None."""
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return _error(None, PARSE_ERROR, f"Parse error: {error}")
    if not isinstance(message, list):
        return handle_message(session, message)
    if not message:
        return _error(None, INVALID_REQUEST, "Empty batch")
    version = session.protocol_version
    if version is None or version >= NO_BATCHES_FROM:
        return _error(
            None,
            INVALID_REQUEST,
            "JSON-RPC batches are not supported in protocol "
            f"{version or NO_BATCHES_FROM}",
        )
    replies = [
        reply
        for reply in (handle_message(session, item) for item in message)
        if reply is not None
    ]
    return replies or None


def serve(
    workspace: Workspace,
    stdin: IO[bytes] | None = None,
    stdout: IO[bytes] | None = None,
) -> None:
    """Answer messages from stdin until it closes."""
    source = stdin if stdin is not None else sys.stdin.buffer
    sink = stdout if stdout is not None else sys.stdout.buffer
    session = Session(workspace)
    for raw in source:
        line = raw.strip()
        if not line:
            continue
        reply = handle_line(session, line)
        if reply is not None:
            _write(sink, reply)


def _write(sink: IO[bytes], payload: Any) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    sink.write(data.encode("utf-8") + b"\n")
    sink.flush()
