"""A minimal MCP client for a server on stdio, for tests and benchmarks.

It speaks the same framing as `bytely.mcp.server` (one JSON-RPC message per
line), so it can drive `bytely mcp` or any other stdio MCP server.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

PROTOCOL_VERSION = "2024-11-05"


class McpError(RuntimeError):
    """The server answered with an error, or a tool reported one."""


class McpSession:
    """A long-lived MCP server process, used through its tools."""

    def __init__(
        self,
        command: list[str],
        cwd: str | Path,
        env: dict[str, str] | None = None,
    ) -> None:
        """Start the server and complete the MCP handshake."""
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.next_id = 0
        self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "bytely-client", "version": "0"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        listed = self.request("tools/list", {}).get("tools", [])
        self.tools = [tool["name"] for tool in listed]

    def __enter__(self) -> McpSession:
        """Use the session as a context manager that closes it."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the server."""
        self.close()

    def _send(self, message: dict[str, Any]) -> None:
        stdin: IO[bytes] = self.process.stdin  # type: ignore[assignment]
        stdin.write(json.dumps(message).encode("utf-8") + b"\n")
        stdin.flush()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return its result, skipping notifications."""
        self.next_id += 1
        self._send(
            {
                "jsonrpc": "2.0",
                "id": self.next_id,
                "method": method,
                "params": params,
            }
        )
        stdout: IO[bytes] = self.process.stdout  # type: ignore[assignment]
        while True:
            line = stdout.readline()
            if not line:
                raise McpError(f"MCP server closed during {method}")
            message = json.loads(line)
            if message.get("id") != self.next_id:
                continue
            if "error" in message:
                raise McpError(f"{method}: {message['error']}")
            result: dict[str, Any] = message.get("result", {})
            return result

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call a tool and return its text; a tool error raises McpError."""
        result = self.request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        text = "".join(
            part.get("text", "")
            for part in result.get("content", [])
            if isinstance(part, dict)
        )
        if result.get("isError"):
            raise McpError(text)
        return text

    def tool_for(self, suffix: str) -> str | None:
        """The server's tool whose name ends with `suffix`."""
        return next((t for t in self.tools if t.endswith(suffix)), None)

    def close(self) -> None:
        """Stop the server."""
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
