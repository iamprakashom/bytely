"""The MCP server: protocol handling and the six tools."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from bytely.graph.build import build_graph
from bytely.mcp.server import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    NO_GRAPH,
    PARSE_ERROR,
    PROTOCOL_VERSIONS,
    Session,
    handle_line,
    handle_message,
    serve,
)
from bytely.mcp.tools import TOOLS, Workspace

if TYPE_CHECKING:
    from pathlib import Path

STORE = (
    "class Store:\n"
    "    def save(self, key: str) -> None:\n"
    "        write_file(key)\n"
    "\n"
    "\n"
    "def write_file(path: str) -> None:\n"
    "    pass\n"
)


def _project(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "store.py").write_text(STORE, encoding="utf-8")


@pytest.fixture
def session(tmp_path: Path) -> Session:
    """A server session on a project that has been built once."""
    _project(tmp_path)
    build_graph(str(tmp_path))
    return Session(Workspace(tmp_path, create=False))


def _request(method: str, params: Any = None, id_: int = 1) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        message["params"] = params
    return message


def _call(session: Session, tool: str, **arguments: Any) -> dict[str, Any]:
    response = handle_message(
        session,
        _request("tools/call", {"name": tool, "arguments": arguments}),
    )
    assert response is not None
    return response["result"]  # type: ignore[no-any-return]


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


def test_initialize_negotiates_the_protocol_version(session: Session) -> None:
    for requested, expected in (
        ("2025-03-26", "2025-03-26"),
        ("1999-01-01", PROTOCOL_VERSIONS[0]),
    ):
        response = handle_message(
            session, _request("initialize", {"protocolVersion": requested})
        )
        assert response is not None
        result = response["result"]
        assert result["protocolVersion"] == expected
        assert session.protocol_version == expected
        assert result["capabilities"] == {"tools": {"listChanged": False}}
        assert result["serverInfo"]["name"] == "bytely"
        assert "bytely_find_code" in result["instructions"]


def test_notifications_and_responses_get_no_reply(session: Session) -> None:
    for message in (
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}},
        # A response (to a request we never sent) must not be answered.
        {"jsonrpc": "2.0", "id": 5, "result": {}},
        {"jsonrpc": "2.0", "id": 6, "error": {"code": 1, "message": "x"}},
    ):
        assert handle_message(session, message) is None, message


def test_errors_use_standard_codes(session: Session) -> None:
    def code(message: Any) -> int:
        response = handle_message(session, message)
        assert response is not None
        return int(response["error"]["code"])

    assert handle_message(session, _request("ping")) == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {},
    }
    assert code(_request("resources/list")) == METHOD_NOT_FOUND
    assert code({"id": 1, "method": "ping"}) == INVALID_REQUEST
    assert code(["not", "an", "object"]) == INVALID_REQUEST
    assert code(_request("tools/list", params=[1])) == INVALID_PARAMS
    assert (
        code(_request("tools/call", {"name": "nope", "arguments": {}}))
        == INVALID_PARAMS
    )


def test_batches_follow_the_negotiated_protocol(session: Session) -> None:
    ping_batch = json.dumps(
        [_request("ping", id_=7), _request("ping", id_=8)]
    ).encode()

    empty = handle_line(session, b"[]")
    assert empty["error"]["code"] == INVALID_REQUEST

    handle_message(
        session, _request("initialize", {"protocolVersion": "2025-06-18"})
    )
    refused = handle_line(session, ping_batch)
    assert refused["error"]["code"] == INVALID_REQUEST

    handle_message(
        session, _request("initialize", {"protocolVersion": "2025-03-26"})
    )
    answered = handle_line(session, ping_batch)
    assert [reply["id"] for reply in answered] == [7, 8]


def test_tools_list_describes_every_tool(session: Session) -> None:
    response = handle_message(session, _request("tools/list"))
    assert response is not None
    listed = response["result"]["tools"]
    assert [tool["name"] for tool in listed] == [tool.name for tool in TOOLS]
    for tool in listed:
        assert tool["description"]
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert set(schema.get("required", [])) <= set(schema["properties"])


def test_a_directory_without_a_graph_gets_no_tools_and_is_not_indexed(
    tmp_path: Path,
) -> None:
    _project(tmp_path)
    fresh = Session(Workspace(tmp_path, create=False))

    listed = handle_message(fresh, _request("tools/list"))
    assert listed is not None
    assert listed["result"]["tools"] == []
    called = _call(fresh, "bytely_find_code", query="save")
    assert called["isError"]
    assert _text(called) == NO_GRAPH
    assert not (tmp_path / "bytely").exists()


def test_each_tool_answers_from_the_graph(session: Session) -> None:
    found = _call(session, "bytely_find_code", query="save store", limit=1)
    assert not found["isError"]
    text = _text(found)
    assert "save · method · pkg/store.py:L2-L3" in text
    # Code is included by default, as the reference implementation does.
    assert "def save(self, key: str) -> None:" in text
    # The follow-up advice names this surface's tools.
    assert 'bytely_file_api file="pkg/store.py"' in text
    assert "bytely skeleton" not in text

    traced = _call(session, "bytely_trace_calls", symbol="write_file")
    assert "calls ← save · method" in _text(traced)
    out = _call(
        session, "bytely_trace_calls", symbol="Store.save", direction="out"
    )
    assert "calls → write_file · function" in _text(out)

    every = _call(
        session, "bytely_find_all", pattern="WRITE_FILE", ignore_case=True
    )
    assert "2 hits in 2 symbols" in _text(every)

    api = _call(session, "bytely_file_api", file="store.py")
    assert "- L6-L7  function write_file" in _text(api)

    repo = _call(session, "bytely_repo_map", max_dirs=2)
    assert _text(repo).startswith("repo map — 1 file")

    fresh = _call(session, "bytely_check_freshness", no_cache=True)
    assert _text(fresh) == "fresh: Graph is up to date.\n"


def test_trace_calls_takes_files_scopes_and_full_depth(
    session: Session,
) -> None:
    (session.workspace.root / "pkg" / "app.py").write_text(
        "from pkg.store import write_file\n"
        "\n"
        "\n"
        "def run() -> None:\n"
        "    write_file('x')\n",
        encoding="utf-8",
    )
    # A file stands for what it defines: its outside dependents are listed
    # with the definition they reach; calls inside the file are not.
    by_file = _call(session, "bytely_trace_calls", file="pkg/store.py")
    text = _text(by_file)
    assert "calls ← run · function · pkg/app.py:L4-L5  (via write_file)" in (
        text
    )
    assert "save · method" not in text.split("\n", 2)[-1]

    full = _call(
        session, "bytely_trace_calls", symbol="write_file", depth="full"
    )
    assert not full["isError"]

    elsewhere = _call(
        session, "bytely_trace_calls", symbol="write_file", **{"in": "other"}
    )
    assert elsewhere["isError"]
    assert "No symbol matches 'write_file' under other" in _text(elsewhere)


def test_unknown_arguments_are_ignored_and_bad_ones_explained(
    session: Session,
) -> None:
    extra = _call(session, "bytely_repo_map", verbose=True)
    assert not extra["isError"]

    cases = [
        ("bytely_find_code", {}, "query"),
        ("bytely_find_code", {"query": "x", "limit": 0}, "limit"),
        ("bytely_trace_calls", {"symbol": "save", "depth": "deep"}, "depth"),
        ("bytely_trace_calls", {"symbol": "nothing"}, "No symbol matches"),
        ("bytely_find_all", {"pattern": "("}, "Invalid pattern"),
        ("bytely_file_api", {"file": "nope.py"}, "No indexed file matches"),
    ]
    for tool, arguments, expected in cases:
        result = _call(session, tool, **arguments)
        assert result["isError"], (tool, arguments)
        assert expected in _text(result), (tool, arguments)


def test_a_rebuild_is_announced_and_an_unchanged_graph_is_reused(
    session: Session,
) -> None:
    first = _call(session, "bytely_file_api", file="store.py")
    assert not _text(first).startswith("(graph refreshed")
    graph = session.workspace.last.graph if session.workspace.last else None

    again = _call(session, "bytely_file_api", file="store.py")
    assert not _text(again).startswith("(graph refreshed")
    assert session.workspace.last is not None
    assert session.workspace.last.graph is graph  # reused, not re-read

    (session.workspace.root / "pkg" / "extra.py").write_text(
        "def added():\n    pass\n", encoding="utf-8"
    )
    changed = _call(session, "bytely_file_api", file="extra.py")
    assert _text(changed).startswith("(graph refreshed: 1 file re-parsed)")


def test_a_scoped_graph_stays_scoped(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text(
        "def write_file():\n    pass\n", encoding="utf-8"
    )
    build_graph(str(tmp_path), exclude_patterns=["vendor/**"])
    scoped = Session(Workspace(tmp_path, create=False))

    hits = _call(scoped, "bytely_find_all", pattern="write_file")
    assert "vendor/" not in _text(hits)
    fresh = _call(scoped, "bytely_check_freshness")
    assert _text(fresh).startswith("fresh")


def test_serve_reads_lines_and_writes_one_reply_per_request(
    session: Session,
) -> None:
    lines = [
        json.dumps(_request("initialize", {"protocolVersion": "2025-03-26"})),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "",
        "{not json",
        json.dumps([_request("ping", id_=7), _request("ping", id_=8)]),
        json.dumps(_request("tools/list", id_=2)),
    ]
    stdout = io.BytesIO()
    serve(
        session.workspace,
        stdin=io.BytesIO(("\n".join(lines) + "\n").encode("utf-8")),
        stdout=stdout,
    )
    replies = [json.loads(line) for line in stdout.getvalue().splitlines()]

    assert replies[0]["id"] == 1
    assert "protocolVersion" in replies[0]["result"]
    assert replies[1]["error"]["code"] == PARSE_ERROR
    assert [reply["id"] for reply in replies[2]] == [7, 8]
    assert replies[3]["id"] == 2
    assert "tools" in replies[3]["result"]
    assert len(replies) == 4


def test_mcp_command_speaks_json_rpc_on_stdout(session: Session) -> None:
    messages = (
        _request("initialize", {"protocolVersion": "2025-06-18"}),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        _request(
            "tools/call",
            {"name": "bytely_file_api", "arguments": {"file": "store.py"}},
            id_=2,
        ),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "bytely",
            "mcp",
            str(session.workspace.root),
        ],
        input=("\n".join(json.dumps(m) for m in messages) + "\n").encode(),
        capture_output=True,
        timeout=120,
        check=True,
    )
    replies = [
        json.loads(line)
        for line in completed.stdout.decode("utf-8").splitlines()
    ]
    assert [reply["id"] for reply in replies] == [1, 2]
    assert _text(replies[1]["result"]).startswith("skeleton — pkg/store.py")


def test_mcp_command_rejects_a_missing_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "bytely",
            "mcp",
            str(tmp_path / "missing"),
        ],
        input=b"",
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode != 0
    assert b"does not exist" in completed.stderr
