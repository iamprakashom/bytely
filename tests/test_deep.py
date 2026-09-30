"""`bytely build --deep`: the LLM meaning layer, with a scripted model."""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from click.testing import CliRunner

from bytely.ai.deep import run_deep
from bytely.ai.failure import FailureGate, terminal_reason
from bytely.ai.llm.anthropic import AnthropicChatModel
from bytely.ai.llm.http import post_json
from bytely.ai.llm.openai import OpenAIChatModel, _adapt
from bytely.ai.llm.recover import recover_tool_args
from bytely.ai.llm.types import (
    ChatRequest,
    ChatResponse,
    LLMError,
    Message,
    ToolCall,
)
from bytely.ai.providers import needs_key, resolve_config
from bytely.cli import main
from bytely.graph.build import build_graph
from bytely.graph.check import check_graph
from bytely.graph.write import read_graph
from bytely.query.service import Workspace, ask_text

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

PRICING = '''\
def price_with_tax(amount, rate):
    """Apply the regional tax rate to a net amount."""
    if rate < 0:
        raise ValueError("negative rate")
    taxed = amount * (1 + rate)
    return round(taxed, 2)


def describe():
    return "pricing"
'''
CART = """\
from pricing import price_with_tax


def cart_total(items, rate):
    return price_with_tax(sum(items), rate)
"""


class ScriptedModel:
    """Answers each LLM operation the way a well-behaved provider would."""

    label = "scripted"

    def __init__(self, fail_with: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_with = fail_with
        self.lock = threading.Lock()

    def create(self, request: ChatRequest) -> ChatResponse:
        user = request.messages[-1].content
        with self.lock:
            self.calls.append(request.force_tool or "summary")
        if self.fail_with:
            raise LLMError(self.fail_with, 401)
        if request.force_tool == "record_symbols":
            targets = re.findall(
                r"- id=(\S+) \| \S+ \| lines L(\d+)-L(\d+)", user
            )
            symbols = [
                {
                    "id": node_id,
                    "summary": f"Explains {node_id.split('#')[-1]}.",
                    "crux_start": int(start) + 1 if end != start else 0,
                    "crux_end": int(start) + 2 if end != start else 0,
                }
                for node_id, start, end in targets
            ]
            return ChatResponse(
                "",
                [ToolCall("record_symbols", {"symbols": symbols})],
                "tool_use",
            )
        if request.force_tool == "record_graph":
            paths = re.findall(r"^## (\S+)$", user, re.MULTILINE)
            nodes = [
                {
                    "name": "Checkout pricing",
                    "type": "system",
                    "summary": "Totals carts and applies tax.",
                    "sources": paths,
                    "links": [{"to": "Tax rules", "relation": "uses"}],
                },
                {
                    "name": "Tax rules",
                    "type": "concept",
                    "summary": "Rates are never negative.",
                    "sources": [],
                    "links": [{"to": "Nowhere", "relation": "uses"}],
                },
            ]
            # Delivered as text: a gateway that ignored the forced tool.
            return ChatResponse(
                "```json\n" + json.dumps({"nodes": nodes}) + "\n```", [], "stop"
            )
        path = user.splitlines()[0].removeprefix("File: ")
        return ChatResponse(f"{path} handles pricing.", [], "stop")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "shop"
    repo.mkdir()
    (repo / "pricing.py").write_text(PRICING, encoding="utf-8")
    (repo / "cart.py").write_text(CART, encoding="utf-8")
    return repo


def _deep(repo: Path, model: ScriptedModel) -> Any:
    return run_deep(
        str(repo), None, resolve_config("openai", api_key="k"), model=model
    )


def test_deep_build_writes_summaries_crux_and_concepts(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    model = ScriptedModel()
    result = _deep(repo, model)

    assert not result.degraded
    meaning = result.graph.meaning
    assert meaning.pending == 0 and meaning.computed == len(
        read_graph(str(repo / "bytely")).nodes
    )
    graph = read_graph(str(repo / "bytely"))
    price = next(n for n in graph.nodes if n.name == "price_with_tax")
    assert price.summary == "Explains price_with_tax."
    assert price.summary_state == "ready"
    assert price.crux is not None
    assert price.crux.span == "L2-L3"
    assert price.crux.code.startswith('    """Apply the regional')

    concept = (repo / "bytely" / "concepts" / "checkout-pricing.md").read_text(
        "utf-8"
    )
    assert concept.startswith('---\nname: "Checkout pricing"\n')
    assert '- symbol: "price_with_tax"' in concept
    assert "- uses [[tax-rules]]" in concept
    assert "Nowhere" not in concept  # links only to defined nodes
    tax = (repo / "bytely" / "concepts" / "tax-rules.md").read_text("utf-8")
    assert '"pricing.py"' in tax  # inherits the sources of what it links
    card = (repo / "bytely" / "pricing.md").read_text("utf-8")
    assert card.startswith("# pricing.py · [[checkout-pricing]] [[tax-rules]]")
    assert (
        "price_with_tax · function · L1-L6 — Explains price_with_tax." in card
    )
    index = (repo / "bytely" / "INDEX.md").read_text("utf-8")
    assert "## Concepts" in index
    assert "[checkout-pricing](concepts/checkout-pricing.md)" in index

    fresh, message = check_graph(str(repo))
    assert fresh, message

    out = ask_text(Workspace(repo), "price with tax", source=True, limit=1)
    assert "Explains price_with_tax." in out
    assert "crux of L1-L6" in out


def test_meaning_is_carried_over_and_only_changes_are_resummarized(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _deep(repo, ScriptedModel())
    notes = repo / "bytely" / "concepts" / "tax-rules.md"
    notes.write_text(
        notes.read_text("utf-8") + "Team note: rates come from finance.\n",
        encoding="utf-8",
    )

    (repo / "cart.py").write_text(
        CART.replace("sum(items)", "sum(items) + 0"), encoding="utf-8"
    )
    build_graph(str(repo))  # a plain build keeps the meaning layer
    graph = read_graph(str(repo / "bytely"))
    states = {n.name: n.summary_state for n in graph.nodes}
    assert states["price_with_tax"] == "ready"
    assert states["cart_total"] == "stale"
    cart = next(n for n in graph.nodes if n.name == "cart_total")
    assert cart.summary == "Explains cart_total."  # kept as a hint
    fresh, message = check_graph(str(repo))
    assert not fresh and "cart.py#cart_total" in message

    model = ScriptedModel()
    result = _deep(repo, model)
    # Only cart.py's symbols, its file summary, and its synthesis batch.
    assert model.calls.count("record_symbols") == 1
    assert model.calls.count("summary") == 1
    assert result.graph.meaning.computed == 2  # the file node and cart_total
    assert check_graph(str(repo))[0]
    assert "Team note: rates come from finance." in notes.read_text("utf-8")


def test_a_rejected_key_stops_the_pass_at_once(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    model = ScriptedModel(fail_with="401 Unauthorized: invalid api key")
    result = _deep(repo, model)

    assert result.degraded
    assert "rejected the API key" in (result.concepts.fatal or "")
    assert "rejected the API key" in (result.graph.meaning.fatal or "")
    # Stopped after the first failure in each pass, not one call per file.
    assert len(model.calls) <= 2 + 5
    graph = read_graph(str(repo / "bytely"))
    assert graph is not None  # the structural graph is still written
    assert all(n.summary_state == "pending" for n in graph.nodes)


def test_failure_gate_rules() -> None:
    gate = FailureGate()
    for _ in range(4):
        gate.record("500 server error")
    gate.succeeded()
    for _ in range(4):
        gate.record("500 server error")
    assert not gate.stopped
    gate.record("model returned nothing", quality=True)
    assert not gate.stopped
    gate.record("500 server error")
    assert gate.stopped and "5 files in a row" in (gate.fatal or "")
    assert terminal_reason("402 Payment Required") is not None
    assert terminal_reason("insufficient_quota") is not None
    assert terminal_reason("timeout") is None


def test_cli_deep_needs_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BYTELY_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "BYTELY_BASE_URL",
        "BYTELY_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    repo = _repo(tmp_path)
    result = CliRunner().invoke(main, ["build", str(repo), "--deep"])
    assert result.exit_code != 0
    assert "needs an API key" in result.output
    bad = CliRunner().invoke(
        main, ["--provider", "nope", "build", str(repo), "--deep"]
    )
    assert "Unknown provider" in bad.output


def test_provider_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "BYTELY_API_KEY",
        "BYTELY_MODEL",
        "BYTELY_BASE_URL",
        "BYTELY_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    config = resolve_config()
    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.model == "openai/gpt-4o-mini"
    assert config.headers == {"x-title": "bytely"}
    monkeypatch.delenv("OPENROUTER_API_KEY")
    plain = resolve_config(api_key="k")
    assert plain.base_url is None and plain.model == "gpt-4o-mini"
    local = resolve_config("litellm")
    assert local.base_url == "http://localhost:4000" and not needs_key(local)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    claude = resolve_config("anthropic")
    assert (
        claude.api_key == "a-key" and claude.key_source == "ANTHROPIC_API_KEY"
    )


def test_recover_tool_args_variants() -> None:
    payload = {"symbols": [{"id": "a"}]}
    for text in (
        json.dumps(payload),
        "```json\n" + json.dumps(payload) + "\n```",
        json.dumps([{"name": "record_symbols", "parameters": payload}]),
        json.dumps(
            {"name": "record_symbols", "arguments": json.dumps(payload)}
        ),
        "Here you go: " + json.dumps(payload) + " done",
    ):
        assert recover_tool_args(text, ["record_symbols"], "symbols") == payload
    wrong = json.dumps({"name": "other_tool", "parameters": payload})
    assert recover_tool_args(wrong, ["record_symbols"], "symbols") is None
    assert recover_tool_args("no json here", ["x"], "symbols") is None


def test_openai_payload_and_known_refusals() -> None:
    model = OpenAIChatModel("gpt-x", "key", "https://gw.example/v1/")
    assert model.url == "https://gw.example/v1/chat/completions"
    request = ChatRequest(
        [Message("system", "s"), Message("user", "u")],
        tools=[],
        force_tool="record_graph",
    )
    params = model.payload(request)
    assert params["tool_choice"]["function"]["name"] == "record_graph"
    assert params["temperature"] == 0.0

    def refusal(text: str) -> LLMError:
        return LLMError(f"400 Bad Request: {text}", 400)

    with_tool = {**params, "tools": [{"type": "function"}]}
    fixed = _adapt(with_tool, refusal("Invalid tool_choice type: 'object'"))
    assert fixed is not None and fixed["tool_choice"] == "required"
    fixed = _adapt(
        params,
        refusal("max_tokens is not supported; use max_completion_tokens"),
    )
    assert fixed is not None and "max_completion_tokens" in fixed
    fixed = _adapt(params, refusal("temperature does not support 0.0"))
    assert fixed is not None and "temperature" not in fixed
    fixed = _adapt(
        params,
        refusal("Function tools with reasoning_effort are not supported"),
    )
    assert fixed is not None and fixed["reasoning_effort"] == "none"
    assert _adapt(params, LLMError("400 other", 400)) is None


def test_anthropic_payload() -> None:
    model = AnthropicChatModel("claude-x", "key")
    assert model.url == "https://api.anthropic.com/v1/messages"
    params = model.payload(
        ChatRequest(
            [Message("system", "rules"), Message("user", "hi")],
            force_tool="record_symbols",
        )
    )
    assert params["system"] == "rules"
    assert params["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in params
    assert params["tool_choice"] == {"type": "tool", "name": "record_symbols"}


def test_http_retries_rate_limits_but_not_bad_keys() -> None:
    replies = [(429, {"error": {"message": "slow down"}}), (200, {"ok": 1})]
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.append(self.headers.get("authorization", ""))
            self.rfile.read(int(self.headers["content-length"]))
            status, body = replies.pop(0)
            self.send_response(status)
            if status == 429:
                self.send_header("retry-after", "0")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/x"
    try:
        assert post_json(url, {}, {"authorization": "Bearer k"}) == {"ok": 1}
        assert seen == ["Bearer k", "Bearer k"]
        replies.append((401, {"error": {"message": "invalid api key"}}))
        try:
            post_json(url, {}, {})
        except LLMError as error:
            assert error.status == 401 and "invalid api key" in str(error)
        else:
            raise AssertionError("expected a 401")
        assert not replies  # the 401 was not retried
    finally:
        server.shutdown()


def test_cli_deep_end_to_end_over_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path: CLI, config, HTTP, OpenAI wire format, both passes."""
    scripted = ScriptedModel()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(
                self.rfile.read(int(self.headers["content-length"]))
            )
            assert self.path == "/v1/chat/completions"
            forced = (body.get("tool_choice") or {}).get("function", {})
            reply = scripted.create(
                ChatRequest(
                    [
                        Message(m["role"], m["content"])
                        for m in body["messages"]
                    ],
                    force_tool=forced.get("name"),
                )
            )
            message: dict[str, Any] = {
                "role": "assistant",
                "content": reply.text,
            }
            if reply.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.args),
                        },
                    }
                    for call in reply.tool_calls
                ]
            data = json.dumps(
                {
                    "choices": [{"message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(
        "BYTELY_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
    )
    monkeypatch.setenv("BYTELY_API_KEY", "test-key")
    monkeypatch.setenv("BYTELY_MODEL", "test-model")
    repo = _repo(tmp_path)
    try:
        result = CliRunner().invoke(
            main, ["build", str(repo), "--deep", "-j", "2"]
        )
    finally:
        server.shutdown()
    assert result.exit_code == 0, result.output
    assert "Concepts: 2 nodes" in result.output
    assert "pending" in result.output and "0 pending" in result.output
    assert (repo / "bytely" / "concepts" / "checkout-pricing.md").is_file()


class FailingSynthesis(ScriptedModel):
    """Summaries work; synthesis is refused (a spent quota)."""

    def create(self, request: ChatRequest) -> ChatResponse:
        if request.force_tool == "record_graph":
            raise LLMError("429 insufficient_quota", 429)
        return super().create(request)


class PartialCrux(ScriptedModel):
    """Symbol replies omit every definition named `describe`."""

    def create(self, request: ChatRequest) -> ChatResponse:
        reply = super().create(request)
        if request.force_tool != "record_symbols":
            return reply
        kept = [
            s
            for s in reply.tool_calls[0].args["symbols"]
            if not s["id"].endswith("#describe")
        ]
        return ChatResponse(
            "", [ToolCall("record_symbols", {"symbols": kept})], "tool_use"
        )


def test_failed_concept_run_keeps_previous_concepts_and_notes(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _deep(repo, ScriptedModel())
    concepts = repo / "bytely" / "cache" / "concepts.json"
    before = concepts.read_text("utf-8")
    (repo / "cart.py").write_text(CART + "\n# changed\n", encoding="utf-8")

    result = _deep(repo, FailingSynthesis())

    assert result.degraded and result.concepts.kept_previous
    assert concepts.read_text("utf-8") == before
    assert (repo / "bytely" / "concepts" / "checkout-pricing.md").is_file()


def test_removed_concept_page_is_kept_only_for_user_notes(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _deep(repo, ScriptedModel())
    pages = repo / "bytely" / "concepts"
    noted = pages / "tax-rules.md"
    noted.write_text(noted.read_text("utf-8") + "MY NOTE\n", encoding="utf-8")
    concepts = repo / "bytely" / "cache" / "concepts.json"
    data = json.loads(concepts.read_text("utf-8"))
    data["nodes"] = []  # both concepts disappear
    concepts.write_text(json.dumps(data), encoding="utf-8")

    build_graph(str(repo))

    assert not (pages / "checkout-pricing.md").exists()  # default notes
    kept = noted.read_text("utf-8")
    assert "orphaned: true" in kept and kept.endswith("MY NOTE\n")
    build_graph(str(repo))  # stays kept, and no ownership conflict
    assert noted.read_text("utf-8") == kept


def test_truncated_graph_is_rebuilt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    build_graph(str(repo))
    graph_file = repo / "bytely" / ".graph" / "wiring.json"
    graph_file.write_bytes(graph_file.read_bytes()[:100])
    build_graph(str(repo))
    assert read_graph(str(repo / "bytely")) is not None


def test_crux_moves_with_its_definition(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _deep(repo, ScriptedModel())
    (repo / "pricing.py").write_text(
        "# a\n# b\n# c\n" + PRICING, encoding="utf-8"
    )
    build_graph(str(repo))
    node = next(
        n
        for n in read_graph(str(repo / "bytely")).nodes
        if n.name == "price_with_tax"
    )
    assert node.summary_state == "ready" and node.span == "L4-L9"
    assert node.crux is not None and node.crux.span == "L5-L6"
    out = ask_text(Workspace(repo), "price with tax", source=True, limit=1)
    assert '     5      """Apply the regional' in out


def test_partial_symbol_replies_count_as_failures(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    result = _deep(repo, PartialCrux())
    meaning = result.graph.meaning
    assert result.degraded
    assert meaning.failed_files == 1 and meaning.pending == 1
    assert any("left 1 of" in error for error in meaning.errors)


def test_http_connections_are_reused_and_reopened_when_closed() -> None:
    from bytely.ai.llm.http import close_connections

    peers: set[tuple[str, int]] = set()
    count = {"n": 0}

    class KeepAlive(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            peers.add(self.client_address)
            self.rfile.read(int(self.headers["content-length"]))
            count["n"] += 1
            body = json.dumps({"n": count["n"]}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            # After the fourth reply the server drops the connection without
            # saying so, as idle servers do; the client learns on next use.
            if count["n"] == 4:
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), KeepAlive)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1/x"
    try:
        close_connections()
        for expected in range(1, 7):
            assert post_json(url, {}, {}) == {"n": expected}
    finally:
        close_connections()
        server.shutdown()
    # Six calls over two connections: one until the server closed it,
    # then one more, instead of a new handshake per call.
    assert len(peers) == 2


def test_retry_rules_match_the_sdks() -> None:
    from email.utils import formatdate

    from bytely.ai.llm.http import default_backoff, retry_after, should_retry

    for status in (408, 409, 429, 500, 503, 529):
        assert should_retry(status, {})
    for status in (400, 401, 403, 404, 422):
        assert not should_retry(status, {})
    # The server's explicit verdict wins over the status code.
    assert should_retry(400, {"x-should-retry": "true"})
    assert not should_retry(503, {"x-should-retry": "false"})

    assert retry_after({"retry-after-ms": "1500", "retry-after": "9"}) == 1.5
    assert retry_after({"retry-after": "2"}) == 2.0
    now = 1_000_000_000.0
    date = formatdate(now + 30, usegmt=True)
    assert retry_after({"retry-after": date}, now=now) == 30.0
    assert retry_after({"retry-after": "3600"}) == 60.0  # capped
    assert retry_after({"retry-after": "soon"}) is None
    assert retry_after({}) is None

    for attempt, ceiling in ((0, 0.5), (1, 1.0), (3, 4.0), (6, 8.0)):
        delay = default_backoff(attempt)
        assert 0.75 * ceiling <= delay <= ceiling


def test_server_retry_verdict_is_obeyed() -> None:
    replies = [
        (400, {"x-should-retry": "true"}, {"error": "try again"}),
        (200, {}, {"ok": 1}),
        (503, {"x-should-retry": "false"}, {"error": "do not retry"}),
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["content-length"]))
            status, headers, body = replies.pop(0)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("retry-after-ms", "0")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/x"
    try:
        assert post_json(url, {}, {}) == {"ok": 1}
        try:
            post_json(url, {}, {})
        except LLMError as error:
            assert error.status == 503
        else:
            raise AssertionError("expected the 503 to fail at once")
        assert not replies
    finally:
        server.shutdown()


def test_ssl_context_adds_certifi_to_the_system_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ssl

    from bytely.ai.llm import http

    loaded: list[str] = []
    real = ssl.SSLContext.load_verify_locations

    def spy(self: ssl.SSLContext, cafile: str | None = None, **kw: Any) -> None:
        # On Windows the default context itself loads the system store
        # through this call (as `cadata`); only bundle files matter here.
        if cafile is not None:
            loaded.append(str(cafile))
        real(self, cafile=cafile, **kw)

    monkeypatch.setattr(ssl.SSLContext, "load_verify_locations", spy)

    # certifi missing: the system store alone.
    monkeypatch.setattr(http, "certifi_bundle", lambda: None)
    context = http.build_ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert loaded == []

    # certifi present: its bundle is added on top of the system store.
    system = ssl.create_default_context().cert_store_stats()["x509_ca"]
    bundle = tmp_path / "cacert.pem"
    bundle.write_text(ssl.DER_cert_to_PEM_cert(_any_system_ca()), "ascii")
    monkeypatch.setattr(http, "certifi_bundle", lambda: str(bundle))
    context = http.build_ssl_context()
    assert loaded == [str(bundle)]
    assert context.cert_store_stats()["x509_ca"] >= max(system, 1)

    # A bundle that cannot load is skipped, not fatal.
    loaded.clear()
    monkeypatch.setattr(
        http, "certifi_bundle", lambda: str(tmp_path / "no.pem")
    )
    assert http.build_ssl_context().verify_mode == ssl.CERT_REQUIRED
    assert loaded == [str(tmp_path / "no.pem")]


def _any_system_ca() -> bytes:
    """One CA certificate (DER) from the system store, for a real bundle."""
    import ssl

    certs = ssl.create_default_context().get_ca_certs(binary_form=True)
    if certs:
        return bytes(certs[0])
    import pytest

    pytest.skip("no system CA certificates to build a test bundle from")
    raise AssertionError  # unreachable
