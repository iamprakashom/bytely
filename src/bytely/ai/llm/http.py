"""JSON over HTTPS with retries and connection reuse, on the standard library.

Retries follow the official OpenAI and Anthropic SDKs: the server's
`x-should-retry` header wins; otherwise request and lock timeouts (408,
409), rate limits (429), server errors (5xx), and network failures are
retried, waiting as long as `retry-after-ms` or `Retry-After` (seconds or
a date) asks, else 0.5 s doubling to 8 s with jitter. Anything else (a
bad key, a bad request) fails at once, with the provider's error text in
the message so the caller can classify it.

Connections are kept alive and reused, one per host per thread: a deep
build makes thousands of calls, and a fresh TCP + TLS handshake for each
costs 50-150 ms. A pooled connection the server has closed in the meantime
is reopened once, transparently, without counting as a retry. When a proxy
is configured (`HTTPS_PROXY` and friends), requests go through `urllib`,
which applies it.
"""

from __future__ import annotations

import contextlib
import email.utils
import http.client
import json
import random
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from bytely.ai.llm.types import LLMError, transport_retries

TIMEOUT_SECONDS = 180.0
# Retry policy: the official OpenAI and Anthropic SDKs' rules.
INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 8.0
MAX_HONORED_WAIT = 60.0
# Failures that mean a reused connection went stale, not that the request
# failed: the server closed it while it sat idle in the pool.
_STALE = (
    http.client.RemoteDisconnected,
    http.client.CannotSendRequest,
    http.client.BadStatusLine,
    ConnectionResetError,
    BrokenPipeError,
)


class _Reply:
    """A finished HTTP exchange."""

    def __init__(
        self, status: int, reason: str, headers: Any, body: bytes
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers = headers
        self.body = body


class _Pool(threading.local):
    """This thread's open connections, by (scheme, host, port)."""

    def __init__(self) -> None:
        self.connections: dict[
            tuple[str, str, int], http.client.HTTPConnection
        ] = {}


_pool = _Pool()
_ssl_context: ssl.SSLContext | None = None
_ssl_lock = threading.Lock()


def certifi_bundle() -> str | None:
    """The `certifi` CA bundle's path, when that package is installed."""
    try:
        import certifi  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        return None
    try:
        path = certifi.where()
    except Exception:  # noqa: BLE001 - a broken install is the same as none
        return None
    return path if isinstance(path, str) else None


def build_ssl_context() -> ssl.SSLContext:
    """Verify servers against the system's CAs plus `certifi`'s, if present.

    Additive on purpose. The system store (with `SSL_CERT_FILE` and
    `SSL_CERT_DIR`) keeps corporate roots working, as on Windows; the
    `certifi` bundle covers Pythons with no usable system store, such as
    the python.org macOS builds before "Install Certificates.command" has
    run. A bundle that fails to load is skipped, never fatal.
    """
    context = ssl.create_default_context()
    bundle = certifi_bundle()
    if bundle:
        with contextlib.suppress(OSError, ssl.SSLError):
            context.load_verify_locations(cafile=bundle)
    return context


def _context() -> ssl.SSLContext:
    global _ssl_context
    with _ssl_lock:
        if _ssl_context is None:
            _ssl_context = build_ssl_context()
        return _ssl_context


def _uses_proxy(parts: urllib.parse.SplitResult) -> bool:
    proxies = urllib.request.getproxies()
    return parts.scheme in proxies and not urllib.request.proxy_bypass(
        parts.hostname or ""
    )


def close_connections() -> None:
    """Close this thread's pooled connections."""
    for connection in _pool.connections.values():
        connection.close()
    _pool.connections.clear()


def _connection(
    parts: urllib.parse.SplitResult, timeout: float
) -> tuple[tuple[str, str, int], http.client.HTTPConnection, bool]:
    """This thread's connection to the host, and whether it was reused."""
    https = parts.scheme == "https"
    host = parts.hostname or ""
    port = parts.port or (443 if https else 80)
    key = (parts.scheme, host, port)
    existing = _pool.connections.get(key)
    if existing is not None:
        return key, existing, True
    connection: http.client.HTTPConnection = (
        http.client.HTTPSConnection(
            host, port, timeout=timeout, context=_context()
        )
        if https
        else http.client.HTTPConnection(host, port, timeout=timeout)
    )
    _pool.connections[key] = connection
    return key, connection, False


def _drop(key: tuple[str, str, int]) -> None:
    connection = _pool.connections.pop(key, None)
    if connection is not None:
        connection.close()


def _send_pooled(
    url: str, data: bytes, headers: dict[str, str], timeout: float
) -> _Reply:
    parts = urllib.parse.urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    for attempt in range(2):
        key, connection, reused = _connection(parts, timeout)
        try:
            connection.request("POST", path, body=data, headers=headers)
            response = connection.getresponse()
            body = response.read()
        except _STALE:
            _drop(key)
            if reused and attempt == 0:
                continue  # the idle connection was closed; open a new one
            raise
        except (OSError, http.client.HTTPException):
            _drop(key)
            raise
        if response.will_close:
            _drop(key)
        return _Reply(response.status, response.reason, response.headers, body)
    raise ConnectionError(f"could not reach {url}")


def _send_urllib(
    url: str, data: bytes, headers: dict[str, str], timeout: float
) -> _Reply:
    request = urllib.request.Request(  # noqa: S310 - https or a local proxy
        url, data=data, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=timeout, context=_context()
        ) as response:
            return _Reply(
                response.status,
                response.reason,
                response.headers,
                response.read(),
            )
    except urllib.error.HTTPError as error:
        try:
            body = error.read()
        except OSError:
            body = b""
        return _Reply(error.code, str(error.reason), error.headers, body)


def should_retry(status: int, headers: Any) -> bool:
    """Whether a failed response is worth retrying, as the SDKs decide.

    The server's own `x-should-retry` wins; otherwise request timeouts
    (408), lock timeouts (409), rate limits (429), and server errors (5xx,
    including Anthropic's 529 "overloaded").
    """
    verdict = headers.get("x-should-retry") if headers is not None else None
    if verdict == "true":
        return True
    if verdict == "false":
        return False
    return status in (408, 409, 429) or status >= 500


def retry_after(headers: Any, now: float | None = None) -> float | None:
    """The wait the server asked for, in seconds, if it asked.

    `retry-after-ms` first, then `Retry-After` as seconds or as an HTTP
    date. A wait beyond `MAX_HONORED_WAIT` is capped: an hour-long quota
    reset should fail the call (and let the pass report it), not stall a
    worker for an hour.
    """
    if headers is None:
        return None
    wait: float | None = None
    millis = headers.get("retry-after-ms")
    if millis:
        try:
            wait = float(millis) / 1000
        except ValueError:
            wait = None
    value = headers.get("retry-after")
    if wait is None and value:
        try:
            wait = float(value)
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(value)
            except (TypeError, ValueError):
                when = None
            if when is not None:
                wait = when.timestamp() - (time.time() if now is None else now)
    if wait is None:
        return None
    return min(max(0.0, wait), MAX_HONORED_WAIT)


def default_backoff(attempt: int) -> float:
    """The SDKs' backoff: 0.5 s doubling to 8 s, minus up to 25% jitter."""
    base = min(INITIAL_RETRY_DELAY * 2.0**attempt, MAX_RETRY_DELAY)
    return base * (1 - random.random() * 0.25)  # noqa: S311


def _error_text(reply: _Reply) -> str:
    text = reply.body.decode("utf-8", "replace")
    try:
        data = json.loads(text)
        detail = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(detail, dict):
            text = str(detail.get("message") or detail)
        else:
            text = str(detail)
    except ValueError:
        pass
    return f"{reply.status} {reply.reason}: {text.strip()[:500]}"


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    retries: int | None = None,
    timeout: float = TIMEOUT_SECONDS,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """POST `payload` as JSON and return the decoded JSON reply."""
    data = json.dumps(payload).encode("utf-8")
    sent = {"content-type": "application/json", **headers}
    send = (
        _send_urllib
        if _uses_proxy(urllib.parse.urlsplit(url))
        else _send_pooled
    )
    attempts = 1 + (transport_retries() if retries is None else retries)
    for attempt in range(attempts):
        wait: float | None = None
        try:
            reply = send(url, data, sent, timeout)
        except (OSError, http.client.HTTPException) as error:
            if attempt == attempts - 1:
                raise LLMError(
                    f"network error calling {url}: {error}"
                ) from error
        else:
            if 200 <= reply.status < 300:
                try:
                    decoded = json.loads(reply.body.decode("utf-8"))
                except ValueError as error:
                    raise LLMError(
                        f"unreadable reply from {url}: {error}"
                    ) from error
                if not isinstance(decoded, dict):
                    raise LLMError(f"unexpected reply from {url}")
                return decoded
            message = _error_text(reply)
            if (
                not should_retry(reply.status, reply.headers)
                or attempt == attempts - 1
            ):
                raise LLMError(message, reply.status)
            wait = retry_after(reply.headers)
        sleep(wait if wait is not None else default_backoff(attempt))
    raise LLMError(f"request to {url} failed")
