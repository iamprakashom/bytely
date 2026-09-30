"""How many tokens a query saved, and what they were worth.

Each retrieval output opens with one line estimating the tokens it saved
versus reading the files it covers whole. The line is a header, not a
footer: hosts and agents clip long output from the end, and the number is
what the PostToolUse hook sums into the session total. `sum_savings` is
the reader for exactly the text `savings_line` writes.

Money is only ever shown at a measured rate: the Stop hook prices each
turn from the transcript's `usage`, and every surface falls back to tokens
alone when no rate is known rather than inventing one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bytely.graph.types import GraphV1

PREFIX = "[bytely] tokens saved ≈"
_FOOTER = re.compile(r"\[bytely\] tokens saved ≈ ([\d,]+)")
_PATH_TOKEN = re.compile(r"[\w./\\-]+")


def to_tokens(chars: int) -> int:
    """Rough tokens for a text length (about four characters per token)."""
    return round(chars / 4)


@dataclass(frozen=True)
class Baseline:
    """What reading the covered files whole would have cost."""

    files: int
    chars: int


def baseline_for(graph: GraphV1, paths: Iterable[str]) -> Baseline | None:
    """The summed size of the distinct `paths`, from the file nodes' sizes.

    None when no path has a known size, so no estimate is claimed at all.
    """
    sizes = {
        node.path: node.chars
        for node in graph.nodes
        if node.kind == "file" and node.chars is not None
    }
    files = 0
    chars = 0
    for path in dict.fromkeys(paths):
        size = sizes.get(path)
        if size is not None:
            files += 1
            chars += size
    return Baseline(files, chars) if files else None


def paths_in(graph: GraphV1, text: str) -> list[str]:
    """Indexed file paths the text names (as `path` or `path:L1-L9`)."""
    known = {node.path for node in graph.nodes if node.kind == "file"}
    found = []
    for token in _PATH_TOKEN.findall(text):
        path = token.replace("\\", "/").rstrip(".")
        if path in known:
            found.append(path)
    return list(dict.fromkeys(found))


def format_dollars(usd: float) -> str:
    """`$0.04`, or `<$0.01` for amounts too small to show."""
    return "<$0.01" if usd < 0.01 else f"${usd:.2f}"


def turn_nudge(saved: int, rate: float | None) -> str:
    """The ask to report a turn total, riding along in every output.

    Deliberately free of the `[bytely] tokens saved ≈ N` pattern, so its
    example is never counted as a second saving.
    """
    sum_hint = " — sum each such line across your bytely calls — e.g. "
    if rate is None or saved <= 0:
        return (
            " At the end of your reply, tell the user the total bytely tokens "
            f'saved this turn{sum_hint}"⚡ bytely saved ~N tokens this turn".'
        )
    usd = saved * rate / 1_000_000
    return (
        f" This call is worth {format_dollars(usd)} at the rate this session "
        "is actually paying for input tokens. At the end of your reply, tell "
        "the user the total bytely tokens saved this turn and what they were "
        f'worth{sum_hint}"⚡ bytely saved ~N tokens (~$X) this turn".'
    )


def savings_line(
    body: str, baseline: Baseline | None, rate: float | None = None
) -> str:
    """The one-line estimate, or "" when there is nothing honest to claim.

    Nothing is claimed without a baseline, or when the output is not
    smaller than the files (tiny files, where pointers cost as much).
    """
    if baseline is None or baseline.chars <= 0:
        return ""
    pack = to_tokens(len(body))
    base = to_tokens(baseline.chars)
    if base <= pack:
        return ""
    saved = base - pack
    percent = round(saved / base * 100)
    return (
        f"{PREFIX} {saved:,} ({percent}%) — this output ≈ {pack:,} tok vs "
        f"reading the {baseline.files} file(s) it covers whole ≈ {base:,} tok "
        "(estimate)." + turn_nudge(saved, rate)
    )


def with_savings(
    body: str, baseline: Baseline | None, rate: float | None = None
) -> str:
    """`body` with its savings line on top (once: it is summed by hooks)."""
    line = savings_line(body, baseline, rate)
    return f"{line}\n\n{body}" if line else body


def sum_savings(text: str) -> int:
    """The total of every savings line in `text`."""
    return sum(
        int(match.group(1).replace(",", "")) for match in _FOOTER.finditer(text)
    )


# Input list price per million tokens, by model id prefix.
_INPUT_USD_PER_MTOK: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"^claude-(fable|mythos)-5"), 10.0),
    (re.compile(r"^claude-opus-(5|4-[678])"), 5.0),
    (re.compile(r"^claude-sonnet-5"), 2.0),
    (re.compile(r"^claude-sonnet-4-[56]"), 3.0),
    (re.compile(r"^claude-haiku-4-5"), 1.0),
)
CACHE_CREATE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


def input_usd_per_mtok(model: object) -> float | None:
    """The input list price for a model id, or None when it is unknown."""
    if not isinstance(model, str):
        return None
    for pattern, usd in _INPUT_USD_PER_MTOK:
        if pattern.match(model):
            return usd
    return None


@dataclass(frozen=True)
class Usage:
    """One API response's input usage."""

    model: str
    input: int
    cache_create: int
    cache_read: int

    @property
    def tokens(self) -> int:
        """Every input token, however it was billed."""
        return self.input + self.cache_create + self.cache_read

    def cost_micros(self) -> int | None:
        """What the input cost in micro-dollars, or None if unpriced."""
        price = input_usd_per_mtok(self.model)
        if price is None:
            return None
        weighted = (
            self.input
            + self.cache_create * CACHE_CREATE_MULTIPLIER
            + self.cache_read * CACHE_READ_MULTIPLIER
        )
        return round(weighted * price)


def dollars_saved(
    saved: int, cost_micros: object, tokens_billed: object
) -> float | None:
    """Saved tokens priced at the session's blended rate, if one exists."""
    if (
        saved <= 0
        or not isinstance(cost_micros, (int, float))
        or not isinstance(tokens_billed, (int, float))
        or cost_micros <= 0
        or tokens_billed <= 0
    ):
        return None
    return saved * (cost_micros / tokens_billed) / 1_000_000
