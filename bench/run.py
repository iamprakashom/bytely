"""Time Bytely's build, edits, and queries on the pinned benchmark corpus.

Usage:
    python bench/run.py [NAME ...] [--size small|medium|large] [--runs 3]
                        [--mcp 100] [--no-memory] [--imports] [--json OUT]

Fetch the corpus first with `bench/fetch.py`. Each run copies a tree into a
fresh temporary directory and times, as wall-clock seconds:

- `cold`: the first build; `warm`: a rebuild with nothing changed
- each query from `corpus.toml` as a fresh CLI process (`ask`, `grep`,
  `callers`, `skeleton`, `map`), so these include interpreter startup
- `mcp_*`: one long-lived `mcp` session answering `--mcp` queries in turn
  (startup, then per-query latency), which is how agents use the graph
- edit scenarios, each followed by a timed rebuild, applied in order to
  the same copy: `body_edit` (a comment in the median-sized file),
  `new_symbol`, `changed_import`, `added_file`, and `deleted_file`
- `check` after the edits

Peak resident memory (process and children) is sampled every 10 ms unless
`--no-memory`, which avoids the sampler perturbing short commands. Bytely
builds also report per-stage timings (`BYTELY_PHASE_LOG`). Every metric
keeps its raw samples, min, median, p95, standard deviation, and how many
runs failed; expect noise when the machine is busy.

A second tool can be timed alongside Bytely by describing it in
`bench/reference.local.toml` (git-ignored):

    command = "tool"         # executable on PATH, same subcommands
    output_dir = "out"       # folder it writes into the indexed tree
    check = ["check"]        # optional; defaults to ["check", "."]
    [env]                    # extra environment for its runs
    SOME_VAR = "1"

Its MCP tools are matched to Bytely's by name suffix (`_find_code`, ...).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import psutil  # type: ignore[import-untyped]
from fetch import BENCH_DIR, SIZES, load_corpus, select, tree_path

REFERENCE_FILE = BENCH_DIR / "reference.local.toml"
SOURCE_EXTENSIONS = {".py", ".rs", ".js", ".mjs", ".cjs", ".jsx"}
SKIPPED_NAMES = {".git", "node_modules", "bytely", "__pycache__", ".venv"}
QUERIES = ("ask", "grep", "callers", "skeleton", "map")
SCENARIOS = (
    "body_edit",
    "new_symbol",
    "changed_import",
    "added_file",
    "deleted_file",
)
POLL_SECONDS = 0.01
PHASE_LOG_ENV = "BYTELY_PHASE_LOG"

# Query kind -> (MCP tool suffix, argument name).
MCP_TOOLS = {
    "ask": ("_find_code", "query"),
    "grep": ("_find_all", "pattern"),
    "callers": ("_trace_calls", "symbol"),
    "skeleton": ("_file_api", "file"),
    "map": ("_repo_map", None),
}

# What each edit scenario writes, per source language.
PYTHON = {
    "comment": "# bench\n",
    "symbol": "\n\ndef bench_added_symbol():\n    return 1\n",
    "import": "import os as bench_os\n",
}
RUST = {
    "comment": "// bench\n",
    "symbol": "\nfn bench_added_symbol() -> u32 {\n    1\n}\n",
    "import": "use std::fmt as bench_fmt;\n",
}
JAVASCRIPT = {
    "comment": "// bench\n",
    "symbol": "\nfunction benchAddedSymbol() {\n  return 1;\n}\n",
    "import": "const benchFs = require('fs');\n",
}
SNIPPETS = {
    ".py": PYTHON,
    ".rs": RUST,
    **dict.fromkeys((".js", ".mjs", ".cjs", ".jsx"), JAVASCRIPT),
}

Sample = tuple[float, float | None, dict[str, float] | None]


@dataclass
class Tool:
    """A CLI to benchmark: how to invoke it and what it leaves behind."""

    name: str
    prefix: list[str]
    env: dict[str, str] = field(default_factory=dict)
    output_dir: str = "bytely"
    check_args: list[str] = field(default_factory=lambda: ["check", "."])

    def build(self) -> list[str]:
        """Return the build command line."""
        return [*self.prefix, "build", "."]

    def check(self) -> list[str]:
        """Return the check command line."""
        return [*self.prefix, *self.check_args]

    def query(self, kind: str, argument: str | None) -> list[str]:
        """Return a query command line."""
        return [*self.prefix, kind, *([argument] if argument else [])]


def load_tools() -> list[Tool]:
    """Return Bytely, plus the local reference tool when one is set up."""
    # `-P`: don't put the measured tree first on sys.path, where its own
    # modules (Click's `types.py`) would shadow the stdlib.
    tools = [Tool("bytely", [sys.executable, "-P", "-m", "bytely"])]
    if REFERENCE_FILE.is_file():
        with REFERENCE_FILE.open("rb") as handle:
            config = tomllib.load(handle)
        executable = shutil.which(config["command"])
        if executable is None:
            sys.exit(
                f"{config['command']} (from {REFERENCE_FILE.name}) "
                "is not on PATH"
            )
        tools.append(
            Tool(
                "reference",
                [executable],
                env=config.get("env", {}),
                output_dir=config["output_dir"],
                check_args=config.get("check", ["check", "."]),
            )
        )
    return tools


class Runner:
    """Runs commands, timing them and optionally sampling their memory."""

    def __init__(self, memory: bool, phase_log: Path) -> None:
        """Set up a runner; `phase_log` collects Bytely's stage timings."""
        self.memory = memory
        self.phase_log = phase_log

    def run(self, command: list[str], cwd: Path, env: dict[str, str]) -> Sample:
        """Run a command that must succeed; return (s, peak MB, phases)."""
        self.phase_log.unlink(missing_ok=True)
        start = time.perf_counter()
        process = psutil.Popen(
            command,
            cwd=cwd,
            env={**os.environ, **env, PHASE_LOG_ENV: str(self.phase_log)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        sampler = _MemorySampler(process) if self.memory else None
        stdout, stderr = process.communicate()
        elapsed = time.perf_counter() - start
        peak = sampler.stop() if sampler else None
        if process.returncode != 0:
            raise RuntimeError(
                f"{' '.join(command)} failed in {cwd}:\n"
                f"{stdout.decode(errors='replace')}\n"
                f"{stderr.decode(errors='replace')}"
            )
        return elapsed, peak, self._phases()

    def _phases(self) -> dict[str, float] | None:
        try:
            lines = self.phase_log.read_text("utf-8").splitlines()
        except OSError:
            return None
        return json.loads(lines[-1])["phases"] if lines else None


class _MemorySampler:
    """Track the peak RSS of a process tree from a background thread."""

    def __init__(self, process: Any) -> None:
        self.process = process
        self.peak = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        while not self.done.is_set():
            try:
                family = [self.process, *self.process.children(recursive=True)]
                total = sum(p.memory_info().rss for p in family)
                self.peak = max(self.peak, total)
            except psutil.Error:
                pass
            self.done.wait(POLL_SECONDS)

    def stop(self) -> float:
        self.done.set()
        self.thread.join()
        return self.peak / 2**20


def source_files(tree: Path) -> list[Path]:
    """Return the tree's source files, smallest first."""
    return sorted(
        (
            path
            for path in tree.rglob("*")
            if path.suffix in SOURCE_EXTENSIONS
            and path.is_file()
            and not SKIPPED_NAMES.intersection(path.relative_to(tree).parts)
        ),
        key=lambda path: (path.stat().st_size, str(path)),
    )


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n" + text)


def touch_median_file(files: list[Path], work: Path, tree: Path) -> None:
    """Append a comment to the median-sized source file in the copy."""
    target = work / files[len(files) // 2].relative_to(tree)
    _append(target, SNIPPETS[target.suffix]["comment"])


def apply_scenario(
    scenario: str, files: list[Path], work: Path, tree: Path
) -> None:
    """Make one realistic edit to the working copy."""
    median = work / files[len(files) // 2].relative_to(tree)
    snippets = SNIPPETS[median.suffix]
    if scenario == "body_edit":
        _append(median, snippets["comment"])
    elif scenario == "new_symbol":
        _append(median, snippets["symbol"])
    elif scenario == "changed_import":
        _append(median, snippets["import"])
    elif scenario == "added_file":
        added = median.with_name(f"bench_added{median.suffix}")
        added.write_text(snippets["symbol"].lstrip(), encoding="utf-8")
    elif scenario == "deleted_file":
        # A different file from the edited one: a third of the way up.
        (work / files[len(files) // 3].relative_to(tree)).unlink()
    else:
        raise ValueError(scenario)


class McpSession:
    """A long-lived MCP server over stdio, one JSON message per line."""

    def __init__(self, command: list[str], cwd: Path, env: dict[str, str]):
        """Start the server and complete the MCP handshake."""
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env={**os.environ, **env},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.next_id = 0
        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "bytely-bench", "version": "0"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = self.request("tools/list", {}).get("tools", [])
        self.tools = [tool["name"] for tool in tools]

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
                raise RuntimeError(f"MCP server closed during {method}")
            message = json.loads(line)
            if message.get("id") != self.next_id:
                continue
            if "error" in message:
                raise RuntimeError(f"{method}: {message['error']}")
            result: dict[str, Any] = message.get("result", {})
            if result.get("isError"):
                raise RuntimeError(f"{method}: {result.get('content')}")
            return result

    def tool_for(self, suffix: str) -> str | None:
        """The server's tool whose name ends with `suffix`."""
        return next((t for t in self.tools if t.endswith(suffix)), None)

    def close(self) -> None:
        """Stop the server."""
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()


def measure_mcp(
    tool: Tool, work: Path, queries: dict[str, str], count: int
) -> dict[str, list[float]]:
    """Time an MCP session's startup and `count` queries, per query kind."""
    start = time.perf_counter()
    session = McpSession([*tool.prefix, "mcp"], work, tool.env)
    results: dict[str, list[float]] = {
        "mcp_startup": [time.perf_counter() - start]
    }
    try:
        calls = []
        for kind, (suffix, argument) in MCP_TOOLS.items():
            name = session.tool_for(suffix)
            if name is None or (argument and kind not in queries):
                continue
            arguments = {argument: queries[kind]} if argument else {}
            calls.append((kind, name, arguments))
        for index in range(count):
            kind, name, arguments = calls[index % len(calls)]
            began = time.perf_counter()
            try:
                session.request(
                    "tools/call", {"name": name, "arguments": arguments}
                )
            except RuntimeError:
                results.setdefault(f"mcp_{kind}_failed", []).append(1.0)
                continue
            results.setdefault(f"mcp_{kind}", []).append(
                time.perf_counter() - began
            )
    finally:
        session.close()
    return results


@dataclass
class Collected:
    """Every sample of one tool on one repository, across runs."""

    seconds: dict[str, list[float]] = field(default_factory=dict)
    peak_mb: dict[str, list[float]] = field(default_factory=dict)
    phases: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)

    def add(self, metric: str, sample: Sample) -> None:
        """Record one successful sample of a metric."""
        seconds, peak, phases = sample
        self.seconds.setdefault(metric, []).append(seconds)
        if peak is not None:
            self.peak_mb.setdefault(metric, []).append(peak)
        if phases is not None:
            self.phases.setdefault(metric, []).append(phases)

    def fail(self, metric: str, error: Exception) -> None:
        """Record a failed sample of a metric."""
        self.failures[metric] = self.failures.get(metric, 0) + 1
        print(
            f"    {metric} failed: {str(error).strip().splitlines()[-1][:100]}"
        )


def measure(
    tool: Tool,
    tree: Path,
    queries: dict[str, str],
    files: list[Path],
    skipped: set[str],
    runner: Runner,
    collected: Collected,
    mcp_queries: int,
) -> None:
    """Run every metric once for a tool on a fresh copy of a tree."""
    with tempfile.TemporaryDirectory(prefix="bytely-bench-") as temp_dir:
        work = Path(temp_dir) / tree.name
        shutil.copytree(tree, work, ignore=shutil.ignore_patterns(*skipped))

        def timed(metric: str, command: list[str]) -> None:
            try:
                collected.add(metric, runner.run(command, work, tool.env))
            except RuntimeError as error:
                collected.fail(metric, error)

        timed("cold", tool.build())
        timed("warm", tool.build())
        for kind in QUERIES:
            timed(kind, tool.query(kind, queries.get(kind)))
        if mcp_queries:
            try:
                latencies = measure_mcp(tool, work, queries, mcp_queries)
            except (RuntimeError, OSError, ValueError) as error:
                collected.fail("mcp_startup", error)
            else:
                for metric, values in latencies.items():
                    if metric.endswith("_failed"):
                        name = metric.removesuffix("_failed")
                        collected.failures[name] = collected.failures.get(
                            name, 0
                        ) + len(values)
                    else:
                        collected.seconds.setdefault(metric, []).extend(values)
        for scenario in SCENARIOS:
            apply_scenario(scenario, files, work, tree)
            timed(scenario, tool.build())
        timed("check", tool.check())


def startup(tool: Tool, runner: Runner) -> float:
    """Return the median seconds of `--version` over five runs."""
    quiet = Runner(memory=False, phase_log=runner.phase_log)
    return statistics.median(
        quiet.run([*tool.prefix, "--version"], Path.cwd(), tool.env)[0]
        for _ in range(5)
    )


def import_times(limit: int = 15) -> list[tuple[str, float]]:
    """Return Bytely's slowest imports at startup, cumulative ms."""
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "bytely", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in completed.stderr.splitlines():
        if not line.startswith("import time:") or "cumulative" in line:
            continue
        _, cumulative, name = (
            part.strip()
            for part in line.removeprefix("import time:").split("|")
        )
        rows.append((name.strip(), int(cumulative) / 1000))
    return sorted(rows, key=lambda row: row[1], reverse=True)[:limit]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def summarize(collected: Collected) -> dict[str, dict[str, Any]]:
    """Reduce samples to statistics per metric, keeping the raw samples."""
    metrics = list(collected.seconds) + [
        m for m in collected.failures if m not in collected.seconds
    ]
    summary: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        seconds = collected.seconds.get(metric, [])
        row: dict[str, Any] = {
            "samples": [round(s, 4) for s in seconds],
            "failures": collected.failures.get(metric, 0),
        }
        if seconds:
            row.update(
                min_s=round(min(seconds), 4),
                median_s=round(statistics.median(seconds), 4),
                p95_s=round(_percentile(seconds, 0.95), 4),
                stdev_s=round(statistics.stdev(seconds), 4)
                if len(seconds) > 1
                else 0.0,
            )
        if collected.peak_mb.get(metric):
            row["peak_mb"] = round(max(collected.peak_mb[metric]), 1)
        if collected.phases.get(metric):
            row["phases_median_s"] = {
                phase: round(
                    statistics.median(
                        run.get(phase, 0.0) for run in collected.phases[metric]
                    ),
                    4,
                )
                for phase in collected.phases[metric][0]
            }
        summary[metric] = row
    return summary


def git_head() -> str:
    """Return Bytely's commit, marked dirty if any tracked or bench file is."""
    root = BENCH_DIR.parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True
        ).stdout.strip()

    head = git("rev-parse", "--short", "HEAD")
    # Tracked changes anywhere, plus untracked files in what the
    # benchmark runs (agent configs and other local files don't count).
    dirty = git("status", "--porcelain", "--untracked-files=no") or git(
        "status", "--porcelain", "--", "src", "bench", "pyproject.toml"
    )
    return f"{head}-dirty" if dirty else head


def print_table(tool: str, summary: dict[str, dict[str, Any]]) -> None:
    """Print one tool's results for one repository."""
    print(f"  {tool}")
    for metric, row in summary.items():
        if "median_s" not in row:
            print(f"    {metric:16} failed x{row['failures']}")
            continue
        peak = f"{row['peak_mb']:>8.1f} MB" if "peak_mb" in row else ""
        failed = f"  ({row['failures']} failed)" if row["failures"] else ""
        print(
            f"    {metric:16} {row['median_s']:>8.3f}s "
            f"{row['p95_s']:>8.3f}s {row['stdev_s']:>7.3f}s "
            f"n={len(row['samples']):<4}{peak}{failed}"
        )


def print_phases(summary: dict[str, dict[str, Any]]) -> None:
    """Print Bytely's per-stage build timings."""
    for metric, row in summary.items():
        phases = row.get("phases_median_s")
        if phases:
            cells = " ".join(
                f"{name}={seconds:.2f}"
                for name, seconds in phases.items()
                if seconds >= 0.005
            )
            print(f"    {metric:16} stages: {cells}")


def main() -> int:
    """Benchmark the selected repositories."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--size", choices=SIZES, help="largest size to run")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--mcp",
        type=int,
        default=100,
        metavar="N",
        help="queries per MCP session (0 skips the MCP benchmark)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="don't sample peak memory (less perturbation of short runs)",
    )
    parser.add_argument("--json", type=Path, help="write results here")
    parser.add_argument(
        "--imports",
        action="store_true",
        help="also list Bytely's slowest startup imports",
    )
    args = parser.parse_args()

    corpus = load_corpus()
    tools = load_tools()
    skipped = SKIPPED_NAMES | {tool.output_dir for tool in tools}
    with tempfile.TemporaryDirectory(prefix="bytely-bench-log-") as log_dir:
        runner = Runner(not args.no_memory, Path(log_dir) / "phases.jsonl")
        results: dict[str, Any] = {
            "bytely": git_head(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "runs": args.runs,
            "mcp_queries": args.mcp,
            "memory_sampled": not args.no_memory,
            "startup_s": {
                tool.name: round(startup(tool, runner), 3) for tool in tools
            },
            "repos": {},
        }
        print(
            f"bytely {results['bytely']} on {results['platform']}, "
            f"python {results['python']}, {results['cpus']} cpus"
        )
        print(f"startup (--version, median of 5): {results['startup_s']}")
        if args.imports:
            results["imports_ms"] = import_times()
            print("slowest imports (cumulative ms):")
            for module, ms in results["imports_ms"]:
                print(f"  {ms:>8.1f}  {module}")

        for name, repo in select(corpus, args.names, args.size).items():
            tree = tree_path(name, repo)
            if not tree.is_dir():
                sys.exit(f"{name} is not fetched; run bench/fetch.py {name}")
            files = source_files(tree)
            print(
                f"\n{name} ({repo['language']}, {repo['tag']}): "
                f"{len(files)} source files, {args.runs} runs"
                "   (median / p95 / stdev / samples / peak)"
            )
            entry: dict[str, Any] = {
                "commit": repo["commit"],
                "files": len(files),
            }
            for tool in tools:
                collected = Collected()
                for _ in range(args.runs):
                    measure(
                        tool,
                        tree,
                        repo.get("queries", {}),
                        files,
                        skipped,
                        runner,
                        collected,
                        args.mcp,
                    )
                entry[tool.name] = summarize(collected)
                print_table(tool.name, entry[tool.name])
                if tool.name == "bytely":
                    print_phases(entry[tool.name])
            results["repos"][name] = entry

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
