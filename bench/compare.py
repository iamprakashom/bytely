"""Check a benchmark result against the budgets and the committed baseline.

Usage:
    python bench/compare.py RESULT.json [--baseline bench/baseline.json]
    python bench/compare.py RESULT.json --write-baseline

`RESULT.json` comes from `bench/run.py --json`. The check fails (exit 1)
when a metric exceeds a budget in `budgets.toml`, or regresses past the
allowed fraction and noise floor relative to the baseline. Repositories
or metrics missing from either side are skipped, so a partial run can be
checked against a full baseline.

`--write-baseline` stores the result's Bytely numbers (medians and peak
memory only, no other tool's) as the new baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
BUDGETS_FILE = BENCH_DIR / "budgets.toml"
BASELINE_FILE = BENCH_DIR / "baseline.json"
CONTEXT_KEYS = ("bytely", "python", "platform", "cpus", "runs", "mcp_queries")


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a file."""
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        sys.exit(f"{path} does not hold a JSON object")
    return data


def baseline_of(result: dict[str, Any]) -> dict[str, Any]:
    """Reduce a result to Bytely's medians and peaks, per repo and metric."""
    repos = {}
    for name, entry in result.get("repos", {}).items():
        metrics = {}
        for metric, row in entry.get("bytely", {}).items():
            if "median_s" not in row:
                continue
            kept = {"median_s": row["median_s"]}
            if "peak_mb" in row:
                kept["peak_mb"] = row["peak_mb"]
            metrics[metric] = kept
        repos[name] = {
            "commit": entry.get("commit"),
            "files": entry.get("files"),
            "metrics": metrics,
        }
    return {
        **{key: result.get(key) for key in CONTEXT_KEYS},
        "startup_s": result.get("startup_s", {}).get("bytely"),
        "repos": repos,
    }


def check_budgets(result: dict[str, Any], budgets: dict[str, Any]) -> list[str]:
    """Return a line per budget the result exceeds."""
    problems = []
    startup = result.get("startup_s", {}).get("bytely")
    if startup is not None and startup > budgets["startup_s"]:
        problems.append(
            f"startup: {startup:.3f}s > budget {budgets['startup_s']}s"
        )
    for name, entry in result.get("repos", {}).items():
        for metric, row in entry.get("bytely", {}).items():
            if row.get("failures"):
                problems.append(f"{name} {metric}: {row['failures']} failed")
            median = row.get("median_s")
            if (
                metric in budgets["queries"]
                and median is not None
                and median > budgets["query_s"]
            ):
                problems.append(
                    f"{name} {metric}: {median:.3f}s > budget "
                    f"{budgets['query_s']}s"
                )
            peak = row.get("peak_mb")
            if peak is not None and peak > budgets["peak_mb"]:
                problems.append(
                    f"{name} {metric}: {peak:.0f} MB > budget "
                    f"{budgets['peak_mb']} MB"
                )
    return problems


def check_regressions(
    result: dict[str, Any], baseline: dict[str, Any], budgets: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return (regressions, improvements) against the baseline."""
    allowed = budgets["regression"]
    regressions, improvements = [], []
    for name, entry in result.get("repos", {}).items():
        base = baseline.get("repos", {}).get(name)
        if base is None:
            continue
        if base.get("commit") != entry.get("commit"):
            regressions.append(f"{name}: corpus commit differs; not compared")
            continue
        for metric, row in entry.get("bytely", {}).items():
            old = base["metrics"].get(metric)
            if old is None:
                continue
            for key, floor, unit in (
                ("median_s", budgets["noise_floor_s"], "s"),
                ("peak_mb", budgets["noise_floor_mb"], " MB"),
            ):
                if key not in row or key not in old:
                    continue
                new_value, old_value = row[key], old[key]
                change = (new_value - old_value) / old_value if old_value else 0
                line = (
                    f"{name} {metric} {key}: {old_value:g}{unit} -> "
                    f"{new_value:g}{unit} ({change:+.0%})"
                )
                if change > allowed and new_value - old_value > floor:
                    regressions.append(line)
                elif change < -allowed and old_value - new_value > floor:
                    improvements.append(line)
    return regressions, improvements


def main() -> int:
    """Check a result, or store it as the baseline."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("result", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE_FILE)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="store this result as the baseline instead of checking it",
    )
    args = parser.parse_args()
    result = load_json(args.result)

    if args.write_baseline:
        args.baseline.write_text(
            json.dumps(baseline_of(result), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.baseline}")
        return 0

    with BUDGETS_FILE.open("rb") as handle:
        budgets = tomllib.load(handle)
    problems = check_budgets(result, budgets)
    improvements: list[str] = []
    if args.baseline.is_file():
        regressions, improvements = check_regressions(
            result, load_json(args.baseline), budgets
        )
        problems += regressions
    else:
        print(f"no baseline at {args.baseline}; checking budgets only")

    for line in improvements:
        print(f"improved  {line}")
    for line in problems:
        print(f"FAIL      {line}")
    print(f"{len(problems)} problem(s), {len(improvements)} improvement(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
