"""Fetch the benchmark corpus at its pinned commits.

Usage:
    python bench/fetch.py [NAME ...] [--size small|medium|large]

Each repository is shallow-fetched at the exact commit named in
`corpus.toml` into `.bench-cache/<name>/`. A repository with `subdir`
checks out only that folder. Re-running skips repositories already at
their pinned commit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
CORPUS_FILE = BENCH_DIR / "corpus.toml"
CACHE_DIR = BENCH_DIR.parent / ".bench-cache"
SIZES = ("small", "medium", "large")


def load_corpus() -> dict[str, dict[str, Any]]:
    """Return the corpus manifest, keyed by repository name."""
    with CORPUS_FILE.open("rb") as handle:
        return tomllib.load(handle)


def select(
    corpus: dict[str, dict[str, Any]], names: list[str], size: str | None
) -> dict[str, dict[str, Any]]:
    """Pick repositories by name, and up to a size, from the corpus."""
    unknown = sorted(set(names) - set(corpus))
    if unknown:
        sys.exit(f"not in corpus.toml: {', '.join(unknown)}")
    chosen = {name: corpus[name] for name in names} if names else corpus
    if size is not None:
        limit = SIZES.index(size)
        chosen = {
            name: repo
            for name, repo in chosen.items()
            if SIZES.index(repo["size"]) <= limit
        }
    return chosen


def tree_path(name: str, repo: dict[str, Any]) -> Path:
    """Return the folder that is benchmarked for a repository."""
    checkout = CACHE_DIR / name
    return checkout / repo["subdir"] if "subdir" in repo else checkout


def _git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def fetch(name: str, repo: dict[str, Any]) -> Path:
    """Check out a repository at its pinned commit and return its tree."""
    checkout = CACHE_DIR / name
    if (checkout / ".git").is_dir():
        head = _git("rev-parse", "HEAD", cwd=checkout)
        if head == repo["commit"]:
            print(f"{name}: up to date")
            return tree_path(name, repo)
    else:
        checkout.mkdir(parents=True, exist_ok=True)
        _git("init", "--quiet", cwd=checkout)
        _git("remote", "add", "origin", repo["url"], cwd=checkout)
    if "subdir" in repo:
        _git("sparse-checkout", "set", repo["subdir"], cwd=checkout)
    print(f"{name}: fetching {repo['tag']} ({repo['commit'][:10]})")
    _git(
        "fetch",
        "--quiet",
        "--depth",
        "1",
        "origin",
        repo["commit"],
        cwd=checkout,
    )
    _git("checkout", "--quiet", "--force", "FETCH_HEAD", cwd=checkout)
    return tree_path(name, repo)


def main() -> int:
    """Fetch the selected repositories."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--size", choices=SIZES, help="largest size to fetch")
    args = parser.parse_args()
    for name, repo in select(load_corpus(), args.names, args.size).items():
        fetch(name, repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
