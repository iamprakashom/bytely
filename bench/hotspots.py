"""Profile one Bytely command on a benchmark repository with cProfile.

Usage:
    python bench/hotspots.py NAME [--stage cold|warm|one_file|check]
                            [--top 30] [--out FILE.prof]

The repository (fetched with `bench/fetch.py`) is copied into a temporary
directory and brought to the state the stage needs: `warm`, `one_file`,
and `check` run a build first, and `one_file` then changes the median-sized
source file. Only the stage itself is profiled, in-process. The hottest
functions are printed by cumulative and by own time; `--out` keeps the
raw profile for a viewer such as snakeviz.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import shutil
import sys
import tempfile
from pathlib import Path

from fetch import load_corpus, select, tree_path
from run import SKIPPED_NAMES, source_files, touch_median_file

STAGES = ("cold", "warm", "one_file", "check")


def bytely(*args: str) -> None:
    """Run the Bytely CLI in-process, as `bytely ARGS` would."""
    from bytely.cli import main

    try:
        main(args=list(args), standalone_mode=False)
    except SystemExit as error:
        if error.code not in (0, None):
            raise


def main() -> int:
    """Profile the selected stage on one repository."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name")
    parser.add_argument("--stage", choices=STAGES, default="cold")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", type=Path, help="save the raw profile")
    args = parser.parse_args()

    repo = select(load_corpus(), [args.name], None)[args.name]
    tree = tree_path(args.name, repo)
    if not tree.is_dir():
        sys.exit(f"{args.name} is not fetched; run bench/fetch.py")

    with tempfile.TemporaryDirectory(prefix="bytely-profile-") as temp_dir:
        work = Path(temp_dir) / tree.name
        ignore = shutil.ignore_patterns(*SKIPPED_NAMES)
        shutil.copytree(tree, work, ignore=ignore)
        os.chdir(work)
        if args.stage != "cold":
            bytely("build", ".")
        if args.stage == "one_file":
            touch_median_file(source_files(tree), work, tree)
        command = ("check", ".") if args.stage == "check" else ("build", ".")

        profiler = cProfile.Profile()
        profiler.runcall(bytely, *command)
        os.chdir(Path(temp_dir).parent)

    if args.out:
        profiler.dump_stats(args.out)
    for order in ("cumulative", "tottime"):
        buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=buffer)
        stats.strip_dirs().sort_stats(order).print_stats(args.top)
        print(f"=== {args.name} {args.stage}, by {order} ===")
        print(buffer.getvalue().split("\n", 6)[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
