"""Compare Bytely graphs with recorded reference-implementation baselines.

Baselines come from `scripts/record_reference_baselines.py`. Every difference
between a fixture's baseline and Bytely's output must appear, with a reason
code, in `tests/parity/divergences/<fixture>.json`. A new difference fails
the test, and so does a listed one that no longer occurs, so the manifests
always describe exactly how Bytely differs from the reference implementation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from bytely.graph.build import build_graph
from bytely.graph.write import read_graph
from tests.parity.snapshot import REASONS, diff, load_json, normalize_bytely

PARITY_ROOT = Path(__file__).resolve().parent / "parity"
FIXTURES = sorted(
    path.name for path in (PARITY_ROOT / "fixtures").iterdir() if path.is_dir()
)

pytestmark = pytest.mark.parity


def _bytely_snapshot(fixture: str, work: Path) -> dict[str, Any]:
    tree = work / fixture
    shutil.copytree(PARITY_ROOT / "fixtures" / fixture, tree)
    build_graph(str(tree))
    graph = read_graph(str(tree / "bytely"))
    assert graph is not None
    return normalize_bytely(graph)


def _as_key(item: list[Any]) -> str:
    return json.dumps(item, ensure_ascii=False)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_fixture_differs_from_the_reference_only_as_documented(
    fixture: str, tmp_path: Path
) -> None:
    baseline = load_json(PARITY_ROOT / "baselines" / f"{fixture}.json")
    manifest = load_json(PARITY_ROOT / "divergences" / f"{fixture}.json")

    unknown = {entry["reason"] for entry in manifest} - REASONS.keys()
    assert not unknown, f"unknown reason codes: {sorted(unknown)}"

    actual = {
        _as_key(item)
        for item in diff(baseline, _bytely_snapshot(fixture, tmp_path))
    }
    expected = {_as_key(entry["diff"]) for entry in manifest}

    unexpected = sorted(actual - expected)
    stale = sorted(expected - actual)
    message = []
    if unexpected:
        message.append(
            "New differences from the reference implementation; fix them "
            "or add them to the manifest with a reason:\n"
            + "\n".join(
                f'  {{"diff": {item}, "reason": "?"}},' for item in unexpected
            )
        )
    if stale:
        message.append(
            "Listed differences that no longer occur; remove them:\n"
            + "\n".join(f"  {item}" for item in stale)
        )
    assert not message, "\n\n".join(message)


def test_every_parity_fixture_has_a_baseline_and_manifest() -> None:
    for fixture in FIXTURES:
        assert (PARITY_ROOT / "baselines" / f"{fixture}.json").is_file()
        assert (PARITY_ROOT / "divergences" / f"{fixture}.json").is_file()
