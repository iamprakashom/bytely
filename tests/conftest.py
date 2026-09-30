"""Shared pytest fixtures for this repository."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest

import bytely.graph.check as graph_check

TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "pytest-workspace"


def pytest_configure(config: pytest.Config) -> None:
    """Create the test temp root inside the workspace."""
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def _workspace_temporary_directory(prefix: str) -> Iterator[str]:
    temp_path = TEMP_ROOT / f"{prefix}{uuid4().hex}"
    temp_path.mkdir()
    try:
        yield str(temp_path)
    finally:
        rmtree(temp_path)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Provide an isolated workspace temp directory with inherited access."""
    original_cwd = Path.cwd()
    test_temp_path = TEMP_ROOT / f"{request.node.name}-{uuid4().hex}"
    test_temp_path.mkdir()
    try:
        yield test_temp_path
    finally:
        os.chdir(original_cwd)
        rmtree(test_temp_path)


@pytest.fixture(autouse=True)
def route_graph_temp_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep graph freshness-check temp output within the test sandbox."""
    monkeypatch.setattr(
        graph_check, "TemporaryDirectory", _workspace_temporary_directory
    )
