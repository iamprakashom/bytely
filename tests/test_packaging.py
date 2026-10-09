"""Checks for metadata and entry points required by published artifacts."""

from __future__ import annotations

from importlib.metadata import distribution, version

import bytely


def test_installed_version_matches_package_version() -> None:
    assert version("bytely") == bytely.__version__


def test_console_entry_point_is_published() -> None:
    entry_points = distribution("bytely").entry_points
    bytely_entry = next(
        entry_point
        for entry_point in entry_points
        if entry_point.group == "console_scripts"
        and entry_point.name == "bytely"
    )
    assert bytely_entry.value == "bytely.cli:main"
