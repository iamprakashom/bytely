"""The CI report that posts test results to a pull request."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".circleci" / "pr_report.py"
_spec = importlib.util.spec_from_file_location("pr_report", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
pr_report = importlib.util.module_from_spec(_spec)
sys.modules["pr_report"] = pr_report
_spec.loader.exec_module(pr_report)

Api = Callable[[str, str, dict[str, Any] | None], Any]

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4" time="12.5">
<testcase classname="tests.test_a" name="test_ok" time="0.10"/>
<testcase classname="tests.test_a" name="test_slow" time="9.00"/>
<testcase classname="tests.test_b" name="test_bad[rust]" time="1.00">
  <failure message="AssertionError: assert 1 == 2">long traceback</failure>
</testcase>
<testcase classname="tests.test_b" name="test_broken" time="0.20">
  <error message="">fixture exploded
second line</error>
</testcase>
<testcase classname="tests.test_c" name="test_skip" time="0.00">
  <skipped message="not fetched"/>
</testcase>
</testsuite></testsuites>
"""

PASSING = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="1" time="3.0">
<testcase classname="tests.test_live" name="test_live" time="2.5"/>
</testsuite>
"""


def _reports(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "reports"
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, "utf-8")
    return root


def test_totals_per_job_across_report_files(tmp_path: Path) -> None:
    root = _reports(
        tmp_path,
        {
            "test-3.11/junit.xml": JUNIT,
            # The Windows job writes two files: unit and integration.
            "test-windows/junit.xml": PASSING,
            "test-windows/integration.xml": PASSING,
        },
    )
    jobs = {job.name: job for job in pr_report.parse_reports(root)}

    linux = jobs["test-3.11"]
    assert (linux.passed, linux.failed, linux.errors, linux.skipped) == (
        2,
        1,
        1,
        1,
    )
    assert linux.seconds == pytest.approx(12.5)
    failure = next(c for c in linux.cases if c.outcome == "failed")
    assert failure.name == "tests.test_b::test_bad[rust]"
    assert failure.message == "AssertionError: assert 1 == 2"
    error = next(c for c in linux.cases if c.outcome == "error")
    assert error.message == "fixture exploded"  # empty attribute: use text

    windows = jobs["test-windows"]
    assert (windows.passed, windows.reports, windows.seconds) == (2, 2, 6.0)


def test_unreadable_report_counts_as_an_error(tmp_path: Path) -> None:
    root = _reports(tmp_path, {"integration/junit.xml": "<testsuite"})
    (job,) = pr_report.parse_reports(root)
    assert job.errors == 1
    assert job.cases[0].message.startswith("unreadable")


def test_missing_reports_folder_reports_nothing(tmp_path: Path) -> None:
    assert pr_report.parse_reports(tmp_path / "absent") == []


def test_render_passing_run(tmp_path: Path) -> None:
    root = _reports(tmp_path, {"test-3.12/junit.xml": PASSING})
    body = pr_report.render(
        pr_report.parse_reports(root),
        sha="9f3c2a1deadbeef",
        workflow_url="https://app.circleci.com/pipelines/workflows/w1",
        expected=["test-3.12"],
    )
    assert body.startswith(pr_report.MARKER)
    assert "### ✅ Tests · [`9f3c2a1`](https://app.circleci.com/" in body
    assert "| test-3.12 | 1 | 0 | 0 | 3s |" in body
    assert "Failures" not in body


def test_render_failures_and_missing_jobs(tmp_path: Path) -> None:
    root = _reports(tmp_path, {"test-3.11/junit.xml": JUNIT})
    body = pr_report.render(
        pr_report.parse_reports(root),
        sha="abc1234",
        expected=["test-3.11", "test-windows"],
    )
    assert body.splitlines()[1].startswith(
        "### ❌ Tests · `abc1234` · 2 passed"
    )
    assert "| test-3.11 ❌ | 2 | 2 | 1 | 12s |" in body
    assert "| test-windows ⚠️ | — | — | — | no results |" in body
    assert "<summary>Failures (2)</summary>" in body
    assert "`tests.test_b::test_bad[rust]` (test-3.11): AssertionError" in body
    assert "9.00s `tests.test_a::test_slow`" in body  # slowest first


def test_render_caps_failures_and_size(tmp_path: Path) -> None:
    cases = "".join(
        f'<testcase classname="t" name="test_{i}" time="0">'
        f'<failure message="{"x" * 400}"/></testcase>'
        for i in range(30)
    )
    root = _reports(
        tmp_path,
        {"job/junit.xml": f'<testsuite time="1">{cases}</testsuite>'},
    )
    body = pr_report.render(pr_report.parse_reports(root), sha="abc1234")
    assert f"… and {30 - pr_report.MAX_FAILURES} more" in body
    assert len(body) <= pr_report.MAX_BODY


def test_table_cells_escape_pipes(tmp_path: Path) -> None:
    root = _reports(
        tmp_path,
        {
            "job/junit.xml": '<testsuite time="1"><testcase classname="t" '
            'name="test_x" time="0"><failure message="a | b"/></testcase>'
            "</testsuite>"
        },
    )
    body = pr_report.render(pr_report.parse_reports(root), sha="abc1234")
    assert "a \\| b" in body


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/iamprakashom/bytely",
        "https://github.com/iamprakashom/bytely.git",
        "git@github.com:iamprakashom/bytely.git",
    ],
)
def test_parse_repo(url: str) -> None:
    assert pr_report.parse_repo(url) == ("iamprakashom", "bytely")


def test_parse_repo_rejects_other_hosts() -> None:
    with pytest.raises(ValueError, match="not a GitHub repository URL"):
        pr_report.parse_repo("https://gitlab.com/a/b")


def _summary(tmp_path: Path, name: str, files: dict[str, str]) -> Any:
    root = _reports(tmp_path / name, files)
    return pr_report.summarize(pr_report.parse_reports(root), f"{name}0000")


def test_summary_round_trips_through_a_file(tmp_path: Path) -> None:
    summary = _summary(tmp_path, "base", {"test-3.11/junit.xml": JUNIT})
    job = summary["jobs"]["test-3.11"]
    assert (job["passed"], job["failed"], job["errors"], job["skipped"]) == (
        2,
        1,
        1,
        1,
    )
    assert job["tests"] == sorted(job["tests"])
    assert "tests.test_b::test_bad[rust]" in job["tests"]

    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), "utf-8")
    assert pr_report.load_baseline(path) == summary


@pytest.mark.parametrize(
    "text",
    [None, "not json", '{"version": 99, "jobs": {}}', "[]"],
    ids=["missing", "not-json", "other-version", "not-an-object"],
)
def test_unusable_baseline_is_ignored(tmp_path: Path, text: str | None) -> None:
    path = tmp_path / "summary.json"
    if text is not None:
        path.write_text(text, "utf-8")
    assert pr_report.load_baseline(path) is None


def test_render_shows_changes_since_the_baseline(tmp_path: Path) -> None:
    baseline = _summary(tmp_path, "base", {"test-3.11/junit.xml": PASSING})
    root = _reports(tmp_path / "now", {"test-3.11/junit.xml": JUNIT})
    body = pr_report.render(
        pr_report.parse_reports(root), sha="abc1234", baseline=baseline
    )
    # Five tests are new and the one the baseline had is gone.
    assert (
        "Compared with `main` at `base000`: +4 tests (5 added, 1 removed)"
        in (body)
    )
    assert "| test-3.11 ❌ | 2 (+1) | 2 (+2) | 1 (+1) | 12s |" in body
    assert "<summary>Added tests (5)</summary>" in body
    assert "- `tests.test_b::test_bad[rust]`" in body
    assert "<summary>Removed tests (1)</summary>" in body
    assert "- `tests.test_live::test_live`" in body


def test_render_unchanged_run_and_new_job(tmp_path: Path) -> None:
    baseline = _summary(tmp_path, "base", {"test-3.11/junit.xml": PASSING})
    root = _reports(
        tmp_path / "now",
        {
            "test-3.11/junit.xml": PASSING,
            "integration-linux/junit.xml": PASSING,
        },
    )
    body = pr_report.render(
        pr_report.parse_reports(root), sha="abc1234", baseline=baseline
    )
    assert "no tests added or removed" in body
    assert "| test-3.11 | 1 | 0 | 0 | 3s |" in body
    # A job the baseline did not have shows plain counts, marked new.
    assert "| integration-linux (new) | 1 | 0 | 0 | 3s |" in body
    assert "Added tests" not in body
    assert "Removed tests" not in body


def test_render_says_when_no_baseline_exists(tmp_path: Path) -> None:
    root = _reports(tmp_path, {"test-3.11/junit.xml": PASSING})
    body = pr_report.render(
        pr_report.parse_reports(root), sha="abc1234", baseline_missing=True
    )
    assert "_No `main` baseline yet, so changes are not shown._" in body


def test_lists_of_tests_are_capped(tmp_path: Path) -> None:
    cases = "".join(
        f'<testcase classname="t" name="test_{i:03d}" time="0"/>'
        for i in range(pr_report.MAX_LISTED_TESTS + 7)
    )
    baseline = _summary(tmp_path, "base", {"job/junit.xml": PASSING})
    root = _reports(
        tmp_path / "now",
        {"job/junit.xml": f'<testsuite time="1">{cases}</testsuite>'},
    )
    body = pr_report.render(
        pr_report.parse_reports(root), sha="abc1234", baseline=baseline
    )
    assert "- … and 7 more" in body


def test_cli_writes_a_summary_and_compares_with_a_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _reports(tmp_path, {"test-3.13/junit.xml": PASSING})
    common = [
        str(root),
        "--repo-url",
        "https://github.com/iamprakashom/bytely",
        "--branch",
        "main",
        "--sha",
        "feed1234",
        "--dry-run",
    ]
    saved = tmp_path / "stats" / "main.json"
    assert pr_report.main([*common, "--write-summary", str(saved)]) == 0
    assert pr_report.load_baseline(saved) is not None

    missing = tmp_path / "absent.json"
    assert pr_report.main([*common, "--baseline", str(missing)]) == 0
    assert "No `main` baseline yet" in capsys.readouterr().out

    assert pr_report.main([*common, "--baseline", str(saved)]) == 0
    assert "Compared with `main` at `feed123`" in capsys.readouterr().out


class FakeGitHub:
    """Records calls and answers from canned pages of comments."""

    def __init__(self, pulls: list[dict[str, Any]], pages: list[list[Any]]):
        self.pulls = pulls
        self.pages = pages
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def __call__(
        self, method: str, path: str, body: dict[str, Any] | None
    ) -> Any:
        self.calls.append((method, path, body))
        if method == "GET" and "/pulls?" in path:
            return self.pulls
        if method == "GET" and "/comments?" in path:
            page = int(path.rsplit("page=", 1)[1])
            return self.pages[page - 1] if page <= len(self.pages) else []
        return {}


def _run(tmp_path: Path, api: Api, *extra: str) -> int:
    root = _reports(tmp_path, {"test-3.13/junit.xml": PASSING})
    return int(
        pr_report.main(
            [
                str(root),
                "--repo-url",
                "https://github.com/iamprakashom/bytely",
                "--branch",
                "feat/x",
                "--sha",
                "abc1234",
                *extra,
            ],
            api=api,
        )
    )


def test_creates_the_comment_when_none_is_marked(tmp_path: Path) -> None:
    api = FakeGitHub([{"number": 7}], [[{"id": 1, "body": "LGTM"}]])
    assert _run(tmp_path, api) == 0
    method, path, body = api.calls[-1]
    assert (method, path) == (
        "POST",
        "/repos/iamprakashom/bytely/issues/7/comments",
    )
    assert body is not None
    assert body["body"].startswith(pr_report.MARKER)
    # The branch is looked up as owner:branch, URL-encoded.
    assert "head=iamprakashom%3Afeat%2Fx" in api.calls[0][1]


def test_edits_the_marked_comment_on_a_later_page(tmp_path: Path) -> None:
    other = [{"id": n, "body": "chatter"} for n in range(100)]
    marked = [{"id": 555, "body": f"{pr_report.MARKER}\nold results"}]
    api = FakeGitHub([{"number": 7}], [other, marked])
    assert _run(tmp_path, api) == 0
    method, path, _ = api.calls[-1]
    assert (method, path) == (
        "PATCH",
        "/repos/iamprakashom/bytely/issues/comments/555",
    )
    assert not any(call[0] == "POST" for call in api.calls)


def test_no_open_pull_request_skips(tmp_path: Path) -> None:
    api = FakeGitHub([], [])
    assert _run(tmp_path, api) == 0
    assert all(call[0] == "GET" for call in api.calls)


def test_without_a_token_nothing_is_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    root = _reports(tmp_path, {"job/junit.xml": PASSING})
    code = pr_report.main(
        [
            str(root),
            "--repo-url",
            "https://github.com/iamprakashom/bytely",
            "--branch",
            "feat/x",
            "--sha",
            "abc1234",
        ]
    )
    assert code == 0


def test_dry_run_only_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = FakeGitHub([{"number": 7}], [])
    assert _run(tmp_path, api, "--dry-run") == 0
    assert api.calls == []
    assert pr_report.MARKER in capsys.readouterr().out


def test_github_error_fails_the_job(tmp_path: Path) -> None:
    def refusing(method: str, path: str, body: dict[str, Any] | None) -> Any:
        raise urllib.error.HTTPError(
            path, 403, "Forbidden", Message(), io.BytesIO(b'{"message":"no"}')
        )

    assert _run(tmp_path, refusing) == 1
