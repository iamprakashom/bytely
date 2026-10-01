"""Post a pipeline's test results to its pull request as one comment.

Usage:
    python .circleci/pr_report.py REPORTS_DIR --repo-url URL --branch NAME
        --sha SHA [--workflow-url URL] [--expect JOB,JOB,...]
        [--baseline FILE] [--base-name NAME] [--write-summary FILE]
        [--dry-run]

REPORTS_DIR holds one folder per CI job, named after the job, with the
JUnit XML files pytest wrote there. The report totals them per job and
overall, and lists failures and the slowest tests. The comment carries a
hidden marker, so each push edits the same comment instead of adding one.

`--write-summary` saves this run's counts and test names as JSON; CI keeps
the one from the latest `main` run. Given that file as `--baseline`, the
report shows each count's change and the tests added or removed since.

GitHub access comes from GITHUB_TOKEN. Without a token (pull requests
from forks get no secrets) or without an open pull request for the
branch, the report is printed and the script exits 0. It exits 1 only
when GitHub refuses or cannot be reached, so a broken setup shows up.
Standard library only, so it runs before any dependency is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MARKER = "<!-- bytely-ci-stats -->"
API_ROOT = "https://api.github.com"
MAX_FAILURES = 20
SLOWEST = 5
# GitHub rejects comments over 65,536 characters.
MAX_BODY = 60_000
MAX_COMMENT_PAGES = 20
MAX_LISTED_TESTS = 50
SUMMARY_VERSION = 1

Api = Callable[[str, str, dict[str, Any] | None], Any]


@dataclass
class Case:
    """One test case's outcome."""

    job: str
    name: str
    seconds: float
    outcome: str  # passed | failed | error | skipped
    message: str = ""


@dataclass
class JobStats:
    """Totals for one CI job's reports."""

    name: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    seconds: float = 0.0
    reports: int = 0
    cases: list[Case] = field(default_factory=list)

    @property
    def problems(self) -> int:
        """Failed and errored tests."""
        return self.failed + self.errors

    def add(self, case: Case) -> None:
        """Count one test case."""
        self.cases.append(case)
        if case.outcome == "failed":
            self.failed += 1
        elif case.outcome == "error":
            self.errors += 1
        elif case.outcome == "skipped":
            self.skipped += 1
        else:
            self.passed += 1


def _first_line(text: str | None) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _outcome(testcase: ET.Element) -> tuple[str, str]:
    for tag, outcome in (
        ("failure", "failed"),
        ("error", "error"),
        ("skipped", "skipped"),
    ):
        child = testcase.find(tag)
        if child is not None:
            message = child.get("message") or child.text
            return outcome, _first_line(message)
    return "passed", ""


def _case_name(testcase: ET.Element) -> str:
    classname, name = testcase.get("classname", ""), testcase.get("name", "")
    return f"{classname}::{name}" if classname else name


def parse_reports(root: Path) -> list[JobStats]:
    """Read every job folder's JUnit files under `root`."""
    jobs = []
    folders = (
        sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    )
    for folder in folders:
        stats = JobStats(folder.name)
        for report in sorted(folder.rglob("*.xml")):
            stats.reports += 1
            try:
                # Test output, possibly from a fork's tests. ElementTree never
                # fetches external entities, and expat >= 2.4 caps entity
                # expansion, so a hostile file can fail the parse but not
                # read files or exhaust memory.
                document = ET.parse(report).getroot()  # noqa: S314
            except ET.ParseError as error:
                stats.add(
                    Case(
                        stats.name,
                        report.name,
                        0.0,
                        "error",
                        f"unreadable: {error}",
                    )
                )
                continue
            suites = (
                [document]
                if document.tag == "testsuite"
                else list(document.iter("testsuite"))
            )
            for suite in suites:
                stats.seconds += float(suite.get("time") or 0)
                for testcase in suite.iter("testcase"):
                    outcome, message = _outcome(testcase)
                    stats.add(
                        Case(
                            stats.name,
                            _case_name(testcase),
                            float(testcase.get("time") or 0),
                            outcome,
                            message,
                        )
                    )
        jobs.append(stats)
    return jobs


def summarize(jobs: list[JobStats], sha: str) -> dict[str, Any]:
    """A run's counts and test names per job, for a later comparison."""
    return {
        "version": SUMMARY_VERSION,
        "sha": sha,
        "jobs": {
            job.name: {
                "passed": job.passed,
                "failed": job.failed,
                "errors": job.errors,
                "skipped": job.skipped,
                "tests": sorted({case.name for case in job.cases}),
            }
            for job in jobs
            if job.reports
        },
    }


def load_baseline(path: Path) -> dict[str, Any] | None:
    """A summary written by `--write-summary`, or None if unusable."""
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(data, dict)
        or data.get("version") != SUMMARY_VERSION
        or not isinstance(data.get("jobs"), dict)
    ):
        return None
    return data


def _delta(value: int, before: int | None) -> str:
    """`value`, followed by its change when there is one: `270 (+1)`."""
    if before is None or value == before:
        return f"{value:,}"
    return f"{value:,} ({value - before:+,})"


def _base_problems(base: dict[str, Any]) -> int | None:
    if "failed" not in base:
        return None
    return int(base["failed"]) + int(base.get("errors", 0))


def _test_names(jobs: dict[str, Any]) -> set[str]:
    return {name for job in jobs.values() for name in job.get("tests", [])}


def _listed(title: str, names: list[str]) -> list[str]:
    lines = ["", f"<details><summary>{title} ({len(names)})</summary>", ""]
    lines += [f"- `{name}`" for name in names[:MAX_LISTED_TESTS]]
    if len(names) > MAX_LISTED_TESTS:
        lines.append(f"- … and {len(names) - MAX_LISTED_TESTS} more")
    return [*lines, "", "</details>"]


def _duration(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole // 60}m {whole % 60:02d}s" if whole >= 60 else f"{whole}s"


def _cell(text: str) -> str:
    """Make text safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def render(
    jobs: list[JobStats],
    *,
    sha: str,
    workflow_url: str | None = None,
    expected: list[str] | None = None,
    baseline: dict[str, Any] | None = None,
    base_name: str = "main",
    baseline_missing: bool = False,
) -> str:
    """The comment body: totals, per-job table, failures, slowest tests.

    With a `baseline` summary, each count shows its change and the tests
    added or removed since are listed; `baseline_missing` says that a
    comparison was asked for but no baseline was available.
    """
    present = {job.name for job in jobs if job.reports}
    missing = [name for name in expected or [] if name not in present]
    passed = sum(job.passed for job in jobs)
    problems = sum(job.problems for job in jobs)
    skipped = sum(job.skipped for job in jobs)
    ok = problems == 0 and not missing
    icon = "✅" if ok else "❌"
    short = sha[:7]
    commit = f"[`{short}`]({workflow_url})" if workflow_url else f"`{short}`"

    lines = [
        MARKER,
        (
            f"### {icon} Tests · {commit} · {passed:,} passed · "
            f"{problems:,} failed · {skipped:,} skipped"
        ),
    ]
    base_jobs: dict[str, Any] = baseline["jobs"] if baseline else {}
    added: list[str] = []
    removed: list[str] = []
    if baseline:
        now = {case.name for job in jobs for case in job.cases}
        before = _test_names(base_jobs)
        added, removed = sorted(now - before), sorted(before - now)
        base_sha = str(baseline.get("sha", ""))[:7]
        change = (
            f"{len(now) - len(before):+,} tests ({len(added):,} added, "
            f"{len(removed):,} removed)"
            if added or removed
            else "no tests added or removed"
        )
        lines.append(f"Compared with `{base_name}` at `{base_sha}`: {change}")
    elif baseline_missing:
        lines.append(
            f"_No `{base_name}` baseline yet, so changes are not shown._"
        )
    lines += [
        "",
        "| Job | Passed | Failed | Skipped | Time |",
        "|---|--:|--:|--:|--:|",
    ]
    for job in jobs:
        if not job.reports:
            continue
        mark = " ❌" if job.problems else ""
        base = base_jobs.get(job.name)
        if baseline and base is None:
            mark += " (new)"
        base = base or {}
        lines.append(
            f"| {_cell(job.name)}{mark} "
            f"| {_delta(job.passed, base.get('passed'))} "
            f"| {_delta(job.problems, _base_problems(base))} "
            f"| {_delta(job.skipped, base.get('skipped'))} "
            f"| {_duration(job.seconds)} |"
        )
    lines.extend(
        f"| {_cell(name)} ⚠️ | — | — | — | no results |" for name in missing
    )

    failures = [
        case
        for job in jobs
        for case in job.cases
        if case.outcome in ("failed", "error")
    ]
    if failures:
        lines += [
            "",
            f"<details open><summary>Failures ({len(failures)})</summary>",
            "",
        ]
        for case in failures[:MAX_FAILURES]:
            detail = f": {_cell(case.message)[:300]}" if case.message else ""
            lines.append(f"- `{case.name}` ({case.job}){detail}")
        if len(failures) > MAX_FAILURES:
            lines.append(f"- … and {len(failures) - MAX_FAILURES} more")
        lines += ["", "</details>"]
    if missing:
        lines += [
            "",
            (
                f"⚠️ No test results from: {', '.join(missing)} "
                "(the job failed before tests ran, or was canceled)."
            ),
        ]

    if added:
        lines += _listed("Added tests", added)
    if removed:
        lines += _listed("Removed tests", removed)

    timed = sorted(
        (
            case
            for job in jobs
            for case in job.cases
            if case.outcome != "skipped"
        ),
        key=lambda case: case.seconds,
        reverse=True,
    )[:SLOWEST]
    if timed:
        lines += ["", "<details><summary>Slowest tests</summary>", ""]
        lines += [
            f"- {case.seconds:.2f}s `{case.name}` ({case.job})"
            for case in timed
        ]
        lines += ["", "</details>"]

    lines += ["", "<sub>Updated by CircleCI on every push.</sub>"]
    body = "\n".join(lines) + "\n"
    if len(body) > MAX_BODY:
        body = body[: MAX_BODY - 40] + "\n\n…(truncated)\n"
    return body


def parse_repo(url: str) -> tuple[str, str]:
    """Owner and name from a GitHub https or ssh URL."""
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if match is None:
        raise ValueError(f"not a GitHub repository URL: {url}")
    return match.group(1), match.group(2)


def github_api(token: str) -> Api:
    """A function that calls GitHub's REST API with `token`."""

    def call(method: str, path: str, body: dict[str, Any] | None) -> Any:
        # Always https: the URL is API_ROOT plus a path built here.
        request = urllib.request.Request(  # noqa: S310
            API_ROOT + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "bytely-ci-report",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            payload = response.read()
        return json.loads(payload) if payload else None

    return call


def find_pull_request(
    api: Api, owner: str, repo: str, branch: str
) -> int | None:
    """The open pull request whose head is `branch` in this repository."""
    head = urllib.parse.quote(f"{owner}:{branch}", safe="")
    pulls = api(
        "GET", f"/repos/{owner}/{repo}/pulls?head={head}&state=open", None
    )
    return int(pulls[0]["number"]) if pulls else None


def upsert_comment(
    api: Api, owner: str, repo: str, number: int, body: str
) -> str:
    """Edit the marked comment on the pull request, or create it."""
    for page in range(1, MAX_COMMENT_PAGES + 1):
        comments = api(
            "GET",
            f"/repos/{owner}/{repo}/issues/{number}/comments"
            f"?per_page=100&page={page}",
            None,
        )
        if not comments:
            break
        for comment in comments:
            if MARKER in (comment.get("body") or ""):
                api(
                    "PATCH",
                    f"/repos/{owner}/{repo}/issues/comments/{comment['id']}",
                    {"body": body},
                )
                return "updated"
    api(
        "POST",
        f"/repos/{owner}/{repo}/issues/{number}/comments",
        {"body": body},
    )
    return "created"


def main(argv: list[str] | None = None, api: Api | None = None) -> int:
    """Render the report and post it; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", type=Path)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--workflow-url")
    parser.add_argument(
        "--expect",
        default="",
        help="comma-separated jobs that must have reported results",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="a summary to compare with (from --write-summary on main)",
    )
    parser.add_argument(
        "--base-name", default="main", help="what the baseline is, for text"
    )
    parser.add_argument(
        "--write-summary", type=Path, help="save this run's summary here"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    expected = [name.strip() for name in args.expect.split(",") if name.strip()]
    jobs = parse_reports(args.reports)
    if args.write_summary:
        args.write_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary.write_text(
            json.dumps(summarize(jobs, args.sha)), encoding="utf-8"
        )
    baseline = load_baseline(args.baseline) if args.baseline else None
    body = render(
        jobs,
        sha=args.sha,
        workflow_url=args.workflow_url,
        expected=expected,
        baseline=baseline,
        base_name=args.base_name,
        baseline_missing=args.baseline is not None and baseline is None,
    )
    print(body)
    if args.dry_run:
        return 0

    if api is None:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            print("No GITHUB_TOKEN (a fork's pull request?); not commenting.")
            return 0
        api = github_api(token)
    owner, repo = parse_repo(args.repo_url)
    try:
        number = find_pull_request(api, owner, repo, args.branch)
        if number is None:
            print(f"No open pull request for {args.branch}; not commenting.")
            return 0
        action = upsert_comment(api, owner, repo, number, body)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        print(f"GitHub API error {error.code} {error.reason}: {detail}")
        return 1
    except urllib.error.URLError as error:
        print(f"Could not reach GitHub: {error.reason}")
        return 1
    print(f"Comment {action} on pull request #{number}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
