"""TEMPORARY: fails on purpose to check the CI failure report. Do not merge."""


def test_ci_failure_report_lists_this_test() -> None:
    assert 1 == 2, "planted failure: the PR comment should list this test"
