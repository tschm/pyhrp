"""Tests for the GitHub Makefile targets using safe dry-runs.

These tests validate that the .github/github.mk targets are correctly exposed
and emit the expected commands without actually executing them.
"""

from __future__ import annotations

import pytest

# Import run_make from local conftest (setup_tmp_makefile is autouse)
from api.conftest import run_make

# Every GitHub helper target defined in github.mk; all must appear in `make help`.
_GH_TARGETS = (
    "gh-install",
    "view-prs",
    "view-issues",
    "failed-workflows",
    "whoami",
    "workflow-status",
    "latest-release",
)

# Map each target to a substring that must appear in its dry-run output, proving
# the recipe emits the intended command rather than merely parsing. Under `make -n`
# even @-prefixed recipe lines are printed, so the command text is observable.
_TARGET_COMMANDS = (
    ("gh-install", "command -v gh"),
    ("view-prs", "gh pr list"),
    ("view-issues", "gh issue list"),
    ("failed-workflows", "gh run list"),
    ("whoami", "gh auth status"),
    ("workflow-status", "gh workflow list"),
    ("latest-release", "gh release view"),
)


def test_gh_targets_exist(logger):
    """Verify that every GitHub target is listed in help."""
    result = run_make(logger, ["help"], dry_run=False)
    output = result.stdout

    for target in _GH_TARGETS:
        assert target in output, f"Target {target} not found in help output"


@pytest.mark.parametrize(("target", "expected_command"), _TARGET_COMMANDS)
def test_gh_target_emits_command(logger, target, expected_command):
    """Verify each GitHub target dry-runs cleanly and emits its expected command."""
    result = run_make(logger, [target])
    assert result.returncode == 0
    assert expected_command in result.stdout, (
        f"Target {target} did not emit expected command {expected_command!r}; got:\n{result.stdout}"
    )
