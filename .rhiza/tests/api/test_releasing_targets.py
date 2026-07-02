"""Tests for the releasing/versioning Makefile targets using safe dry-runs.

This file and its associated tests flow down via a SYNC action from the
jebel-quant/rhiza repository (https://github.com/jebel-quant/rhiza).

The release targets live in .rhiza/make.d/releasing.mk. These tests validate
that `changelog` and `release-status` resolve and emit the expected commands
without executing them, by invoking `make -n` (dry-run) via the shared
`run_make` helper. The autouse `setup_tmp_makefile` fixture copies releasing.mk
(and github.mk, which defines FORGE_TYPE) into an isolated temp dir.

The GitHub branch of `release-status` recurses via $(MAKE) and pipes to a
pager, both of which GNU make runs even under -n, so its recipe shape is
asserted by reading the source file rather than by dry-running it.
"""

from __future__ import annotations

from test_utils import run_make, strip_ansi


class TestReleasingTargets:
    """Smoke tests for the releasing.mk targets."""

    def test_changelog_target_dry_run(self, logger):
        """Changelog target should generate CHANGELOG.md via git-cliff in dry-run output."""
        proc = run_make(logger, ["changelog"])
        out = proc.stdout
        assert "no rule to make target" not in proc.stderr.lower()
        assert "git-cliff --output CHANGELOG.md" in out

    def test_release_status_resolves_without_forge(self, logger):
        """Release-status should resolve and report a missing forge when none is detected.

        The temp repo has neither .github/workflows/ nor .gitlab-ci.yml, so
        FORGE_TYPE is 'unknown' and the target prints the detection error while
        still exiting cleanly (it must never be an undefined target).
        """
        proc = run_make(logger, ["release-status"])
        assert "no rule to make target" not in proc.stderr.lower()
        assert proc.returncode == 0
        assert "Could not detect forge type" in strip_ansi(proc.stdout)

    def test_release_status_delegates_to_github_views(self, root):
        """On a GitHub forge, release-status should delegate to the workflow/release views."""
        content = (root / ".rhiza" / "make.d" / "releasing.mk").read_text()
        assert "ifeq ($(FORGE_TYPE),github)" in content, "release-status should branch on FORGE_TYPE"
        github_branch = content.split("ifeq ($(FORGE_TYPE),github)", 1)[1].split("else", 1)[0]
        assert "workflow-status" in github_branch, "GitHub branch should show workflow status"
        assert "latest-release" in github_branch, "GitHub branch should show the latest release"

    def test_release_status_warns_on_gitlab(self, root):
        """On a GitLab forge, release-status should warn that it is unsupported rather than fail."""
        content = (root / ".rhiza" / "make.d" / "releasing.mk").read_text()
        assert "else ifeq ($(FORGE_TYPE),gitlab)" in content
        gitlab_branch = content.split("else ifeq ($(FORGE_TYPE),gitlab)", 1)[1].split("else", 1)[0]
        assert "not yet supported for GitLab" in gitlab_branch


def test_release_targets_defined_in_make_d(root):
    """Guard that the changelog/release-status targets are sourced from releasing.mk."""
    releasing_mk = root / ".rhiza" / "make.d" / "releasing.mk"
    assert releasing_mk.exists(), "releasing.mk should exist in .rhiza/make.d"
    content = releasing_mk.read_text()
    for target in ("changelog:", "release-status:"):
        assert target in content, f"{target} should be defined in releasing.mk"
