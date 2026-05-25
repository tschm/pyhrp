"""Tests for the rhiza_release.yml workflow configuration.

Validates that the release workflow is correctly defined and delegates
to the canonical reusable workflow from jebel-quant/rhiza.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = Path(".github") / "workflows" / "rhiza_release.yml"
REUSABLE_WORKFLOW = "jebel-quant/rhiza/.github/workflows/rhiza_release.yml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_workflow(root: Path) -> dict:
    """Load and parse the release workflow YAML file."""
    workflow_file = root / WORKFLOW_PATH
    if not workflow_file.exists():
        pytest.fail(f"Workflow file not found: {workflow_file}")
    with open(workflow_file) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Structure tests — validate the YAML content of rhiza_release.yml
# ---------------------------------------------------------------------------


class TestReleaseWorkflowStructure:
    """Validate the static content of rhiza_release.yml."""

    @pytest.fixture(scope="class")
    def workflow(self, root):
        """Load and return the parsed release workflow YAML."""
        return _load_workflow(root)

    def test_workflow_file_exists(self, root):
        """Workflow file must exist at the expected path."""
        assert (root / WORKFLOW_PATH).exists()

    def test_workflow_triggers_on_version_tags(self, workflow):
        """Workflow must trigger on version tags (v*)."""
        triggers = workflow.get("on") or workflow.get(True) or {}
        push = triggers.get("push", {})
        tags = push.get("tags", [])
        assert any("v*" in tag for tag in tags), "Workflow must trigger on v* tags"

    def test_workflow_has_contents_write_permission(self, workflow):
        """Workflow must have contents: write permission to push CHANGELOG.md."""
        permissions = workflow.get("permissions", {})
        assert permissions.get("contents") == "write", "Workflow must have contents: write permission"

    # --- reusable workflow delegation ---

    def test_single_release_job(self, workflow):
        """Workflow must define exactly one job named 'release'."""
        jobs = workflow.get("jobs", {})
        assert list(jobs.keys()) == ["release"], f"Expected ['release'], got: {list(jobs.keys())}"

    def test_release_job_uses_reusable_workflow(self, workflow):
        """Release job must delegate to the canonical rhiza reusable workflow."""
        job = workflow["jobs"]["release"]
        uses = job.get("uses", "")
        assert uses.startswith(REUSABLE_WORKFLOW), f"release job must use {REUSABLE_WORKFLOW}@<version>, got: {uses}"

    def test_release_job_inherits_secrets(self, workflow):
        """Release job must pass secrets via 'secrets: inherit'."""
        job = workflow["jobs"]["release"]
        assert job.get("secrets") == "inherit", "release job must set secrets: inherit"
