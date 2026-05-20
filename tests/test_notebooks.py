"""Tests for the Marimo notebooks."""

import runpy
from pathlib import Path


def test_notebooks(root_dir: Path) -> None:
    """Test that all Marimo notebooks load without errors."""
    for py_file in (root_dir / "book" / "marimo").glob("*.py"):
        runpy.run_path(str(py_file))
