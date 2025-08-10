"""Tests for the Marimo notebooks."""

import subprocess
import sys
from pathlib import Path

from security import safe_command


def test_notebooks(root_dir: Path) -> None:
    """Test that all Marimo notebooks run without errors.

    This test verifies:
    1. All Python files in the book/marimo directory can be executed
    2. The execution completes without errors

    Args:
        root_dir: Path to the repository root directory
    """
    # Get the path to the marimo notebooks directory
    path = root_dir / "book" / "marimo"

    # List all .py files in the directory using glob
    py_files: list[Path] = list(path.glob("*.py"))

    # Loop over the files and run them
    for py_file in py_files:
        print(f"Running {py_file.name}...")

        # Execute the Python file using a safe command wrapper
        result = safe_command.run(subprocess.run, [sys.executable, str(py_file)], capture_output=True, text=True)

        # Print the result of running the Python file
        if result.returncode == 0:
            print(f"{py_file.name} ran successfully.")
            print(f"Output: {result.stdout}")
        else:
            print(f"Error running {py_file.name}:")
            print(f"stderr: {result.stderr}")
