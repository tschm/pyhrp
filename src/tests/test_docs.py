"""Tests for the README.md file."""

import doctest
import os
import re
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture


@pytest.fixture()
def docstring(root_dir: Path) -> str:
    """Extract Python code blocks from README.md and prepare them for doctest.

    This fixture:
    1. Reads the README.md file
    2. Extracts all Python code blocks (enclosed in ```python ... ```)
    3. Joins them into a single string
    4. Formats it as a docstring for doctest to process

    Args:
        root_dir: Path to the repository root directory

    Returns:
        str: A string containing all Python code blocks from README.md
    """
    # Read the README.md file
    with open(root_dir / "README.md") as f:
        content = f.read()

    # Extract Python code blocks (assuming they are in triple backticks)
    blocks: list[str] = re.findall(r"```python(.*?)```", content, re.DOTALL)

    # Join all code blocks into a single string
    code = "\n".join(blocks).strip()

    # Add a docstring wrapper for doctest to process the code
    docstring = f"\n{code}\n"

    return docstring


def test_blocks(root_dir: Path, docstring: str, capfd: CaptureFixture) -> None:
    """Test that Python code blocks in README.md run without errors.

    This test verifies:
    1. All Python code examples in the README.md can be executed
    2. The execution completes without errors

    Args:
        root_dir: Path to the repository root directory
        docstring: String containing Python code blocks from README.md
        capfd: Pytest fixture to capture stdout/stderr
    """
    # Change to the root directory to ensure imports work correctly
    os.chdir(root_dir)

    try:
        # Run the code blocks as doctests
        doctest.run_docstring_examples(docstring, globals())
    except doctest.DocTestFailure as e:
        # If a DocTestFailure occurs, capture it and manually fail the test
        pytest.fail(f"Doctests failed: {e}")

    # Capture the output after running doctests
    captured = capfd.readouterr()

    # If there is any output (error message), fail the test
    if captured.out:
        pytest.fail(f"Doctests failed with the following output:\n{captured.out} and \n{docstring}")
