"""Tests for validating code examples in the project documentation.

This file is part of the tschm/.config-templates repository
(https://github.com/tschm/.config-templates).


This module contains tests that extract Python code blocks from the README.md file
and run them through doctest to ensure they are valid and working as expected.
This helps maintain accurate and working examples in the documentation.
"""

import doctest
import os
import re
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture


@pytest.fixture(scope="session", name="root_dir")
def root_fixture() -> Path:
    """Provide the path to the project root directory.

    This fixture returns the absolute path to the root directory of the project,
    which is useful for accessing files relative to the project root.

    Returns:
        Path: The absolute path to the project root directory

    """
    return Path(__file__).parent.parent.parent


@pytest.fixture()
def docstring(root_dir: Path) -> str:
    """Extract Python code blocks from README.md and prepare them for doctest.

    This fixture reads the README.md file, extracts all Python code blocks
    (enclosed in triple backticks with 'python' language identifier), and
    combines them into a single docstring that can be processed by doctest.

    Args:
        root_dir: Path to the project root directory

    Returns:
        str: A docstring containing all Python code examples from README.md

    """
    # Read the README.md file
    with open(root_dir / "README.md") as f:
        content = f.read()

    # Extract Python code blocks (assuming they are in triple backticks)
    blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)

    code = "\n".join(blocks).strip()

    # Add a docstring wrapper for doctest to process the code
    docstring = f"\n{code}\n"

    return docstring


def test_blocks(root_dir: Path, docstring: str, capfd: CaptureFixture[str]) -> None:
    """Test that all Python code blocks in README.md execute without errors.

    This test runs all the Python code examples from the README.md file
    through doctest to ensure they execute correctly. It captures any
    output or errors and fails the test if any issues are detected.

    Args:
        root_dir: Path to the project root directory
        docstring: String containing all Python code examples from README.md
        capfd: Pytest fixture for capturing stdout/stderr output

    Raises:
        pytest.fail: If any doctest fails or produces unexpected output

    """
    # Change to the root directory to ensure imports work correctly
    os.chdir(root_dir)

    try:
        # Run the code examples through doctest
        doctest.run_docstring_examples(docstring, globals())
    except doctest.DocTestFailure as e:
        # If a DocTestFailure occurs, capture it and manually fail the test
        pytest.fail(f"Doctests failed: {e}")

    # Capture the output after running doctests
    captured = capfd.readouterr()

    # If there is any output (error message), fail the test
    if captured.out:
        pytest.fail(f"Doctests failed with the following output:\n{captured.out} and \n{docstring}")
