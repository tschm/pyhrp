import doctest
import io
import re
import sys

import pytest


@pytest.fixture()
def docstring():
    # Read the README.md file
    with open("../../README.md") as f:
        content = f.read()

    # Extract Python code blocks (assuming they are in triple backticks)
    blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)

    code = "\n".join(blocks).strip()

    # Add a docstring wrapper for doctest to process the code
    docstring = f"\n{code}\n"

    return docstring


def test_blocks2(docstring):
    # Create a StringIO buffer to capture doctest output
    captured_output = io.StringIO()
    sys.stdout = captured_output  # Redirect stdout to the StringIO buffer

    try:
        doctest.run_docstring_examples(docstring, globals())
    except doctest.DocTestFailure as e:
        # If a DocTestFailure occurs, capture it and manually fail the test
        pytest.fail(f"Doctests failed: {e}")
    finally:
        sys.stdout = sys.__stdout__  # Reset stdout to the original state

    # Check captured output for errors
    if captured_output.getvalue():
        pytest.fail(f"Doctests failed with the following output:\n{captured_output.getvalue()}")
