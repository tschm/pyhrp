import doctest
import re
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture()
def docstring(resource_dir):
    # Read the README.md file
    with open(resource_dir.parent.parent.parent / "README.md") as f:
        content = f.read()

    # Extract Python code blocks (assuming they are in triple backticks)
    blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)

    code = "\n".join(blocks).strip()

    # Add a docstring wrapper for doctest to process the code
    docstring = f"\n{code}\n"

    return docstring


# Test where we mock pd.read_csv
@pytest.fixture
def mock_read_csv(resource_dir):
    # Mocked DataFrame that you control
    mock_df = pd.read_csv(resource_dir / "stock_prices.csv", index_col=0, parse_dates=True)

    with patch("pandas.read_csv", return_value=mock_df) as mock:
        yield mock


def test_blocks(docstring, capfd, mock_read_csv):
    try:
        doctest.run_docstring_examples(docstring, globals())
    except doctest.DocTestFailure as e:
        # If a DocTestFailure occurs, capture it and manually fail the test
        pytest.fail(f"Doctests failed: {e}")

    # Capture the output after running doctests
    captured = capfd.readouterr()

    # If there is any output (error message), fail the test
    if captured.out:
        pytest.fail(f"Doctests failed with the following output:\n{captured.out} and \n{docstring}")
