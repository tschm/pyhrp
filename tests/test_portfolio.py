"""Tests for the Portfolio class."""

import plotly.graph_objects as go
import polars as pl
import pytest

from pyhrp.cluster import Portfolio


def test_portfolio() -> None:
    """Test portfolio creation, plotting, and variance calculation.

    This test verifies:
    1. Portfolio creation and weight assignment
    2. Portfolio plotting functionality
    3. Portfolio variance calculation with a covariance matrix
    """
    p = Portfolio()
    a = "A"
    b = "B"
    c = "C"

    p[a] = 0.4
    p[b] = 0.3
    p[c] = 0.3

    # Diagonal covariance matrix (columns: A, B, C — rows in same order)
    cov = pl.DataFrame({"A": [2.0, 0.0, 0.0], "B": [0.0, 3.0, 0.0], "C": [0.0, 0.0, 4.0]})

    fig: go.Figure = p.plot(names=["A", "B", "C"])
    assert fig is not None

    # Expected variance: 0.4^2 * 2 + 0.3^2 * 3 + 0.3^2 * 4 = 0.95
    assert p.variance(cov) == pytest.approx(0.95)


def test_getset_item() -> None:
    """Test the __getitem__ and __setitem__ methods of Portfolio class."""
    p = Portfolio()
    a = "A"

    p[a] = 0.4
    assert p[a] == 0.4
