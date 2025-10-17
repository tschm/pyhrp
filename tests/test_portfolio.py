"""Tests for the Portfolio class."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from pyhrp.cluster import Portfolio


def test_portfolio() -> None:
    """Test portfolio creation, plotting, and variance calculation.

    This test verifies:
    1. Portfolio creation and weight assignment
    2. Portfolio plotting functionality
    3. Portfolio variance calculation with a covariance matrix
    """
    # Create a portfolio with three assets
    p = Portfolio()
    a = "A"
    b = "B"
    c = "C"

    # Assign weights
    p[a] = 0.4
    p[b] = 0.3
    p[c] = 0.3

    # Create a diagonal covariance matrix
    cov = pd.DataFrame(index=[a, b, c], columns=[a, b, c], data=np.diag([2, 3, 4]))

    # Test plotting (but don't show the plot in tests)
    ax: Axes = p.plot(names=["A", "B", "C"])
    assert ax is not None

    # Test variance calculation
    # Expected variance: 0.4^2 * 2 + 0.3^2 * 3 + 0.3^2 * 4 = 0.95
    assert p.variance(cov) == pytest.approx(0.95)


def test_getset_item() -> None:
    """Test the __getitem__ and __setitem__ methods of Portfolio class.

    This test verifies:
    1. Setting a weight for an asset in a portfolio
    2. Retrieving the weight for an asset from a portfolio
    """
    # Create a portfolio and an asset
    p = Portfolio()
    a = "A"

    # Set and verify weight
    p[a] = 0.4
    assert p[a] == 0.4
