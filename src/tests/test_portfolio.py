import numpy as np
import pandas as pd
import pytest

from pyhrp.cluster import Asset, Portfolio


def test_portfolio():
    """Test portfolio creation, plotting, and variance calculation."""
    # Create a portfolio with three assets
    p = Portfolio()
    a = Asset(name="A")
    b = Asset(name="B")
    c = Asset(name="C")

    # Assign weights
    p[a] = 0.4
    p[b] = 0.3
    p[c] = 0.3

    # Create a diagonal covariance matrix
    cov = pd.DataFrame(index=[a, b, c], columns=[a, b, c], data=np.diag([2, 3, 4]))

    # Test plotting (but don't show the plot in tests)
    ax = p.plot(names=["A", "B", "C"])
    assert ax is not None

    # Test variance calculation
    assert p.variance(cov) == pytest.approx(0.95)


def test_getset_item():
    p = Portfolio()
    a = Asset(name="A")
    p[a] = 0.4
    assert p[a] == 0.4


def test_asset_equality():
    a = Asset(name="A")
    b = Asset(name="A")
    assert a == b
    assert not a < 2
