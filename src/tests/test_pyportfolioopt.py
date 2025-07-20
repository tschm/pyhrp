"""Tests comparing our HRP implementation with PyPortfolioOpt's implementation."""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pypfopt import HRPOpt

from pyhrp.hrp import hrp


def test_allocation(prices: DataFrame) -> None:
    """Test that our HRP implementation matches PyPortfolioOpt's implementation.

    This test verifies:
    1. Our HRP implementation produces similar results to PyPortfolioOpt's HRP
    2. The difference in portfolio weights is within an acceptable tolerance

    Args:
        prices: DataFrame of asset prices
    """
    # Calculate returns from prices
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Get weights from PyPortfolioOpt
    optimizer = HRPOpt(returns)
    weights: Series = pd.Series(optimizer.optimize(linkage_method="single"))

    # Get weights from our implementation
    cluster = hrp(prices=prices, method="single", bisection=True)
    w = cluster.portfolio.weights

    # Verify that the weights are similar (within a small tolerance)
    # Using L2 norm (Euclidean distance) to measure the difference
    assert np.linalg.norm(weights - w[weights.index]) < 0.006
