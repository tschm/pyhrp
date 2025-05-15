import numpy as np
import pandas as pd
from pypfopt import HRPOpt

from pyhrp.hrp import hrp


def test_allocation(prices):
    """Test that our HRP implementation matches PyPortfolioOpt's implementation."""
    # Calculate returns from prices
    returns = prices.pct_change().dropna(axis=0, how="all")

    # Get weights from PyPortfolioOpt
    optimizer = HRPOpt(returns)
    weights = pd.Series(optimizer.optimize(linkage_method="single"))

    # Get weights from our implementation
    cluster = hrp(prices=prices, method="single", bisection=True)
    w = cluster.portfolio.weights

    # Verify that the weights are similar (within a small tolerance)
    assert np.linalg.norm(weights - w[weights.index]) < 0.006
