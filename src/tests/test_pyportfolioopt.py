import numpy as np
import pandas as pd
from pypfopt import HRPOpt

from pyhrp.hrp import hrp


def test_allocation(prices):
    # reproduce the results of the implementation in PyPortfolioOpt
    returns = prices.pct_change().dropna(axis=0, how="all")

    optimizer = HRPOpt(returns)
    weights = pd.Series(optimizer.optimize(linkage_method="single"))
    print(weights)

    cluster = hrp(prices=prices, method="single", bisection=True)
    w = cluster.portfolio.weights
    print(w)

    assert np.linalg.norm(weights - w[weights.index]) < 0.006
