import numpy as np
import pandas as pd
from pypfopt import HRPOpt

from pyhrp.cluster import hrp


def test_allocation(prices):
    # reproduce the results of the implementation in PyPortfolioOpt
    returns = prices.pct_change().dropna(axis=0, how="all")

    optimizer = HRPOpt(returns)
    weights = optimizer.optimize(linkage_method="single")
    ww = pd.Series(weights)

    cluster = hrp(prices=prices, method="single", bisection=True)
    print(cluster.weights)

    assert np.linalg.norm(ww - cluster.weights) < 0.006
