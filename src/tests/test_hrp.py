from __future__ import annotations

import numpy as np
import pandas as pd

from pyhrp.graph import dendrogram
from pyhrp.hrp import dist, hrp, linkage, tree


def test_dist():
    a = np.array([[1.0, 0.2 / np.sqrt(2.0)], [0.2 / np.sqrt(2.0), 1.0]])
    np.testing.assert_allclose(dist(a), np.array([6.552017e-01]), rtol=1e-6, atol=1e-6)


def test_quasi_diag(resource_dir):
    prices = pd.read_csv(resource_dir / "stock_prices.csv", parse_dates=True, index_col="date").truncate(
        before="2017-01-01"
    )

    # compute returns
    returns = prices.pct_change().dropna(axis=0, how="all").fillna(0.0)

    np.testing.assert_allclose(returns.cov().values, np.genfromtxt(resource_dir / "covariance2.csv"))
    np.testing.assert_allclose(returns.corr().values, np.genfromtxt(resource_dir / "correlation2.csv"))

    cor = returns.corr().values
    links = linkage(dist(cor), method="single")
    dendrogram(links=links)
    # uncomment this line if you want to generate a new test resource
    # np.savetxt(resource("links.csv"), links, delimiter=",")

    np.testing.assert_array_almost_equal(links, np.loadtxt(resource_dir / "links.csv", delimiter=","))

    node = tree(links)

    ids = node.pre_order()
    assert ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]

    ordered_tickers = prices.keys()[ids].to_list()
    print(ordered_tickers)
    assert ordered_tickers == [
        "UAA",
        "WMT",
        "SBUX",
        "AMD",
        "RRC",
        "GE",
        "T",
        "XOM",
        "BABA",
        "AAPL",
        "AMZN",
        "MA",
        "GOOG",
        "FB",
        "PFE",
        "GM",
        "BAC",
        "JPM",
        "SHLD",
        "BBY",
    ]


def test_hrp(resource_dir):
    prices = pd.read_csv(resource_dir / "stock_prices.csv", parse_dates=True, index_col="date").truncate(
        before="2017-01-01"
    )

    root = hrp(prices=prices, method="ward")

    x = pd.read_csv(resource_dir / "weights_hrp.csv", index_col=0, header=0).squeeze()

    x.index.name = None

    pd.testing.assert_series_equal(x, root.weights, check_exact=False)
