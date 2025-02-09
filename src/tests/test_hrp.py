import numpy as np

from pyhrp.hrp import root


def test_quasi_diag(resource_dir, prices):
    # compute returns
    returns = prices.pct_change().dropna(axis=0, how="all").fillna(0.0)

    np.testing.assert_allclose(returns.cov().values, np.genfromtxt(resource_dir / "covariance2.csv"))
    np.testing.assert_allclose(returns.corr().values, np.genfromtxt(resource_dir / "correlation2.csv"))


def test_root(prices, resource_dir):
    returns = prices.pct_change().dropna(axis=0, how="all").fillna(0.0)

    dendrogram = root(cor=returns.corr().values, method="single", bisection=False)
    ids = dendrogram.root.pre_order()
    assert ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]

    np.testing.assert_array_almost_equal(dendrogram.linkage, np.loadtxt(resource_dir / "links.csv", delimiter=","))


def test_bisection(prices, resource_dir):
    returns = prices.pct_change().dropna(axis=0, how="all").fillna(0.0)

    dendrogram = root(cor=returns.corr().values, method="single", bisection=True)
    ids = dendrogram.root.pre_order()
    assert ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]


def test_graph(prices):
    returns = prices.pct_change().dropna(axis=0, how="all")

    # compute covariance matrix and correlation matrices (both as DataFrames)
    cor = returns.corr()

    # you can either use a pre-computed node or you can construct a new dendrogram
    dendrogram = root(cor.values, method="single", bisection=True)
    dendrogram.plot()
