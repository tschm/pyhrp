import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyhrp.hrp import build_tree


def test_linkage(returns, resource_dir):
    dendrogram = build_tree(cor=returns.corr(), method="single", bisection=False)
    assert dendrogram.ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]

    np.testing.assert_array_almost_equal(dendrogram.linkage, np.loadtxt(resource_dir / "links.csv", delimiter=","))


def test_bisection(returns, resource_dir):
    dendrogram = build_tree(cor=returns.corr(), method="single", bisection=True)
    # The order doesn't change when using bisection
    assert dendrogram.ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]


def test_plot_bisection(returns):
    # compute covariance matrix and correlation matrices (both as DataFrames)
    cor = returns.corr()

    # you can either use a pre-computed node or you can construct a new dendrogram
    dendrogram = build_tree(cor=cor, method="single", bisection=True)
    print(dendrogram.root)
    print(dendrogram.linkage)
    # assert False

    dendrogram.plot()

    plt.show()

    assert dendrogram.names == [
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


@pytest.mark.parametrize("method", ["single", "ward", "average", "complete"])
def test_invariant_order(returns, method):
    cor = returns.corr()
    dendrogram1 = build_tree(cor=cor, method=method, bisection=True)
    dendrogram2 = build_tree(cor=cor, method=method, bisection=False)
    assert dendrogram1.assets == dendrogram2.assets == cor.columns.tolist()
    assert dendrogram1.ids == dendrogram2.ids
    assert dendrogram1.names == dendrogram2.names

    _, ax = plt.subplots()

    dendrogram2.plot(ax=ax)
    ax.set_title(method)
    plt.show()
