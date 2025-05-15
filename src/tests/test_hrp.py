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
    """Test building a dendrogram with bisection and verify the node order."""
    # compute covariance matrix and correlation matrices (both as DataFrames)
    cor = returns.corr()

    # Construct a new dendrogram with bisection
    dendrogram = build_tree(cor=cor, method="single", bisection=True)

    # Verify the dendrogram has the expected structure
    assert dendrogram.root is not None
    assert dendrogram.linkage is not None

    # Verify the order of nodes in the dendrogram
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
    """Test that the order of nodes is invariant to the bisection parameter."""
    cor = returns.corr()
    dendrogram1 = build_tree(cor=cor, method=method, bisection=True)
    dendrogram2 = build_tree(cor=cor, method=method, bisection=False)

    # Verify that the assets, ids, and names are the same regardless of bisection
    assert dendrogram1.assets == dendrogram2.assets == cor.columns.tolist()
    assert dendrogram1.ids == dendrogram2.ids
    assert dendrogram1.names == dendrogram2.names
