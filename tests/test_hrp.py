"""Tests for the hierarchical clustering and dendrogram building functionality."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from pyhrp.hrp import build_tree


def test_linkage(returns: DataFrame, resource_dir: Path) -> None:
    """Test the linkage matrix generation in the build_tree function.

    This test verifies:
    1. The correct order of node IDs in the dendrogram
    2. The linkage matrix matches the expected values from a reference file

    Args:
        returns: DataFrame of asset returns
        resource_dir: Path to test resources directory
    """
    # Build dendrogram without bisection
    dendrogram = build_tree(cor=returns.corr(), method="single", bisection=False)

    # Verify the order of node IDs
    assert dendrogram.ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]

    # Verify the linkage matrix against reference data
    np.testing.assert_array_almost_equal(dendrogram.linkage, np.loadtxt(resource_dir / "links.csv", delimiter=","))


def test_bisection(returns: DataFrame, resource_dir: Path) -> None:
    """Test the bisection method in the build_tree function.

    This test verifies that the order of node IDs remains consistent
    when using the bisection method.

    Args:
        returns: DataFrame of asset returns
        resource_dir: Path to test resources directory
    """
    # Build dendrogram with bisection
    dendrogram = build_tree(cor=returns.corr(), method="single", bisection=True)

    # The order doesn't change when using bisection
    assert dendrogram.ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]


def test_plot_bisection(returns: DataFrame) -> None:
    """Test building a dendrogram with bisection and verify the node order.

    This test verifies:
    1. The dendrogram is properly constructed with bisection
    2. The dendrogram has the expected structure (root and linkage)
    3. The order of asset names in the dendrogram matches the expected order

    Args:
        returns: DataFrame of asset returns
    """
    # Compute correlation matrix
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
def test_invariant_order(returns: DataFrame, method: str) -> None:
    """Test that the order of nodes is invariant to the bisection parameter.

    This test verifies that regardless of whether bisection is used or not,
    the resulting dendrogram maintains the same order of assets, IDs, and names
    for different clustering methods.

    Args:
        returns: DataFrame of asset returns
        method: Clustering method to use (single, ward, average, or complete)
    """
    # Compute correlation matrix
    cor = returns.corr()

    # Build dendrograms with and without bisection
    dendrogram1 = build_tree(cor=cor, method=method, bisection=True)
    dendrogram2 = build_tree(cor=cor, method=method, bisection=False)

    pd.testing.assert_index_equal(dendrogram1.assets, dendrogram2.assets)

    # assert dendrogram1.assets == dendrogram2.assets == cor.columns.tolist()
    assert dendrogram1.ids == dendrogram2.ids
    assert dendrogram1.names == dendrogram2.names
