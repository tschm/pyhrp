"""Tests for the one_over_n algorithm with real market data."""

import pytest
from pandas import DataFrame

from pyhrp.algos import one_over_n
from pyhrp.cluster import Portfolio
from pyhrp.hrp import build_tree


def test_one_over_n_large(returns: DataFrame) -> None:
    """Test the one_over_n algorithm with real market data.

    This test verifies:
    1. The one_over_n algorithm works with larger, real-world datasets
    2. The number of portfolios matches the number of tree levels
    3. Each portfolio has weights that sum to 1.0
    4. All assets are included in each portfolio
    5. All weights are positive

    Args:
        returns: DataFrame of asset returns
    """
    # Build dendrogram from correlation matrix
    cor = returns.corr()
    dendrogram = build_tree(cor=cor, method="ward")

    # Collect portfolios from one_over_n algorithm
    portfolios: list[tuple[int, Portfolio]] = list(one_over_n(dendrogram))

    # Check that we get the expected number of levels
    assert len(portfolios) > 0
    assert len(portfolios) == len(dendrogram.root.levels)

    # Check properties of each portfolio
    for level, portfolio in portfolios:
        # Weights should sum to 1
        assert sum(portfolio.weights.values) == pytest.approx(1.0)

        # All assets should be in the portfolio
        assert set(portfolio.assets) == set(dendrogram.assets)

        # Each asset should have a positive weight
        for weight in portfolio.weights.values:
            assert weight > 0
