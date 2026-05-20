"""Tests for the one_over_n algorithm with real market data."""

import pytest
from polars import DataFrame

from pyhrp.algos import one_over_n
from pyhrp.cluster import Portfolio
from pyhrp.hrp import _compute_corr, build_tree


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
    cor = _compute_corr(returns)
    dendrogram = build_tree(cor=cor, method="ward")

    portfolios: list[tuple[int, Portfolio]] = list(one_over_n(dendrogram))

    assert len(portfolios) > 0
    assert len(portfolios) == len(dendrogram.root.levels)

    for _level, portfolio in portfolios:
        assert sum(portfolio.weights_dict.values()) == pytest.approx(1.0)
        assert set(portfolio.assets) == set(dendrogram.assets)
        for weight in portfolio.weights_dict.values():
            assert weight > 0
