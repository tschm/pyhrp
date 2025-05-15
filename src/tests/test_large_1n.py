import pytest

from pyhrp.algos import one_over_n
from pyhrp.hrp import build_tree


def test_one_over_n_large(returns):
    """Test the one_over_n algorithm with real market data."""
    # Build dendrogram from correlation matrix
    cor = returns.corr()
    dendrogram = build_tree(cor=cor, method="ward")

    # Collect portfolios from one_over_n algorithm
    portfolios = list(one_over_n(dendrogram))

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
