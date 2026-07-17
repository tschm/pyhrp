"""Tests for the portfolio optimization algorithms.

Covers risk_parity, schur_risk_parity, one_over_n and the _solve helper,
including property-based and numerical edge-case checks.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from polars import DataFrame

from pyhrp.algos import _solve, one_over_n, risk_parity, schur_risk_parity
from pyhrp.cluster import Cluster, Portfolio
from pyhrp.covariance import compute_corr, compute_cov
from pyhrp.dendrogram import Dendrogram, build_tree
from pyhrp.treelib import Node


@st.composite
def covariance_matrices(draw: st.DrawFn) -> pl.DataFrame:
    """Generate symmetric positive-definite covariance matrices."""
    n_assets = draw(st.integers(min_value=2, max_value=8))
    base = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_assets, n_assets),
            elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        )
    )
    cov = base @ base.T + np.eye(n_assets) * 1e-6
    assets = [f"A{i}" for i in range(n_assets)]
    return pl.DataFrame(dict(zip(assets, cov, strict=True)))


def test_riskparity() -> None:
    """Test the risk parity algorithm implementation.

    This test verifies:
    1. Creation of Asset and Cluster objects
    2. Portfolio assignment to clusters
    3. Risk parity calculation with a simple covariance matrix
    4. Resulting portfolio weights and variance calculation
    """
    # Create left cluster with asset A
    left = Cluster(value=1)
    left.portfolio["A"] = 1.0

    # Create right cluster with asset B
    right = Cluster(value=0)
    right.portfolio["B"] = 1.0

    # Covariance matrix: cov(B,B)=2, cov(A,A)=4, cov(A,B)=cov(B,A)=1
    # Columns: B, A — row 0 = B's row, row 1 = A's row
    cov = pl.DataFrame({"B": [2.0, 1.0], "A": [1.0, 4.0]})

    # Create parent cluster
    cl = Cluster(value=2, left=left, right=right)

    # Apply risk parity algorithm
    cluster = risk_parity(cl, cov=cov)

    # Verify the resulting portfolio weights (alphabetically sorted: A, B)
    # Expected weights: [1/3, 2/3]
    np.testing.assert_allclose(
        np.array(list(cluster.portfolio.weights.values())),
        np.array([1.0, 2.0]) / 3.0,
    )

    # Verify the resulting portfolio variance
    np.testing.assert_almost_equal(cluster.portfolio.variance(cov), 1.7777777777777777)


def test_risk_parity_non_cluster_left() -> None:
    """TypeError is raised when risk_parity encounters a non-Cluster left child."""
    root = Cluster(value=10)
    root.left = Node(1)
    root.right = Cluster(value=2)
    cov = pl.DataFrame({"A": [1.0]})
    with pytest.raises(TypeError, match="Expected left child to be a Cluster"):
        risk_parity(root, cov)


def test_risk_parity_non_cluster_right() -> None:
    """TypeError is raised when risk_parity encounters a non-Cluster right child."""
    root = Cluster(value=10)
    root.left = Cluster(value=1)
    root.right = Node(2)
    cov = pl.DataFrame({"A": [1.0]})
    with pytest.raises(TypeError, match="Expected right child to be a Cluster"):
        risk_parity(root, cov)


def test_risk_parity_idempotent() -> None:
    """Repeated allocation on the same tree yields identical weights."""
    cov = pl.DataFrame({"A": [4.0, 0.0], "B": [0.0, 1.0]})
    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    first = dict(risk_parity(root=root, cov=cov).portfolio.weights)
    second = dict(risk_parity(root=root, cov=cov).portfolio.weights)
    assert first == second


def test_risk_parity_zero_variance_split() -> None:
    """Two riskless children split the weight equally instead of producing NaNs."""
    cov = pl.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})
    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    weights = risk_parity(root=root, cov=cov).portfolio.weights
    assert weights == {"A": 0.5, "B": 0.5}


def test_gamma_zero_matches_hrp(prices: DataFrame) -> None:
    """At gamma=0, Schur allocation must produce the same weights as standard risk parity."""
    cov = compute_cov(
        prices.select(pl.all().pct_change())
        .filter(pl.any_horizontal(pl.all().is_not_null()))
        .fill_null(0.0)
        .fill_nan(0.0)
    )
    cor = compute_corr(
        prices.select(pl.all().pct_change())
        .filter(pl.any_horizontal(pl.all().is_not_null()))
        .fill_null(0.0)
        .fill_nan(0.0)
    )
    root_hrp = build_tree(cor, method="ward").root
    root_schur = build_tree(cor, method="ward").root

    hrp_cluster = risk_parity(root=root_hrp, cov=cov)
    schur_cluster = schur_risk_parity(root=root_schur, cov=cov, gamma=0.0)

    for asset in hrp_cluster.portfolio.assets:
        assert hrp_cluster.portfolio[asset] == pytest.approx(schur_cluster.portfolio[asset], rel=1e-6)


def test_schur_two_assets() -> None:
    """Verify Schur complement on a 2-asset portfolio reduces to known formula."""
    # cov = [[4, 1], [1, 1]] — a=4, d=1, b=1
    # a_aug = 4 - gamma * 1 * 1/1 * 1 = 4 - gamma
    # d_aug = 1 - gamma * 1 * 1/4 * 1 = 1 - gamma/4
    cov = pl.DataFrame({"A": [4.0, 1.0], "B": [1.0, 1.0]})
    gamma = 0.8

    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    cluster = schur_risk_parity(root=root, cov=cov, gamma=gamma)

    a_aug = 4.0 - gamma * 1.0
    d_aug = 1.0 - gamma * 0.25
    alpha_left = d_aug / (a_aug + d_aug)
    alpha_right = a_aug / (a_aug + d_aug)

    assert cluster.portfolio["A"] == pytest.approx(alpha_left, rel=1e-10)
    assert cluster.portfolio["B"] == pytest.approx(alpha_right, rel=1e-10)


def test_schur_risk_parity_raises_for_non_cluster_left() -> None:
    """schur_risk_parity raises TypeError when left child is not a Cluster."""
    cov = pl.DataFrame({"A": [4.0, 0.0], "B": [0.0, 1.0]})
    root = Cluster(2)
    root.left = Node(0)  # type: ignore[assignment]
    root.right = Cluster(1)
    with pytest.raises(TypeError, match="Expected left child to be a Cluster"):
        schur_risk_parity(root=root, cov=cov, gamma=0.5)


def test_schur_risk_parity_raises_for_non_cluster_right() -> None:
    """schur_risk_parity raises TypeError when right child is not a Cluster."""
    cov = pl.DataFrame({"A": [4.0, 0.0], "B": [0.0, 1.0]})
    left = Cluster(0)
    root = Cluster(2)
    root.left = left
    root.right = Node(1)  # type: ignore[assignment]
    with pytest.raises(TypeError, match="Expected right child to be a Cluster"):
        schur_risk_parity(root=root, cov=cov, gamma=0.5)


@pytest.mark.parametrize("gamma", [-0.1, 1.5, 5.0])
def test_gamma_out_of_range_raises(gamma: float) -> None:
    """schur_risk_parity rejects gamma outside [0, 1]."""
    cov = pl.DataFrame({"A": [4.0, 0.0], "B": [0.0, 1.0]})
    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    with pytest.raises(ValueError, match="gamma must be in"):
        schur_risk_parity(root=root, cov=cov, gamma=gamma)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_gamma_boundaries_accepted(gamma: float) -> None:
    """schur_risk_parity accepts the boundary values 0 and 1."""
    cov = pl.DataFrame({"A": [4.0, 0.0], "B": [0.0, 1.0]})
    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    cluster = schur_risk_parity(root=root, cov=cov, gamma=gamma)
    assert sum(cluster.portfolio.weights.values()) == pytest.approx(1.0)


def test_one_over_n() -> None:
    """Test the one_over_n algorithm with a simple tree structure.

    This test verifies:
    1. The one_over_n algorithm correctly generates portfolios for each level
    2. The number of portfolios matches the number of tree levels
    3. Portfolio weights sum to 1.0 at each level
    4. All assets are included in the portfolio
    """
    # Create a simple tree structure
    root = Cluster(10)
    root.left = Cluster(11)
    root.right = Cluster(0)

    root.left.left = Cluster(1)
    root.left.right = Cluster(2)

    # Create assets
    a = "A"
    b = "B"
    c = "C"

    # Create dendrogram
    dendrogram = Dendrogram(root=root, assets=[a, b, c])

    # Collect portfolios from one_over_n algorithm
    portfolios: list[tuple[int, Portfolio]] = list(one_over_n(dendrogram))

    # Check that we get the expected number of levels
    assert len(portfolios) == len(root.levels)

    # Check the first level portfolio
    level0, portfolio0 = portfolios[0]
    assert level0 == 0

    # The first level should have weights that sum to 1
    assert sum(portfolio0.weights.values()) == pytest.approx(1.0)

    # Check that all assets are in the portfolio
    assert set(portfolio0.assets) == {a, b, c}

    # Check that weights decrease with each level
    if len(portfolios) > 1:
        _, portfolio1 = portfolios[1]
        # The sum of weights should still be 1 at each level
        assert sum(portfolio1.weights.values()) == pytest.approx(1.0)


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
    cor = compute_corr(returns)
    dendrogram = build_tree(cor=cor, method="ward")

    portfolios: list[tuple[int, Portfolio]] = list(one_over_n(dendrogram))

    assert len(portfolios) > 0
    assert len(portfolios) == len(dendrogram.root.levels)

    for _level, portfolio in portfolios:
        assert sum(portfolio.weights.values()) == pytest.approx(1.0)
        assert set(portfolio.assets) == set(dendrogram.assets)
        for weight in portfolio.weights.values():
            assert weight > 0


def test_solve_singular_falls_back_to_lstsq() -> None:
    """_solve returns the minimum-norm least-squares solution for a singular matrix."""
    m = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1, singular
    b = np.array([1.0, 2.0])
    x = _solve(m, b)
    assert np.allclose(m @ x, b)
    assert np.all(np.isfinite(x))


@pytest.mark.property
@settings(deadline=None, max_examples=200)
@given(cov=covariance_matrices())
def test_risk_parity_property_weights(cov: pl.DataFrame) -> None:
    """risk_parity should produce normalized long-only weights."""
    cov_np = cov.to_numpy()
    std = np.sqrt(np.diag(cov_np))
    corr = cov_np / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    cor = pl.DataFrame(dict(zip(cov.columns, corr, strict=True)))
    root = build_tree(cor=cor, method="single", bisection=False).root

    cluster = risk_parity(root=root, cov=cov)
    weights = np.array(list(cluster.portfolio.weights.values()))

    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert float(weights.sum()) == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_risk_parity_single_asset_weight_is_one() -> None:
    """Single-asset universe should allocate full weight to that asset."""
    cov = pl.DataFrame({"A": [0.25]})
    root = Cluster(0)

    cluster = risk_parity(root=root, cov=cov)

    assert cluster.portfolio.weights == {"A": 1.0}


def test_risk_parity_two_asset_closed_form_solution() -> None:
    """Two-asset universe should match closed-form risk-parity weights."""
    cov = pl.DataFrame({"A": [4.0, 0.0], "B": [0.0, 1.0]})
    root = Cluster(2, left=Cluster(0), right=Cluster(1))

    cluster = risk_parity(root=root, cov=cov)

    assert cluster.portfolio["A"] == pytest.approx(0.2)
    assert cluster.portfolio["B"] == pytest.approx(0.8)
    assert sum(cluster.portfolio.weights.values()) == pytest.approx(1.0)


def test_risk_parity_near_singular_covariance_matrix() -> None:
    """Near-singular covariance matrix should still yield valid weights."""
    rho = 0.999999
    cov = pl.DataFrame({"A": [1.0, rho], "B": [rho, 1.0]})
    root = Cluster(2, left=Cluster(0), right=Cluster(1))

    cluster = risk_parity(root=root, cov=cov)
    weights = np.array(list(cluster.portfolio.weights.values()))

    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 1.0)
    assert float(weights.sum()) == pytest.approx(1.0, rel=1e-6, abs=1e-6)
