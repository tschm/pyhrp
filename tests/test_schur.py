"""Tests for Schur Complementary Allocation (Peter Cotton, arXiv:2411.05807)."""

from __future__ import annotations

import polars as pl
import pytest
from polars import DataFrame

from pyhrp.algos import risk_parity, schur_risk_parity
from pyhrp.cluster import Cluster
from pyhrp.hrp import build_tree, compute_corr, compute_cov, hrp, schur_hrp
from pyhrp.treelib import Node


def test_schur_weights_sum_to_one(prices: DataFrame) -> None:
    """Portfolio weights must sum to 1."""
    cluster = schur_hrp(prices=prices, method="ward", gamma=0.5)
    total = sum(cluster.portfolio.weights.values())
    assert total == pytest.approx(1.0, rel=1e-6)


def test_gamma_zero_matches_hrp(prices: DataFrame) -> None:
    """At gamma=0, Schur HRP must produce the same weights as standard HRP."""
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


def test_schur_reduces_variance_vs_hrp(prices: DataFrame) -> None:
    """Schur HRP with gamma>0 should produce equal or lower variance than standard HRP."""
    returns = (
        prices.select(pl.all().pct_change())
        .filter(pl.any_horizontal(pl.all().is_not_null()))
        .fill_null(0.0)
        .fill_nan(0.0)
    )
    cov = compute_cov(returns)

    hrp_cluster = hrp(prices=prices, method="ward")
    schur_cluster = schur_hrp(prices=prices, method="ward", gamma=1.0)

    v_hrp = hrp_cluster.portfolio.variance(cov)
    v_schur = schur_cluster.portfolio.variance(cov)

    assert v_schur <= v_hrp + 1e-10


def test_all_weights_positive(prices: DataFrame) -> None:
    """All portfolio weights must be strictly positive (no short positions)."""
    cluster = schur_hrp(prices=prices, method="ward", gamma=0.5)
    for asset, w in cluster.portfolio.weights.items():
        assert w > 0.0, f"Weight for {asset} is non-positive: {w}"


@pytest.mark.parametrize("gamma", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_weights_sum_to_one_for_all_gamma(prices: DataFrame, gamma: float) -> None:
    """Weights sum to 1 for all gamma values."""
    cluster = schur_hrp(prices=prices, method="ward", gamma=gamma)
    assert sum(cluster.portfolio.weights.values()) == pytest.approx(1.0, rel=1e-6)


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


def test_schur_accepts_custom_node(prices: DataFrame) -> None:
    """schur_hrp accepts a pre-built node and uses it."""
    returns = (
        prices.select(pl.all().pct_change())
        .filter(pl.any_horizontal(pl.all().is_not_null()))
        .fill_null(0.0)
        .fill_nan(0.0)
    )
    cor = compute_corr(returns)
    node = build_tree(cor, method="single").root
    cluster = schur_hrp(prices=prices, node=node, gamma=0.5)
    assert sum(cluster.portfolio.weights.values()) == pytest.approx(1.0, rel=1e-6)


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
