"""Tests for input validation and edge-case handling."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import pyhrp
from pyhrp import build_tree, compute_corr, compute_cov, hrp, risk_parity, schur_hrp, schur_risk_parity
from pyhrp.cluster import Cluster


def test_top_level_exports() -> None:
    """The public API is importable directly from the pyhrp package."""
    for name in pyhrp.__all__:
        assert getattr(pyhrp, name) is not None


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


def test_zero_variance_asset_raises_with_name() -> None:
    """A constant price series produces a clear error naming the asset."""
    prices = pl.DataFrame(
        {
            "A": [100.0, 101.0, 99.0, 102.0, 103.0],
            "B": [50.0, 50.0, 50.0, 50.0, 50.0],
        }
    )
    with (
        pytest.warns(RuntimeWarning),
        pytest.raises(ValueError, match=r"non-finite values for assets \['B'\]"),
    ):
        hrp(prices)


def test_compute_cov_single_asset() -> None:
    """Covariance of a single asset is a 1x1 matrix, not an obscure TypeError."""
    df = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    cov = compute_cov(df)
    assert cov.shape == (1, 1)
    assert cov["A"][0] == pytest.approx(1.0)


def test_compute_corr_single_asset() -> None:
    """Correlation of a single asset is a 1x1 matrix."""
    df = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    corr = compute_corr(df)
    assert corr.shape == (1, 1)
    assert corr["A"][0] == pytest.approx(1.0)


def test_build_tree_requires_two_assets() -> None:
    """build_tree rejects correlation matrices with fewer than two assets."""
    cor = pl.DataFrame({"A": [1.0]})
    with pytest.raises(ValueError, match="at least two assets"):
        build_tree(cor)


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


def test_schur_singular_block_does_not_crash() -> None:
    """Collinear assets (singular covariance block) fall back to least squares."""
    rng = np.random.default_rng(42)
    a = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 50))
    c = 50.0 + np.cumsum(rng.normal(0.0, 1.0, 50))
    d = 20.0 + np.cumsum(rng.normal(0.0, 1.0, 50))
    prices = pl.DataFrame({"A": a, "B": 2.0 * a, "C": c, "D": d})

    cluster = schur_hrp(prices, gamma=1.0)
    weights = cluster.portfolio.weights
    assert all(np.isfinite(w) for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
