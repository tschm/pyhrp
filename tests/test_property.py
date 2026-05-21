"""Property-based and numerical edge-case tests for HRP."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pyhrp.algos import risk_parity
from pyhrp.cluster import Cluster
from pyhrp.hrp import build_tree


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
    return pl.DataFrame(dict(zip(assets, cov, strict=False)))


@st.composite
def correlation_matrices(draw: st.DrawFn) -> pl.DataFrame:
    """Generate valid correlation matrices from covariance matrices."""
    cov = draw(covariance_matrices())
    cov_np = cov.to_numpy()
    std = np.sqrt(np.diag(cov_np))
    corr = cov_np / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    assets = cov.columns
    return pl.DataFrame(dict(zip(assets, corr, strict=False)))


@pytest.mark.property
@settings(deadline=None, max_examples=50)
@given(cor=correlation_matrices())
def test_build_tree_property_valid_corr_matrix(cor: pl.DataFrame) -> None:
    """build_tree should accept valid random correlation matrices."""
    dendrogram = build_tree(cor=cor, method="single", bisection=False)

    assert dendrogram.linkage is not None
    assert dendrogram.assets == cor.columns
    assert len(dendrogram.root.leaves) == len(cor.columns)


@pytest.mark.property
@settings(deadline=None, max_examples=50)
@given(cov=covariance_matrices())
def test_risk_parity_property_weights(cov: pl.DataFrame) -> None:
    """risk_parity should produce normalized long-only weights."""
    cov_np = cov.to_numpy()
    std = np.sqrt(np.diag(cov_np))
    corr = cov_np / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    cor = pl.DataFrame(dict(zip(cov.columns, corr, strict=False)))
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


def test_build_tree_depth_one_bisection_two_assets() -> None:
    """Two-asset bisection tree should have depth 1 and valid weights."""
    cor = pl.DataFrame({"A": [1.0, 0.25], "B": [0.25, 1.0]})
    cov = pl.DataFrame({"A": [0.04, 0.01], "B": [0.01, 0.09]})

    dendrogram = build_tree(cor=cor, method="single", bisection=True)

    assert len(dendrogram.root.levels) == 2
    assert dendrogram.root.left is not None
    assert dendrogram.root.right is not None
    assert dendrogram.root.left.is_leaf
    assert dendrogram.root.right.is_leaf

    cluster = risk_parity(root=dendrogram.root, cov=cov)
    assert sum(cluster.portfolio.weights.values()) == pytest.approx(1.0)
