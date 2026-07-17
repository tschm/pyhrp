"""Tests for the HRP and Schur allocation entry points.

Covers the top-level hrp/schur_hrp functions, the package-level public API,
and the marimo notebook integration that drives the same pipeline.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars import DataFrame

import pyhrp
from pyhrp.cluster import Cluster
from pyhrp.covariance import compute_corr, compute_cov
from pyhrp.dendrogram import build_tree
from pyhrp.hrp import hrp, schur_hrp


def test_top_level_exports() -> None:
    """The public API is importable directly from the pyhrp package."""
    for name in pyhrp.__all__:
        assert getattr(pyhrp, name) is not None


def test_hrp(prices: DataFrame, resource_dir: Path) -> None:
    """Test the HRP function with standard hierarchical clustering.

    This test verifies:
    1. The HRP function correctly calculates portfolio weights
    2. The resulting weights match the expected values from a reference file

    Args:
        prices: DataFrame of asset prices
        resource_dir: Path to test resources directory
    """
    cluster = hrp(prices=prices, method="ward", bisection=False)
    w = cluster.portfolio.weights  # dict[str, float], sorted alphabetically

    ref = pl.read_csv(resource_dir / "weights_hrp.csv")
    ref = ref.rename({ref.columns[0]: "asset"})

    for row in ref.iter_rows(named=True):
        assert w[row["asset"]] == pytest.approx(row["Weights"], rel=1e-5)


def test_marcos(resource_dir: Path, prices: DataFrame) -> None:
    """Test the HRP function with bisection method (Marcos Lopez de Prado's approach).

    This test verifies:
    1. The HRP function with bisection correctly calculates portfolio weights
    2. The resulting weights match the expected values from a reference file

    Args:
        resource_dir: Path to test resources directory
        prices: DataFrame of asset prices
    """
    cluster = hrp(prices=prices, method="ward", bisection=True)
    w = cluster.portfolio.weights  # dict[str, float], sorted alphabetically

    ref = pl.read_csv(resource_dir / "weights_marcos.csv")
    ref = ref.rename({ref.columns[0]: "asset"})

    for row in ref.iter_rows(named=True):
        assert w[row["asset"]] == pytest.approx(row["Weights"], rel=1e-5)


def test_hrp_without_node(prices: pl.DataFrame) -> None:
    """Test hrp function without providing a pre-built node."""
    result = hrp(prices=prices, node=None, method="single", bisection=False)

    assert result is not None
    assert isinstance(result, Cluster)
    assert len(result.portfolio.assets) > 0


def test_hrp_with_node(prices: pl.DataFrame, returns: pl.DataFrame) -> None:
    """Test hrp function with a pre-built node."""
    cor = compute_corr(returns)
    dendrogram = build_tree(cor=cor, method="ward", bisection=False)

    result = hrp(prices=prices, node=dendrogram.root, method="ward", bisection=False)

    assert result is not None
    assert isinstance(result, Cluster)


def test_hrp_with_bisection(prices: pl.DataFrame) -> None:
    """Test hrp function with bisection enabled."""
    result = hrp(prices=prices, node=None, method="single", bisection=True)

    assert result is not None
    assert isinstance(result, Cluster)


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_hrp_with_different_methods(prices: pl.DataFrame, method: str) -> None:
    """Test hrp with all supported linkage methods."""
    result = hrp(prices=prices, node=None, method=method, bisection=False)

    assert result is not None
    assert isinstance(result, Cluster)


def test_hrp_weights_sum_to_one(prices: pl.DataFrame) -> None:
    """Test that HRP weights sum to approximately 1."""
    result = hrp(prices=prices, node=None, method="ward", bisection=False)

    weights_sum = sum(result.portfolio.weights.values())
    assert weights_sum == pytest.approx(1.0, rel=1e-6)


def test_hrp_with_small_dataset() -> None:
    """Test hrp with a minimal dataset (2 assets)."""
    prices = pl.DataFrame(
        {"A": [100, 101, 102, 101, 103, 104, 103, 105, 106, 107], "B": [50, 51, 50, 52, 51, 53, 54, 53, 55, 56]}
    )

    result = hrp(prices=prices, node=None, method="single", bisection=False)

    assert result is not None
    assert len(result.portfolio.assets) == 2
    weights_sum = sum(result.portfolio.weights.values())
    assert weights_sum == pytest.approx(1.0, rel=1e-6)


def test_hrp_handles_missing_data_in_prices() -> None:
    """Test that hrp correctly handles price data with missing values."""
    col_a = [float(100 + i + (i % 3)) for i in range(20)]
    col_a[5] = None  # type: ignore[call-overload]
    col_b = [float(50 + i * 0.5 - (i % 2)) for i in range(20)]
    col_b[10] = None  # type: ignore[call-overload]
    col_c = [float(75 + i * 0.3) for i in range(20)]

    prices = pl.DataFrame({"A": col_a, "B": col_b, "C": col_c})

    result = hrp(prices=prices, node=None, method="ward", bisection=False)

    assert result is not None
    weights_sum = sum(result.portfolio.weights.values())
    assert weights_sum == pytest.approx(1.0, rel=1e-6)


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


def test_schur_weights_sum_to_one(prices: DataFrame) -> None:
    """Portfolio weights must sum to 1."""
    cluster = schur_hrp(prices=prices, method="ward", gamma=0.5)
    total = sum(cluster.portfolio.weights.values())
    assert total == pytest.approx(1.0, rel=1e-6)


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


def test_notebooks() -> None:
    """Test notebooks execute and expose expected typed outputs."""
    repo_root = Path(__file__).resolve().parents[2]
    notebooks_dir = repo_root / "book" / "marimo"
    prices_path = repo_root / "tests" / "resources" / "stock_prices.csv"

    for py_file in notebooks_dir.glob("*.py"):
        namespace = runpy.run_path(str(py_file))

        if py_file.name == "demo.py":
            prices = namespace["_load_prices"](prices_path)
            returns = prices.select(pl.all().pct_change()).drop_nulls()
            cov, cor = namespace["_compute_cov_and_corr"](returns)
            root = namespace["risk_parity"](namespace["build_tree"](cor, method="ward").root, cov)

            assert isinstance(root, Cluster)
            assert isinstance(root.portfolio.weights, dict)
            assert root.portfolio.weights
            weights = list(root.portfolio.weights.values())
            assert abs(sum(weights) - 1.0) < 1e-9, "HRP weights must sum to 1.0"
            assert all(0.0 <= w <= 1.0 for w in weights), "All HRP weights must be in [0, 1]"
        else:
            assert "app" in namespace
