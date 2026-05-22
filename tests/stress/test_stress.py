"""Stress tests for HRP at extreme universe sizes."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pyhrp.hrp import hrp


def _synthetic_prices(asset_count: int, periods: int = 500, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.01, size=(periods, asset_count))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pl.DataFrame(prices, schema=[f"asset_{idx:03d}" for idx in range(asset_count)])


@pytest.mark.stress
def test_hrp_500_assets() -> None:
    """HRP on a 500-asset universe should produce valid weights."""
    prices = _synthetic_prices(asset_count=500, seed=500)
    root = hrp(prices=prices, method="ward", bisection=False)
    weights = list(root.portfolio.weights.values())
    assert abs(sum(weights) - 1.0) < 1e-9
    assert all(0.0 <= w <= 1.0 for w in weights)
    assert len(weights) == 500


@pytest.mark.stress
def test_hrp_1000_assets() -> None:
    """HRP on a 1000-asset universe should produce valid weights."""
    prices = _synthetic_prices(asset_count=1000, seed=1000)
    root = hrp(prices=prices, method="ward", bisection=False)
    weights = list(root.portfolio.weights.values())
    assert abs(sum(weights) - 1.0) < 1e-9
    assert all(0.0 <= w <= 1.0 for w in weights)
    assert len(weights) == 1000
