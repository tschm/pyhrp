"""Performance and stress benchmarks for HRP and tree building."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from pyhrp.hrp import _compute_corr, build_tree, hrp


def _synthetic_prices(asset_count: int, periods: int = 500, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.01, size=(periods, asset_count))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pl.DataFrame(prices, schema=[f"asset_{idx:03d}" for idx in range(asset_count)])


@pytest.mark.stress
def test_benchmark_hrp_20_assets(benchmark: BenchmarkFixture, prices: pl.DataFrame) -> None:
    benchmark(lambda: hrp(prices=prices, method="ward", bisection=False))


@pytest.mark.stress
def test_benchmark_hrp_100_assets(benchmark: BenchmarkFixture) -> None:
    prices = _synthetic_prices(asset_count=100, seed=100)
    benchmark(lambda: hrp(prices=prices, method="ward", bisection=False))


@pytest.mark.stress
def test_benchmark_hrp_200_assets(benchmark: BenchmarkFixture) -> None:
    prices = _synthetic_prices(asset_count=200, seed=200)
    benchmark(lambda: hrp(prices=prices, method="ward", bisection=False))


@pytest.mark.stress
def test_benchmark_build_tree_100_assets(benchmark: BenchmarkFixture) -> None:
    prices = _synthetic_prices(asset_count=100, seed=300)
    returns = prices.select(pl.all().pct_change()).drop_nulls()
    cor = _compute_corr(returns)
    benchmark(lambda: build_tree(cor=cor, method="ward", bisection=False))


@pytest.mark.stress
def test_benchmark_build_tree_200_assets(benchmark: BenchmarkFixture) -> None:
    prices = _synthetic_prices(asset_count=200, seed=400)
    returns = prices.select(pl.all().pct_change()).drop_nulls()
    cor = _compute_corr(returns)
    benchmark(lambda: build_tree(cor=cor, method="ward", bisection=False))
