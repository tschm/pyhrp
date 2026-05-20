"""Tests for the Hierarchical Risk Parity (HRP) function implementation."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars import DataFrame

from pyhrp.hrp import hrp


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
    w = cluster.portfolio.weights_dict  # dict[str, float], sorted alphabetically

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
    w = cluster.portfolio.weights_dict  # dict[str, float], sorted alphabetically

    ref = pl.read_csv(resource_dir / "weights_marcos.csv")
    ref = ref.rename({ref.columns[0]: "asset"})

    for row in ref.iter_rows(named=True):
        assert w[row["asset"]] == pytest.approx(row["Weights"], rel=1e-5)
