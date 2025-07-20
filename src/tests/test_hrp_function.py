"""Tests for the Hierarchical Risk Parity (HRP) function implementation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame

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
    # Calculate HRP portfolio weights using ward linkage without bisection
    cluster = hrp(prices=prices, method="ward", bisection=False)

    # Extract weights and convert asset objects to names for comparison
    w = cluster.portfolio.weights
    w.index = [asset.name for asset in w.index]

    # Load reference weights from file
    # Uncomment this line if you want to generate a new reference file:
    # w.to_csv(resource_dir / "weights_hrp.csv", header=False)
    x = pd.read_csv(resource_dir / "weights_hrp.csv", index_col=0, header=0).squeeze()
    x.index.name = None

    # Verify the calculated weights match the reference weights
    pd.testing.assert_series_equal(x, w, check_exact=False)


def test_marcos(resource_dir: Path, prices: DataFrame) -> None:
    """Test the HRP function with bisection method (Marcos Lopez de Prado's approach).

    This test verifies:
    1. The HRP function with bisection correctly calculates portfolio weights
    2. The resulting weights match the expected values from a reference file

    Args:
        resource_dir: Path to test resources directory
        prices: DataFrame of asset prices
    """
    # Calculate HRP portfolio weights using ward linkage with bisection
    cluster = hrp(prices=prices, method="ward", bisection=True)

    # Extract weights and convert asset objects to names for comparison
    w = cluster.portfolio.weights
    w.index = [asset.name for asset in w.index]

    # Load reference weights from file
    # Uncomment this line if you want to generate a new reference file:
    # w.to_csv(resource_dir / "weights_marcos.csv", header=False)
    x = pd.read_csv(resource_dir / "weights_marcos.csv", index_col=0, header=0).squeeze()
    x.index.name = None

    # Verify the calculated weights match the reference weights
    pd.testing.assert_series_equal(x, w, check_exact=False)
