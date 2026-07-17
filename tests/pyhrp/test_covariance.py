"""Tests for the covariance/correlation estimators."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars import DataFrame

from pyhrp.covariance import compute_corr, compute_cov


def test_compute_cov_matrix_properties(returns: DataFrame) -> None:
    """Test covariance helper returns a symmetric square matrix with matching columns."""
    cov = compute_cov(returns)
    n_assets = len(returns.columns)

    assert cov.shape == (n_assets, n_assets)
    assert cov.columns == returns.columns
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)


def test_compute_corr_matrix_properties(returns: DataFrame) -> None:
    """Test correlation helper returns a square matrix with unit diagonal and matching columns."""
    corr = compute_corr(returns)
    n_assets = len(returns.columns)

    assert corr.shape == (n_assets, n_assets)
    assert corr.columns == returns.columns
    assert np.diag(corr.to_numpy()).tolist() == pytest.approx([1.0] * n_assets)


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
