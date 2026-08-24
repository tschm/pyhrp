"""Tests for the covariance/correlation estimators."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from polars import DataFrame

from pyhrp.covariance import check_finite_matrix, compute_corr, compute_cov, compute_returns
from pyhrp.hrp import hrp, schur_hrp


def test_compute_returns_simple_returns() -> None:
    """Simple returns are pct_change, one row shorter than the price frame."""
    prices = pl.DataFrame({"A": [100.0, 110.0, 99.0], "B": [50.0, 55.0, 44.0]})
    rets = compute_returns(prices)

    assert rets.shape == (2, 2)
    assert rets.columns == prices.columns
    assert rets["A"].to_list() == pytest.approx([0.1, -0.1])
    assert rets["B"].to_list() == pytest.approx([0.1, -0.2])


def test_compute_returns_fills_missing_with_zero() -> None:
    """Nulls and NaNs from missing or flat prices become zero returns, not holes."""
    prices = pl.DataFrame({"A": [100.0, None, 110.0], "B": [0.0, 0.0, 50.0]})
    rets = compute_returns(prices)

    assert rets.null_count().to_numpy().sum() == 0
    assert not np.isnan(rets.to_numpy()).any()


def test_compute_returns_warns_on_missing_prices() -> None:
    """Missing prices trigger a UserWarning naming the affected asset."""
    # A NaN price produces NaN returns that survive the leading-row drop,
    # so column A is flagged before being zero-filled.
    prices = pl.DataFrame({"A": [100.0, float("nan"), 110.0], "B": [50.0, 51.0, 52.0]})
    with pytest.warns(UserWarning, match="A"):
        compute_returns(prices)


def test_compute_returns_silent_when_clean() -> None:
    """No warning is emitted when the returns frame has no gaps."""
    prices = pl.DataFrame({"A": [100.0, 110.0, 99.0], "B": [50.0, 55.0, 44.0]})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_returns(prices)


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


def test_check_finite_matrix_passes_clean_matrix() -> None:
    """A finite matrix passes through unchanged."""
    matrix = pl.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]})

    result = check_finite_matrix(matrix, name="covariance matrix")

    assert result.equals(matrix)


def test_check_finite_matrix_raises_on_nan() -> None:
    """NaN entries raise ValueError naming the offending column."""
    matrix = pl.DataFrame({"A": [1.0, float("nan")], "B": [0.5, 1.0]})

    with pytest.raises(ValueError, match=r"covariance matrix.*A"):
        check_finite_matrix(matrix, name="covariance matrix")


def test_check_finite_matrix_raises_on_null() -> None:
    """Null entries raise ValueError naming the offending column."""
    matrix = pl.DataFrame({"A": [1.0, 0.5], "B": [None, 1.0]}, schema={"A": pl.Float64, "B": pl.Float64})

    with pytest.raises(ValueError, match=r"null entries in column\(s\): B"):
        check_finite_matrix(matrix)


def test_hrp_raises_on_constant_price_series() -> None:
    """Hrp fails loudly on a constant price series (NaN correlation downstream)."""
    prices = pl.DataFrame({"A": [100.0, 100.0, 100.0], "B": [50.0, 51.0, 49.0]})

    with pytest.raises(ValueError, match="non-finite"):
        hrp(prices)


def test_schur_hrp_raises_on_nonfinite_covariance() -> None:
    """schur_hrp fails loudly when the covariance estimate contains NaN."""
    prices = pl.DataFrame(
        {
            "A": [100.0, 100.0, 100.0, 100.0],
            "B": [50.0, 51.0, 49.0, 52.0],
        }
    )

    with pytest.raises(ValueError, match=r"non-finite|covariance"):
        schur_hrp(prices)
