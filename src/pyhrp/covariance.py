"""Covariance and correlation estimation from returns.

This module isolates the second-moment estimators used by the HRP allocation
entry points:
- compute_cov: Covariance matrix from a DataFrame of returns
- compute_corr: Correlation matrix from a DataFrame of returns
- _returns: Simple returns from a DataFrame of prices
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["compute_corr", "compute_cov"]


def compute_cov(df: pl.DataFrame) -> pl.DataFrame:
    """Compute covariance matrix from a DataFrame of returns."""
    cols = df.columns
    cov = np.atleast_2d(np.cov(df.to_numpy().T))
    return pl.DataFrame(dict(zip(cols, cov, strict=True)))


def compute_corr(df: pl.DataFrame) -> pl.DataFrame:
    """Compute correlation matrix from a DataFrame of returns."""
    cols = df.columns
    corr = np.atleast_2d(np.corrcoef(df.to_numpy().T))
    return pl.DataFrame(dict(zip(cols, corr, strict=True)))


def _returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute simple returns from prices.

    Drops leading all-null rows produced by pct_change and fills remaining
    nulls/NaNs (e.g. from missing prices) with zero returns.
    """
    return (
        prices.select(pl.all().pct_change())
        .filter(pl.any_horizontal(pl.all().is_not_null()))
        .fill_null(0.0)
        .fill_nan(0.0)
    )
