"""Covariance and correlation estimation from returns.

This module isolates the second-moment estimators used by the HRP allocation
entry points:
- compute_returns: Simple returns from a DataFrame of prices
- compute_cov: Covariance matrix from a DataFrame of returns
- compute_corr: Correlation matrix from a DataFrame of returns
- check_finite_matrix: Guard that rejects matrices containing NaN/null entries
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

__all__ = ["check_finite_matrix", "compute_corr", "compute_cov", "compute_returns"]


def _warn_on_missing_returns(returns: pl.DataFrame) -> None:
    """Emit a warning when nulls or NaNs survive into the returns frame.

    Args:
        returns (pl.DataFrame): Returns frame straight out of ``pct_change``
    """
    null_counts = returns.null_count().row(0, named=True)
    nan_counts = {c: int(n.is_nan().sum()) for c, n in ((col, returns[col]) for col in returns.columns)}
    affected = {
        asset: count + nan_counts[asset] for asset, count in null_counts.items() if count + nan_counts[asset] > 0
    }
    if affected:
        detail = ", ".join(f"{a} ({n} rows)" for a, n in sorted(affected.items()))
        warnings.warn(
            f"Missing prices detected for: {detail}. "
            "They are filled with zero returns, which biases covariance estimates. "
            "Clean the data upstream or drop affected rows/assets.",
            stacklevel=3,
        )


def compute_returns(prices: pl.DataFrame) -> pl.DataFrame:
    """Compute simple returns from prices.

    Drops leading all-null rows produced by pct_change and fills remaining
    nulls/NaNs (e.g. from missing prices) with zero returns.

    Warning:
        Filling with zero is a dirty helper, not a data-cleaning strategy: a
        suspended asset or an asset listed mid-sample contributes fabricated
        zero returns that bias variance down and correlations toward zero.
        A :class:`UserWarning` is emitted whenever this path fires. Callers
        working with messy universes should clean the data first and compose
        the pipeline manually::

            from pyhrp import build_tree, risk_parity
            from pyhrp import compute_corr, compute_cov

            returns = prices.select(pl.all().pct_change()).drop_nulls()
            root = risk_parity(root=build_tree(compute_corr(returns)).root,
                               cov=compute_cov(returns))

    Args:
        prices (pl.DataFrame): Asset price time series (columns are assets, rows are dates)

    Returns:
        pl.DataFrame: Simple returns, one row shorter than ``prices``

    Examples:
        >>> import polars as pl
        >>> from pyhrp.covariance import compute_returns
        >>> prices = pl.DataFrame({"A": [100.0, 110.0, 99.0]})
        >>> compute_returns(prices)["A"].to_list()
        [0.1, -0.1]
    """
    raw = prices.select(pl.all().pct_change()).filter(pl.any_horizontal(pl.all().is_not_null()))
    _warn_on_missing_returns(raw)
    return raw.fill_null(0.0).fill_nan(0.0)


def check_finite_matrix(matrix: pl.DataFrame, name: str = "matrix") -> pl.DataFrame:
    """Raise if a matrix contains NaN or null entries.

    A covariance matrix with non-finite entries makes every downstream weight
    meaningless; this guard fails loudly instead of propagating garbage.

    Args:
        matrix (pl.DataFrame): Square matrix (columns are assets)
        name (str): Human-readable name used in the error message

    Returns:
        pl.DataFrame: The input matrix, unchanged

    Raises:
        ValueError: If any entry of ``matrix`` is NaN or null.
    """
    bad = {
        col: int(count) + int(matrix[col].is_nan().sum())
        for col, count in matrix.null_count().row(0, named=True).items()
        if count > 0 or bool(matrix[col].is_nan().any())
    }
    if bad:
        detail = ", ".join(f"{c} ({n})" for c, n in sorted(bad.items()))
        msg = f"{name} contains NaN/null entries in column(s): {detail}"
        raise ValueError(msg)
    return matrix


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
