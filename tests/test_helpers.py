"""Direct tests for helper functions in pyhrp.hrp."""

from __future__ import annotations

import numpy as np
import pytest
from polars import DataFrame

from pyhrp.hrp import _compute_corr, _compute_cov


def test_compute_cov_matrix_properties(returns: DataFrame) -> None:
    """Test covariance helper returns a symmetric square matrix with matching columns."""
    cov = _compute_cov(returns)
    n_assets = len(returns.columns)

    assert cov.shape == (n_assets, n_assets)
    assert cov.columns == returns.columns
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)


def test_compute_corr_matrix_properties(returns: DataFrame) -> None:
    """Test correlation helper returns a square matrix with unit diagonal and matching columns."""
    corr = _compute_corr(returns)
    n_assets = len(returns.columns)

    assert corr.shape == (n_assets, n_assets)
    assert corr.columns == returns.columns
    assert np.diag(corr.to_numpy()).tolist() == pytest.approx([1.0] * n_assets)
