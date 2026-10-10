"""Tests for the covariance operator protocol, the dense adapter and the solve guard."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from cvx.linalg import FactorOperator, GramOperator

from pyhrp.operators import CovarianceOperator, DenseCovariance, as_operator, block, solve_block


def test_dense_covariance_matches_matrix() -> None:
    """DenseCovariance reproduces the products, solves and conditioning of its matrix."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal((6, 6))
    a = b @ b.T + np.eye(6)
    op = DenseCovariance(a)
    rows, cols = np.array([0, 3, 5]), np.array([1, 2])
    v = rng.standard_normal(2)
    assert op.n == 6
    np.testing.assert_allclose(op.diag, np.diag(a))
    np.testing.assert_allclose(op.block_matvec(rows, cols, v), a[np.ix_(rows, cols)] @ v)
    np.testing.assert_allclose(a[np.ix_(rows, rows)] @ op.solve_free(rows, np.ones(3)), np.ones(3))
    assert op.rcond_free(rows) == pytest.approx(1.0 / np.linalg.cond(a[np.ix_(rows, rows)]))


def test_dense_covariance_rcond_of_zero_block() -> None:
    """A block with no positive eigenvalue is reported as singular."""
    assert DenseCovariance(np.zeros((2, 2))).rcond_free(np.array([0, 1])) == 0.0


def test_cvx_linalg_operators_satisfy_the_protocol() -> None:
    """cvx-linalg backends are CovarianceOperators without an adapter."""
    assert isinstance(FactorOperator(np.ones(3), np.ones((3, 1)), np.eye(1)), CovarianceOperator)
    assert isinstance(GramOperator(np.ones((4, 3))), CovarianceOperator)
    assert isinstance(DenseCovariance(np.eye(2)), CovarianceOperator)


def test_as_operator_wraps_a_dataframe() -> None:
    """A DataFrame becomes a DenseCovariance named by its columns."""
    cov = pl.DataFrame({"A": [1.0, 0.5], "B": [0.5, 2.0]})
    op, names = as_operator(cov)
    assert isinstance(op, DenseCovariance)
    assert names == ["A", "B"]
    assert as_operator(cov, assets=["A", "B"])[1] == ["A", "B"]


def test_as_operator_rejects_mismatched_dataframe_names() -> None:
    """Names contradicting the DataFrame's columns are refused."""
    with pytest.raises(ValueError, match="do not match"):
        as_operator(pl.DataFrame({"A": [1.0]}), assets=["B"])


def test_as_operator_names_an_operator() -> None:
    """An operator gets the given names, or positional ones by default."""
    op = DenseCovariance(np.eye(3))
    assert as_operator(op)[1] == ["0", "1", "2"]
    assert as_operator(op, assets=["x", "y", "z"])[1] == ["x", "y", "z"]
    with pytest.raises(ValueError, match="expected 3 asset names"):
        as_operator(op, assets=["x"])


def test_as_operator_rejects_other_types() -> None:
    """Neither a DataFrame nor an operator is a TypeError."""
    with pytest.raises(TypeError, match="CovarianceOperator"):
        as_operator(np.eye(2))  # type: ignore[arg-type]


def test_block_materialises_a_sub_block() -> None:
    """block() returns Sigma[rows, cols] from products with the identity."""
    a = np.arange(16.0).reshape(4, 4)
    a = a + a.T
    rows, cols = np.array([1, 3]), np.array([0, 2, 3])
    np.testing.assert_allclose(block(DenseCovariance(a), rows, cols), a[np.ix_(rows, cols)])


def test_solve_block_uses_the_operator_solve() -> None:
    """A well-conditioned block is solved by the operator."""
    op = FactorOperator(np.array([1.0, 2.0, 3.0]), np.ones((3, 1)), np.eye(1))
    idx = np.array([0, 2])
    x = solve_block(op, idx, np.array([1.0, 1.0]))
    np.testing.assert_allclose(block(op, idx, idx) @ x, [1.0, 1.0])


def test_solve_block_singular_falls_back_to_lstsq() -> None:
    """A singular block gets the minimum-norm least-squares solution."""
    m = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1, singular
    b = np.array([1.0, 2.0])
    x = solve_block(DenseCovariance(m), np.array([0, 1]), b)
    np.testing.assert_allclose(m @ x, b)
    np.testing.assert_allclose(x, np.linalg.pinv(m) @ b)


def test_solve_block_failed_solve_falls_back_to_lstsq() -> None:
    """A solve that raises despite a passing condition check falls back to least squares."""

    class Failing(DenseCovariance):
        """Dense operator whose solve always fails."""

        def solve_free(self, free: object, rhs: np.ndarray) -> np.ndarray:
            """Raise as a singular LU factorisation would."""
            raise np.linalg.LinAlgError

    x = solve_block(Failing(np.diag([2.0, 4.0])), np.array([0, 1]), np.array([2.0, 4.0]))
    np.testing.assert_allclose(x, [1.0, 1.0])
