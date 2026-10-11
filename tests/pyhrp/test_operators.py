"""Tests for the covariance operator protocol, the dense adapter and the solve guard."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from cvx.linalg import FactorOperator, GramOperator

from pyhrp.operators import CovarianceOperator, DenseCovariance, as_operator, block, cg_min_norm, solve_block


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
    x = rng.standard_normal(6)
    np.testing.assert_allclose(op.matvec(x), a @ x)
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


def test_solve_block_singular_falls_back_to_cg() -> None:
    """A singular block gets the minimum-norm solution, by conjugate gradients."""
    m = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1, singular
    b = np.array([1.0, 2.0])
    x = solve_block(DenseCovariance(m), np.array([0, 1]), b)
    np.testing.assert_allclose(m @ x, b)
    np.testing.assert_allclose(x, np.linalg.pinv(m) @ b)


def test_solve_block_failed_solve_falls_back_to_cg() -> None:
    """A solve that raises despite a passing condition check falls back to conjugate gradients."""

    class Failing(DenseCovariance):
        """Dense operator whose solve always fails."""

        def rcond_free(self, free: object) -> float:
            """Report a well-conditioned block, so the solve is attempted."""
            return 1.0

        def solve_free(self, free: object, rhs: np.ndarray) -> np.ndarray:
            """Raise as a singular LU factorisation would."""
            raise np.linalg.LinAlgError

    x = solve_block(Failing(np.diag([2.0, 4.0])), np.array([0, 1]), np.array([2.0, 4.0]))
    np.testing.assert_allclose(x, [1.0, 1.0])


def test_cg_matches_lstsq_on_a_singular_gram_block_without_forming_it() -> None:
    """On a Gram block with more assets than observations CG gives lstsq's minimum-norm solution."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((30, 200))
    xc = (x - x.mean(axis=0)) / np.sqrt(29.0)
    op = GramOperator(xc)
    left, right = np.arange(100), np.arange(100, 200)
    b = op.block_matvec(right, left, np.full(100, 0.01))
    assert op.rcond_free(right) < 1e-12

    products = 0

    def apply(v: np.ndarray) -> np.ndarray:
        """Count the block products CG takes."""
        nonlocal products
        products += 1
        return np.asarray(op.block_matvec(right, right, v))

    solution = cg_min_norm(apply, b)
    reference = np.linalg.lstsq(block(op, right, right), b, rcond=None)[0]
    np.testing.assert_allclose(solution, reference, rtol=1e-8, atol=1e-10 * np.abs(reference).max())
    assert b @ solution == pytest.approx(b @ reference, rel=1e-12)
    assert products <= 60  # about the rank (30), far below the block size (100)


def test_cg_zero_rhs_returns_zero() -> None:
    """A zero right-hand side is solved without a product."""
    np.testing.assert_array_equal(cg_min_norm(lambda v: pytest.fail("no product expected"), np.zeros(3)), np.zeros(3))


def test_cg_stops_on_a_null_direction() -> None:
    """An inconsistent system stops when a search direction has no curvature."""
    a = np.diag([1.0, 0.0])
    assert np.all(np.isfinite(cg_min_norm(lambda v: a @ v, np.array([1.0, 1.0]))))


def test_cg_stops_after_twice_the_dimension() -> None:
    """The iteration is capped at 2 * len(b) products."""
    products = 0
    rotation = np.array([[1.0, 1.0], [-1.0, 1.0]])  # positive curvature, but not symmetric: CG never converges

    def apply(v: np.ndarray) -> np.ndarray:
        """Count products with a matrix CG cannot solve."""
        nonlocal products
        products += 1
        return rotation @ v

    cg_min_norm(apply, np.array([1.0, 0.0]), rtol=0.0)
    assert products == 4


def test_solve_block_uses_a_restricted_operator_when_offered() -> None:
    """A singular Gram block is sliced once by restricted() and solved with its matvec."""
    rng = np.random.default_rng(8)
    x = rng.standard_normal((5, 12))
    op = GramOperator(x)
    idx = np.arange(12)
    b = op.block_matvec(idx, idx, rng.standard_normal(12))
    sliced = []
    real = op.restricted

    def counting(free: np.ndarray) -> GramOperator:
        """Record each slicing."""
        sliced.append(free)
        return real(free)

    op.restricted = counting  # type: ignore[method-assign]
    solution = solve_block(op, idx, b)
    assert len(sliced) == 1
    np.testing.assert_allclose(op.block_matvec(idx, idx, solution), b, atol=1e-8)


def test_solve_block_on_an_operator_that_cannot_solve() -> None:
    """NotImplementedError from rcond_free or solve_free sends the block to CG."""

    class ForwardOnly(DenseCovariance):
        """Dense operator offering products only."""

        def rcond_free(self, free: object) -> float:
            """Not available."""
            raise NotImplementedError

    x = solve_block(ForwardOnly(np.diag([2.0, 4.0])), np.array([0, 1]), np.array([2.0, 4.0]))
    np.testing.assert_allclose(x, [1.0, 1.0])
