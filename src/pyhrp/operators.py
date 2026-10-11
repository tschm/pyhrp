"""The covariance as an operator: the products HRP needs, without the matrix.

The allocators never need the covariance as an explicit array. Risk parity needs
quadratic forms on principal blocks, the Schur variant adds cross-block products
and solves against a principal block, and Ward or single-linkage clustering
needs the diagonal plus products with the full matrix. This module captures that
as the structural protocol :class:`CovarianceOperator`, six members of the
``SymmetricOperator`` of `cvx-linalg <https://github.com/Jebel-Quant/linalg>`_,
whose dense, factor (diagonal-plus-low-rank) and Gram (returns-matrix) backends
satisfy it without an adapter. A factor or Gram backend never forms the
``n x n`` covariance, so the allocation runs in ``O(n k)`` memory.

- CovarianceOperator: The structural protocol the allocators consume
- DenseCovariance: The protocol over an explicit matrix (what a DataFrame is wrapped in)
- as_operator: Turn a covariance DataFrame or an operator into an operator plus asset names
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl

__all__ = ["CovarianceOperator", "DenseCovariance", "as_operator", "cg_min_norm"]

# Below this reciprocal condition number a principal block is treated as singular
# and solved by conjugate gradients for the minimum-norm solution.
RCOND_MIN = 1e-12

# Conjugate gradients stop once the residual is this small relative to the right-hand side.
CG_RTOL = 1e-10


@runtime_checkable
class CovarianceOperator(Protocol):
    """A symmetric covariance reached only through block products and a block solve.

    Index sets are 1-D integer arrays of positions into ``range(n)``.
    """

    @property
    def n(self) -> int:
        """Number of assets."""
        ...

    @property
    def diag(self) -> np.ndarray:
        """The variances, ``diag(Sigma)``, as a length-``n`` vector."""
        ...

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Return ``Sigma @ x``."""
        ...

    def block_matvec(self, rows: Any, cols: Any, v: np.ndarray) -> np.ndarray:
        """Return ``Sigma[rows, cols] @ v``."""
        ...

    def solve_free(self, free: Any, rhs: np.ndarray) -> np.ndarray:
        """Return ``Sigma[free, free]^{-1} @ rhs``."""
        ...

    def rcond_free(self, free: Any) -> float:
        """Return the reciprocal condition number of ``Sigma[free, free]``."""
        ...


class DenseCovariance:
    """:class:`CovarianceOperator` over an explicit symmetric matrix.

    This is what a covariance DataFrame is wrapped in, so the DataFrame path and
    the operator path run the same allocation code.

    Args:
        matrix: The ``n x n`` covariance.

    Examples:
        >>> import numpy as np
        >>> from pyhrp.operators import DenseCovariance
        >>> op = DenseCovariance(np.array([[4.0, 1.0], [1.0, 9.0]]))
        >>> op.diag.tolist()
        [4.0, 9.0]
        >>> op.block_matvec(np.array([1]), np.array([0, 1]), np.array([1.0, 1.0])).tolist()
        [10.0]
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """Store the matrix as a float64 array."""
        self._a = np.asarray(matrix, dtype=np.float64)

    @property
    def n(self) -> int:
        """Number of assets."""
        return int(self._a.shape[0])

    @property
    def diag(self) -> np.ndarray:
        """The variances, ``diag(Sigma)``."""
        return np.diagonal(self._a).copy()

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Return ``Sigma @ x``."""
        return np.asarray(self._a @ x)

    def block_matvec(self, rows: Any, cols: Any, v: np.ndarray) -> np.ndarray:
        """Return ``Sigma[rows, cols] @ v``."""
        return np.asarray(self._a[np.ix_(rows, cols)] @ v)

    def solve_free(self, free: Any, rhs: np.ndarray) -> np.ndarray:
        """Return ``Sigma[free, free]^{-1} @ rhs`` by an LU solve on the block."""
        return np.asarray(np.linalg.solve(self._a[np.ix_(free, free)], rhs))

    def rcond_free(self, free: Any) -> float:
        """Return the reciprocal 2-norm condition number of ``Sigma[free, free]``."""
        eig = np.linalg.eigvalsh(self._a[np.ix_(free, free)])
        if eig[-1] <= 0.0:
            return 0.0
        return max(float(eig[0]), 0.0) / float(eig[-1])


def as_operator(
    cov: pl.DataFrame | CovarianceOperator, assets: Sequence[str] | None = None
) -> tuple[CovarianceOperator, list[str]]:
    """Return the covariance as an operator together with the asset names.

    A DataFrame is wrapped in :class:`DenseCovariance` and names its own assets
    (its columns). An operator carries no names, so they come from *assets*, or
    default to ``"0", "1", ...``.

    Args:
        cov: A square covariance DataFrame (columns are assets) or an operator.
        assets: Asset names for an operator; must match the columns for a DataFrame.

    Returns:
        tuple[CovarianceOperator, list[str]]: The operator and the ``n`` asset names.

    Raises:
        TypeError: If *cov* is neither a DataFrame nor a :class:`CovarianceOperator`.
        ValueError: If *assets* does not have one name per asset, or contradicts
            the DataFrame's columns.

    Examples:
        >>> import polars as pl
        >>> from pyhrp.operators import as_operator
        >>> op, names = as_operator(pl.DataFrame({"A": [1.0, 0.0], "B": [0.0, 2.0]}))
        >>> op.n, names
        (2, ['A', 'B'])
    """
    if isinstance(cov, pl.DataFrame):
        if assets is not None and list(assets) != cov.columns:
            msg = f"assets {list(assets)} do not match the covariance columns {cov.columns}"
            raise ValueError(msg)
        return DenseCovariance(cov.to_numpy()), list(cov.columns)
    if not isinstance(cov, CovarianceOperator):
        msg = f"cov must be a polars DataFrame or a CovarianceOperator, got {type(cov).__name__}"
        raise TypeError(msg)
    names = [str(i) for i in range(cov.n)] if assets is None else list(assets)
    if len(names) != cov.n:
        msg = f"expected {cov.n} asset names, got {len(names)}"
        raise ValueError(msg)
    return cov, names


def block(op: CovarianceOperator, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Materialise ``Sigma[rows, cols]`` from ``len(cols)`` products with the identity."""
    return np.asarray(op.block_matvec(rows, cols, np.eye(cols.size)))


def solve_block(op: CovarianceOperator, idx: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve against the principal block ``Sigma[idx, idx]``, robust to singularity.

    Covariance blocks of collinear assets, or of more assets than observations
    (a Gram operator with ``T < n``), are singular. A block the operator reports
    as singular, or whose solve fails, is solved by conjugate gradients over
    ``block_matvec`` (see :func:`cg_min_norm`), which never forms the block.
    """
    if op.rcond_free(idx) >= RCOND_MIN:
        try:
            return np.asarray(op.solve_free(idx, rhs))
        except np.linalg.LinAlgError:
            pass
    return cg_min_norm(_block_product(op, idx), rhs)


def _block_product(op: CovarianceOperator, idx: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """The product ``v -> Sigma[idx, idx] @ v`` for an iterative solve.

    ``block_matvec`` gathers the block's storage on every call (the columns of a
    Gram factor, the rows of factor loadings), which dominates a long CG run. An
    operator that offers ``restricted(idx)``, as the cvx-linalg backends do, is
    sliced once and then multiplied with its plain ``matvec``.
    """
    restricted = getattr(op, "restricted", None)
    if callable(restricted):
        sub = restricted(idx)
        return lambda v: np.asarray(sub.matvec(v))
    return lambda v: np.asarray(op.block_matvec(idx, idx, v))


def cg_min_norm(apply: Callable[[np.ndarray], np.ndarray], b: np.ndarray, rtol: float = CG_RTOL) -> np.ndarray:
    """Minimum-norm solution of ``A x = b`` for a symmetric PSD ``A``, by conjugate gradients.

    The Schur right-hand sides are ``Sigma[R, L] w_L``, and for a PSD ``Sigma``
    the columns of ``Sigma[R, L]`` lie in the range of ``Sigma[R, R]``, so the
    system is consistent even when the block is singular. Started from zero, CG
    then stays in that range and converges to the minimum-norm solution, the
    one ``lstsq`` returns, using only products with ``A``.

    Iteration stops when the residual falls below ``rtol * |b|``, when a search
    direction has no curvature (it lies in the null space, so the residual left
    is round-off outside the range), or after ``2 * len(b)`` steps. In exact
    arithmetic CG terminates within ``rank(A)`` steps.

    Args:
        apply (Callable[[np.ndarray], np.ndarray]): The product ``v -> A @ v``.
        b (np.ndarray): The right-hand side.
        rtol (float): Relative residual tolerance.

    Returns:
        np.ndarray: The approximate minimum-norm solution.

    Examples:
        >>> import numpy as np
        >>> from pyhrp.operators import cg_min_norm
        >>> a = np.array([[1.0, 2.0], [2.0, 4.0]])  # singular
        >>> x = cg_min_norm(lambda v: a @ v, np.array([1.0, 2.0]))
        >>> np.allclose(x, np.linalg.pinv(a) @ [1.0, 2.0])
        True
    """
    x = np.zeros_like(b, dtype=np.float64)
    r = np.asarray(b, dtype=np.float64).copy()
    p = r.copy()
    rr = float(r @ r)
    stop = (rtol**2) * rr
    for _ in range(2 * b.size):
        if rr <= stop:
            break
        ap = apply(p)
        curvature = float(p @ ap)
        if curvature <= 0.0:
            break
        alpha = rr / curvature
        x += alpha * p
        r -= alpha * ap
        rr_next = float(r @ r)
        p = r + (rr_next / rr) * p
        rr = rr_next
    return x
