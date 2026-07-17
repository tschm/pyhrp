"""Portfolio optimization algorithms for hierarchical risk parity.

This module implements various portfolio optimization algorithms:
- risk_parity: The main hierarchical risk parity algorithm
- schur_risk_parity: Schur Complementary Allocation (Cotton, arXiv:2411.05807)
- one_over_n: A simple equal-weight allocation strategy

Allocator contract
------------------
All three allocators take the same inputs — a ``Cluster`` tree (``root``) plus
the asset names — and never mutate the tree they are given: weights are always
rebuilt from scratch, so every allocator is idempotent and a tree can be reused.
``risk_parity`` and ``schur_risk_parity`` share the recursive ``_allocate_with``
scaffolding and each return the fully weighted root ``Cluster``. ``one_over_n``
intentionally differs in its *output*: it is a generator that yields the
equal-weight portfolio one tree level at a time (see its docstring), because its
purpose is to expose the allocation as it deepens rather than a single final
result.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from copy import deepcopy

import numpy as np
import polars as pl

from .cluster import Cluster, Portfolio

__all__ = ["one_over_n", "risk_parity", "schur_risk_parity"]


def risk_parity(root: Cluster, cov: pl.DataFrame) -> Cluster:
    """Compute hierarchical risk parity weights for a cluster tree.

    This is the main algorithm for hierarchical risk parity. It recursively
    traverses the cluster tree and assigns weights to each node based on
    the risk parity principle.

    Note:
        The tree is modified in place: the portfolio of every node is rebuilt
        from scratch, so the function is idempotent and a tree can be reused
        with a different covariance matrix.

    Args:
        root (Cluster): The root node of the cluster tree
        cov (pl.DataFrame): Covariance matrix of asset returns

    Returns:
        Cluster: The root node with portfolio weights assigned

    Examples:
        >>> import polars as pl
        >>> from pyhrp.cluster import Cluster
        >>> from pyhrp.algos import risk_parity
        >>> cov = pl.DataFrame({"A": [4.0, 0.0], "B": [0.0, 1.0]})
        >>> root = Cluster(2, left=Cluster(0), right=Cluster(1))
        >>> cluster = risk_parity(root=root, cov=cov)
        >>> round(cluster.portfolio["B"], 1)
        0.8
    """

    def node_variances(left: Cluster, right: Cluster, cov_np: np.ndarray, index: dict[str, int]) -> tuple[float, float]:
        """Plain block variance of each child sub-portfolio."""
        return (
            _block_variance(left.portfolio, cov_np, index),
            _block_variance(right.portfolio, cov_np, index),
        )

    return _allocate_with(root, cov, node_variances)


def schur_risk_parity(root: Cluster, cov: pl.DataFrame, gamma: float = 0.5) -> Cluster:
    """Compute Schur Complementary Allocation weights for a cluster tree.

    An extension of HRP introduced by Peter Cotton (arXiv:2411.05807) that augments
    sub-covariance matrices with off-diagonal block information via Schur complements.
    At gamma=0 this recovers standard HRP; at gamma=1 it recovers the minimum-variance
    portfolio through the same recursive structure.

    Note:
        The tree is modified in place: the portfolio of every node is rebuilt
        from scratch, so the function is idempotent and a tree can be reused
        with a different covariance matrix or gamma.

    Args:
        root (Cluster): The root node of the cluster tree
        cov (pl.DataFrame): Covariance matrix of asset returns
        gamma (float): Interpolation parameter in [0, 1]. 0 = HRP, 1 = minimum variance.

    Returns:
        Cluster: The root node with portfolio weights assigned

    Raises:
        ValueError: If gamma is outside the interval [0, 1].

    Examples:
        >>> import polars as pl
        >>> from pyhrp.cluster import Cluster
        >>> from pyhrp.algos import schur_risk_parity
        >>> cov = pl.DataFrame({"A": [4.0, 0.0], "B": [0.0, 1.0]})
        >>> root = Cluster(2, left=Cluster(0), right=Cluster(1))
        >>> cluster = schur_risk_parity(root=root, cov=cov, gamma=0.5)
        >>> round(cluster.portfolio["B"], 1)
        0.8
    """
    if not 0.0 <= gamma <= 1.0:
        msg = f"gamma must be in [0, 1], got {gamma}"
        raise ValueError(msg)

    def node_variances(left: Cluster, right: Cluster, cov_np: np.ndarray, index: dict[str, int]) -> tuple[float, float]:
        """Schur-augmented block variance of each child, conditioned on the other."""
        li = [index[a] for a in left.portfolio.assets]
        ri = [index[a] for a in right.portfolio.assets]

        a_mat = cov_np[np.ix_(li, li)]
        b_mat = cov_np[np.ix_(li, ri)]
        d_mat = cov_np[np.ix_(ri, ri)]

        w_left = np.array([left.portfolio[a] for a in left.portfolio.assets])
        w_right = np.array([right.portfolio[a] for a in right.portfolio.assets])

        # Schur-augmented blocks: condition each group on the other
        a_aug = a_mat - gamma * (b_mat @ _solve(d_mat, b_mat.T))
        d_aug = d_mat - gamma * (b_mat.T @ _solve(a_mat, b_mat))

        v_left = float(w_left @ a_aug @ w_left)
        v_right = float(w_right @ d_aug @ w_right)
        return v_left, v_right

    return _allocate_with(root, cov, node_variances)


# Given a node's two children (plus the precomputed covariance array and the
# column->row index), return the (v_left, v_right) risk pair used to split it.
NodeVariances = Callable[[Cluster, Cluster, np.ndarray, dict[str, int]], tuple[float, float]]


def _allocate_with(root: Cluster, cov: pl.DataFrame, node_variances: NodeVariances) -> Cluster:
    """Shared scaffolding for the recursive risk-based allocators.

    Builds the numpy covariance array and column index once, then walks the tree
    bottom-up, splitting each node's weight between its children inversely to the
    ``(v_left, v_right)`` pair supplied by ``node_variances``. The only thing that
    distinguishes ``risk_parity`` from ``schur_risk_parity`` is that per-node
    variance rule; everything else — the ``cov``/``index`` setup, the combine
    wrapper, and the rebuild-from-scratch traversal — lives here.

    Args:
        root (Cluster): The root node of the cluster tree.
        cov (pl.DataFrame): Covariance matrix of asset returns.
        node_variances (NodeVariances): Per-node rule mapping a node's left/right
            children (and the precomputed covariance array and column index) to
            the ``(v_left, v_right)`` risk pair used to split that node.

    Returns:
        Cluster: The root node with portfolio weights assigned.
    """
    cov_np = cov.to_numpy()
    index = {name: i for i, name in enumerate(cov.columns)}

    def combine(cluster: Cluster) -> Cluster:
        """Combine the child portfolios of a cluster via an inverse-variance split."""
        left, right = _children(cluster)
        v_left, v_right = node_variances(left, right, cov_np, index)
        return _split(cluster, v_left, v_right)

    return _allocate(root, cov.columns, combine)


def _allocate(root: Cluster, assets: list[str], combine: Callable[[Cluster], Cluster]) -> Cluster:
    """Traverse the tree bottom-up, assigning leaf portfolios and combining children.

    Every node's portfolio is replaced, never accumulated into, which keeps
    repeated allocations on the same tree idempotent.

    Args:
        root (Cluster): The (sub)tree to allocate weights for
        assets (list[str]): Asset names; a leaf's value indexes into this list
        combine (Callable[[Cluster], Cluster]): Combines the two child portfolios
            of a node into the node's own portfolio

    Returns:
        Cluster: The input node with portfolio weights assigned
    """
    if root.is_leaf:
        root.portfolio = Portfolio()
        root.portfolio[assets[int(root.value)]] = 1.0
        return root

    left, right = _children(root)
    root.left = _allocate(left, assets, combine)
    root.right = _allocate(right, assets, combine)
    return combine(root)


def _children(cluster: Cluster) -> tuple[Cluster, Cluster]:
    """Return the validated left and right children of a non-leaf cluster."""
    if not isinstance(cluster.left, Cluster):
        raise TypeError("Expected left child to be a Cluster")
    if not isinstance(cluster.right, Cluster):
        raise TypeError("Expected right child to be a Cluster")
    return cluster.left, cluster.right


def _block_variance(portfolio: Portfolio, cov_np: np.ndarray, index: dict[str, int]) -> float:
    """Compute the variance of a portfolio against a precomputed covariance array."""
    assets = portfolio.assets
    idx = [index[a] for a in assets]
    w = np.array([portfolio[a] for a in assets])
    return float(w @ cov_np[np.ix_(idx, idx)] @ w)


def _split(cluster: Cluster, v_left: float, v_right: float) -> Cluster:
    """Distribute weight between the two children inversely proportional to risk.

    The split satisfies v_left * alpha_left == v_right * alpha_right with
    alpha_left + alpha_right == 1. If both variances are zero (e.g. riskless
    sub-portfolios), the weight is split equally.

    Args:
        cluster (Cluster): The parent cluster with left and right children
        v_left (float): Variance of the left sub-portfolio
        v_right (float): Variance of the right sub-portfolio

    Returns:
        Cluster: The parent cluster with portfolio weights assigned
    """
    left, right = _children(cluster)
    total = v_left + v_right
    alpha_left = v_right / total if total > 0 else 0.5
    alpha_right = 1.0 - alpha_left

    cluster.portfolio = Portfolio()
    for asset, weight in left.portfolio.weights.items():
        cluster.portfolio[asset] = alpha_left * weight
    for asset, weight in right.portfolio.weights.items():
        cluster.portfolio[asset] = alpha_right * weight

    return cluster


def _solve(m: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve m @ x = b, falling back to least squares for singular matrices.

    Covariance blocks of collinear assets are singular; the minimum-norm
    least-squares solution keeps the Schur augmentation well-defined there.
    """
    try:
        return np.linalg.solve(m, b)
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.lstsq(m, b, rcond=None)[0])


def one_over_n(root: Cluster, assets: list[str]) -> Generator[tuple[int, Portfolio]]:
    """Generate 1/N (equal-weight) portfolios one tree level at a time.

    This implements a hierarchical 1/N strategy where weights are distributed
    equally among the leaves of each cluster, and the weight budget halves at
    each successive level of the tree.

    Unlike :func:`risk_parity` and :func:`schur_risk_parity` — which rebuild a
    single final allocation and return the root ``Cluster`` — this allocator is
    intentionally a **generator**: its purpose is to expose the equal-weight
    allocation as the tree deepens, yielding one portfolio per level. It shares
    the sibling input contract (a ``Cluster`` tree plus the asset names) and, like
    them, does not mutate the tree: weights accumulate in a local buffer, so a
    leaf that terminates at a shallow level keeps its weight in the deeper levels
    (each yielded portfolio is therefore a complete allocation over all assets),
    and re-running on the same tree yields an identical sequence.

    Args:
        root (Cluster): The root node of the cluster tree.
        assets (list[str]): Asset names; a leaf's value indexes into this list.

    Yields:
        tuple[int, Portfolio]: The level number and the (cumulative) equal-weight
        portfolio at that level.

    Examples:
        >>> import polars as pl
        >>> from pyhrp.hrp import build_tree
        >>> from pyhrp.algos import one_over_n
        >>> cor = pl.DataFrame({"A": [1.0, 0.3], "B": [0.3, 1.0]})
        >>> dg = build_tree(cor, method="ward")
        >>> levels = list(one_over_n(dg.root, dg.assets))
        >>> len(levels) > 0
        True
    """
    # Accumulate into a local buffer so the input tree is never mutated.
    portfolio = Portfolio()

    # Initial weight to distribute
    w: float = 1.0

    # Process each level of the tree
    for n, level in enumerate(root.levels):
        for node in level:
            # Distribute weight equally among all leaves in this node
            for leaf in node.leaves:
                portfolio[assets[leaf.value]] = w / node.leaf_count

        # Reduce weight for the next level
        w *= 0.5

        # Yield the current level number and a deep copy of the portfolio
        yield n, deepcopy(portfolio)
