"""Portfolio optimization algorithms for hierarchical risk parity.

This module implements various portfolio optimization algorithms:
- risk_parity: The main hierarchical risk parity algorithm
- schur_risk_parity: Schur Complementary Allocation (Cotton, arXiv:2411.05807)
- one_over_n: A simple equal-weight allocation strategy
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .cluster import Cluster, Portfolio

if TYPE_CHECKING:
    from .hrp import Dendrogram

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
    cov_np = cov.to_numpy()
    index = {name: i for i, name in enumerate(cov.columns)}

    def combine(cluster: Cluster) -> Cluster:
        """Combine the child portfolios of a cluster via inverse-variance split."""
        left, right = _children(cluster)
        v_left = _block_variance(left.portfolio, cov_np, index)
        v_right = _block_variance(right.portfolio, cov_np, index)
        return _split(cluster, v_left, v_right)

    return _allocate(root, cov.columns, combine)


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

    cov_np = cov.to_numpy()
    index = {name: i for i, name in enumerate(cov.columns)}

    def combine(cluster: Cluster) -> Cluster:
        """Combine the child portfolios of a cluster via a Schur-augmented split."""
        left, right = _children(cluster)

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


def one_over_n(dendrogram: Dendrogram) -> Generator[tuple[int, Portfolio]]:
    """Generate portfolios using the 1/N (equal weight) strategy at each tree level.

    This function implements a hierarchical 1/N strategy where weights are
    distributed equally among assets within each cluster at each level of the tree.
    The weight assigned to each cluster decreases by half at each level.

    Args:
        dendrogram: A dendrogram object containing the hierarchical clustering tree
                   and the list of assets

    Yields:
        tuple[int, Portfolio]: A tuple containing the level number and the portfolio
                              at that level

    Examples:
        >>> import polars as pl
        >>> from pyhrp.hrp import build_tree
        >>> from pyhrp.algos import one_over_n
        >>> cor = pl.DataFrame({"A": [1.0, 0.3], "B": [0.3, 1.0]})
        >>> dg = build_tree(cor, method="ward")
        >>> levels = list(one_over_n(dg))
        >>> len(levels) > 0
        True
    """
    root = dendrogram.root
    assets = dendrogram.assets

    # Initial weight to distribute
    w: float = 1.0

    # Process each level of the tree
    for n, level in enumerate(root.levels):
        for node in level:
            # Distribute weight equally among all leaves in this node
            for leaf in node.leaves:
                root.portfolio[assets[leaf.value]] = w / node.leaf_count

        # Reduce weight for the next level
        w *= 0.5

        # Yield the current level number and a deep copy of the portfolio
        yield n, deepcopy(root.portfolio)
