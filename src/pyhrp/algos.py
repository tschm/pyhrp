"""Portfolio optimization algorithms for hierarchical risk parity.

This module implements various portfolio optimization algorithms:
- risk_parity: The main hierarchical risk parity algorithm
- one_over_n: A simple equal-weight allocation strategy
"""

from __future__ import annotations

from collections.abc import Generator
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
    if root.is_leaf:
        # a node is a leaf if has no further relatives downstream.
        asset = cov.columns[int(root.value)]
        root.portfolio[asset] = 1.0
        return root

    # drill down on the left
    if not isinstance(root.left, Cluster):
        raise TypeError("Expected left child to be a Cluster")  # noqa: TRY003
    if not isinstance(root.right, Cluster):
        raise TypeError("Expected right child to be a Cluster")  # noqa: TRY003
    root.left = risk_parity(root.left, cov)
    # drill down on the right
    root.right = risk_parity(root.right, cov)

    # combine left and right into a new cluster
    return _parity(root, cov=cov)


def _parity(cluster: Cluster, cov: pl.DataFrame) -> Cluster:
    """Compute risk parity weights for a parent cluster from its children.

    This function implements the core risk parity principle: allocating weights
    inversely proportional to risk, so that each sub-portfolio contributes
    equally to the total portfolio risk.

    Args:
        cluster (Cluster): The parent cluster with left and right children
        cov (pl.DataFrame): Covariance matrix of asset returns

    Returns:
        Cluster: The parent cluster with portfolio weights assigned
    """
    # Calculate variances of left and right sub-portfolios
    if not isinstance(cluster.left, Cluster):
        raise TypeError("Expected left child to be a Cluster")  # noqa: TRY003
    if not isinstance(cluster.right, Cluster):
        raise TypeError("Expected right child to be a Cluster")  # noqa: TRY003
    v_left = cluster.left.portfolio.variance(cov)
    v_right = cluster.right.portfolio.variance(cov)

    # Calculate weights inversely proportional to risk
    # such that v_left * alpha_left == v_right * alpha_right and alpha_left + alpha_right = 1
    alpha_left = v_right / (v_left + v_right)
    alpha_right = v_left / (v_left + v_right)

    # Combine assets from left and right clusters with their adjusted weights
    assets = {
        **{k: alpha_left * v for k, v in cluster.left.portfolio.weights.items()},
        **{k: alpha_right * v for k, v in cluster.right.portfolio.weights.items()},
    }

    # Assign the combined weights to the parent cluster's portfolio
    for asset, weight in assets.items():
        cluster.portfolio[asset] = weight

    return cluster


def schur_risk_parity(root: Cluster, cov: pl.DataFrame, gamma: float = 0.5) -> Cluster:
    """Compute Schur Complementary Allocation weights for a cluster tree.

    An extension of HRP introduced by Peter Cotton (arXiv:2411.05807) that augments
    sub-covariance matrices with off-diagonal block information via Schur complements.
    At gamma=0 this recovers standard HRP; at gamma=1 it recovers the minimum-variance
    portfolio through the same recursive structure.

    Args:
        root (Cluster): The root node of the cluster tree
        cov (pl.DataFrame): Covariance matrix of asset returns
        gamma (float): Interpolation parameter in [0, 1]. 0 = HRP, 1 = minimum variance.

    Returns:
        Cluster: The root node with portfolio weights assigned

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
    if root.is_leaf:
        asset = cov.columns[int(root.value)]
        root.portfolio[asset] = 1.0
        return root

    if not isinstance(root.left, Cluster):
        raise TypeError("Expected left child to be a Cluster")  # noqa: TRY003
    if not isinstance(root.right, Cluster):
        raise TypeError("Expected right child to be a Cluster")  # noqa: TRY003
    root.left = schur_risk_parity(root.left, cov, gamma)
    root.right = schur_risk_parity(root.right, cov, gamma)

    return _schur_parity(root, cov=cov, gamma=gamma)


def _schur_parity(cluster: Cluster, cov: pl.DataFrame, gamma: float) -> Cluster:
    """Compute Schur-augmented risk parity weights for a parent cluster.

    Replaces sub-covariance blocks a_mat and d_mat with Schur complements
    a_aug = a_mat - gamma * b_mat @ inv(d_mat) @ b_mat.T and
    d_aug = d_mat - gamma * b_mat.T @ inv(a_mat) @ b_mat before splitting risk.
    This incorporates cross-group covariance information that standard HRP discards.
    """
    if not isinstance(cluster.left, Cluster):
        raise TypeError("Expected left child to be a Cluster")  # noqa: TRY003
    if not isinstance(cluster.right, Cluster):
        raise TypeError("Expected right child to be a Cluster")  # noqa: TRY003

    left_assets = cluster.left.portfolio.assets
    right_assets = cluster.right.portfolio.assets

    cov_np = cov.to_numpy()
    cols = cov.columns
    li = [cols.index(a) for a in left_assets]
    ri = [cols.index(a) for a in right_assets]

    a_mat = cov_np[np.ix_(li, li)]
    b_mat = cov_np[np.ix_(li, ri)]
    d_mat = cov_np[np.ix_(ri, ri)]

    w_left = np.array([cluster.left.portfolio[a] for a in left_assets])
    w_right = np.array([cluster.right.portfolio[a] for a in right_assets])

    # Schur-augmented blocks: condition each group on the other
    a_aug = a_mat - gamma * (b_mat @ np.linalg.solve(d_mat, b_mat.T))
    d_aug = d_mat - gamma * (b_mat.T @ np.linalg.solve(a_mat, b_mat))

    v_left = float(w_left @ a_aug @ w_left)
    v_right = float(w_right @ d_aug @ w_right)

    alpha_left = v_right / (v_left + v_right)
    alpha_right = 1.0 - alpha_left

    for asset, weight in cluster.left.portfolio.weights.items():
        cluster.portfolio[asset] = alpha_left * weight
    for asset, weight in cluster.right.portfolio.weights.items():
        cluster.portfolio[asset] = alpha_right * weight

    return cluster


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
