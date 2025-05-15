"""
Portfolio optimization algorithms for hierarchical risk parity.

This module implements various portfolio optimization algorithms:
- risk_parity: The main hierarchical risk parity algorithm
- one_over_n: A simple equal-weight allocation strategy
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Generator

import pandas as pd

from .cluster import Cluster, Portfolio


def risk_parity(root: Cluster, cov: pd.DataFrame) -> Cluster:
    """
    Compute hierarchical risk parity weights for a cluster tree.

    This is the main algorithm for hierarchical risk parity. It recursively
    traverses the cluster tree and assigns weights to each node based on
    the risk parity principle.

    Args:
        root (Cluster): The root node of the cluster tree
        cov (pd.DataFrame): Covariance matrix of asset returns

    Returns:
        Cluster: The root node with portfolio weights assigned
    """
    if root.is_leaf:
        # a node is a leaf if has no further relatives downstream.
        asset = cov.keys().to_list()[root.value]
        root.portfolio[asset] = 1.0
        return root

    # drill down on the left
    root.left = risk_parity(root.left, cov)
    # drill down on the right
    root.right = risk_parity(root.right, cov)

    # combine left and right into a new cluster
    return _parity(root, cov=cov)


def _parity(cluster: Cluster, cov: pd.DataFrame) -> Cluster:
    """
    Compute risk parity weights for a parent cluster from its children.

    This function implements the core risk parity principle: allocating weights
    inversely proportional to risk, so that each sub-portfolio contributes
    equally to the total portfolio risk.

    Args:
        cluster (Cluster): The parent cluster with left and right children
        cov (pd.DataFrame): Covariance matrix of asset returns

    Returns:
        Cluster: The parent cluster with portfolio weights assigned
    """
    # Calculate variances of left and right sub-portfolios
    v_left = cluster.left.portfolio.variance(cov)
    v_right = cluster.right.portfolio.variance(cov)

    # Calculate weights inversely proportional to risk
    # such that v_left * alpha_left == v_right * alpha_right and alpha_left + alpha_right = 1
    alpha_left = v_right / (v_left + v_right)
    alpha_right = v_left / (v_left + v_right)

    # Combine assets from left and right clusters with their adjusted weights
    assets = {
        **(alpha_left * cluster.left.portfolio.weights).to_dict(),
        **(alpha_right * cluster.right.portfolio.weights).to_dict(),
    }

    # Assign the combined weights to the parent cluster's portfolio
    for asset, weight in assets.items():
        cluster.portfolio[asset] = weight

    return cluster


def one_over_n(dendrogram: Any) -> Generator[tuple[int, Portfolio]]:
    """
    Generate portfolios using the 1/N (equal weight) strategy at each tree level.

    This function implements a hierarchical 1/N strategy where weights are
    distributed equally among assets within each cluster at each level of the tree.
    The weight assigned to each cluster decreases by half at each level.

    Args:
        dendrogram: A dendrogram object containing the hierarchical clustering tree
                   and the list of assets

    Yields:
        tuple[int, Portfolio]: A tuple containing the level number and the portfolio
                              at that level
    """
    root = dendrogram.root
    assets = dendrogram.assets

    # Initial weight to distribute
    w = 1

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
