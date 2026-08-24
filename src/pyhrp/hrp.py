"""Hierarchical Risk Parity (HRP) allocation entry points.

This module exposes the top-level allocation functions and re-exports the
supporting building blocks so the public ``pyhrp.hrp`` API is unchanged:
- hrp: Compute HRP portfolio weights from prices
- schur_hrp: Compute Schur Complementary Allocation weights from prices
- build_tree: Build a hierarchical cluster tree (see :mod:`pyhrp.dendrogram`)
- compute_returns: Simple returns from prices (see :mod:`pyhrp.covariance`)
- compute_cov / compute_corr: Second-moment estimators (see :mod:`pyhrp.covariance`)
- Dendrogram: Clustering result container (see :mod:`pyhrp.dendrogram`)
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .algos import risk_parity, schur_risk_parity
from .cluster import Cluster
from .covariance import check_finite_matrix, compute_corr, compute_cov, compute_returns
from .dendrogram import Dendrogram, build_tree

__all__ = [
    "Dendrogram",
    "build_tree",
    "check_finite_matrix",
    "compute_corr",
    "compute_cov",
    "compute_returns",
    "hrp",
    "schur_hrp",
]


def hrp(
    prices: pl.DataFrame,
    node: Cluster | None = None,
    method: Literal["single", "complete", "average", "ward"] = "ward",
    bisection: bool = False,
) -> Cluster:
    """Compute the hierarchical risk parity portfolio weights.

    This is the main entry point for the HRP algorithm. It calculates returns from prices,
    builds a hierarchical clustering tree if not provided, and applies risk parity weights.

    Note:
        A ``node`` passed in is weighted in place and returned; see the allocator
        contract in :mod:`pyhrp.algos`. Passing ``dendrogram.root`` therefore
        rewrites the portfolios inside that ``Dendrogram``, even though the
        ``Dendrogram`` itself is a frozen dataclass.

    Args:
        prices (pl.DataFrame): Asset price time series (columns are assets, rows are dates)
        node (Cluster, optional): Root node of the hierarchical clustering tree.
            If None, a tree will be built from the correlation matrix.
        method (Literal["single", "complete", "average", "ward"]): Linkage method to use for distance calculation
            - "single": minimum distance between points (nearest neighbor)
            - "complete": maximum distance between points (furthest neighbor)
            - "average": average distance between all points
            - "ward": Ward variance minimization
        bisection (bool): Whether to use bisection method for tree construction

    Returns:
        Cluster: The root cluster with portfolio weights assigned according to HRP
    """
    returns = compute_returns(prices)
    cov = check_finite_matrix(compute_cov(returns), name="covariance matrix")
    cor = compute_corr(returns)
    node = node or build_tree(cor, method=method, bisection=bisection).root

    return risk_parity(root=node, cov=cov)


def schur_hrp(
    prices: pl.DataFrame,
    node: Cluster | None = None,
    method: Literal["single", "complete", "average", "ward"] = "ward",
    bisection: bool = False,
    gamma: float = 0.5,
) -> Cluster:
    """Compute Schur Complementary Allocation portfolio weights.

    Extends HRP by augmenting each sub-covariance block with off-diagonal information
    via Schur complements before splitting risk between clusters. Introduced by Peter Cotton
    (arXiv:2411.05807). At gamma=0 this is identical to HRP; at gamma=1 it recovers the
    global minimum-variance portfolio through the same recursive hierarchy.

    Note:
        A ``node`` passed in is weighted in place and returned; see the allocator
        contract in :mod:`pyhrp.algos`. Passing ``dendrogram.root`` therefore
        rewrites the portfolios inside that ``Dendrogram``, even though the
        ``Dendrogram`` itself is a frozen dataclass.

    Args:
        prices (pl.DataFrame): Asset price time series (columns are assets, rows are dates)
        node (Cluster, optional): Root node of the hierarchical clustering tree.
            If None, a tree will be built from the correlation matrix.
        method (Literal["single", "complete", "average", "ward"]): Linkage method for clustering
        bisection (bool): Whether to use bisection method for tree construction
        gamma (float): Schur interpolation parameter in [0, 1].
            0 recovers standard HRP; 1 recovers minimum-variance portfolio.

    Returns:
        Cluster: The root cluster with portfolio weights assigned
    """
    returns = compute_returns(prices)
    cov = check_finite_matrix(compute_cov(returns), name="covariance matrix")
    cor = compute_corr(returns)
    node = node or build_tree(cor, method=method, bisection=bisection).root

    return schur_risk_parity(root=node, cov=cov, gamma=gamma)
