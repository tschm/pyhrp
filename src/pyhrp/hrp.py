"""Hierarchical Risk Parity (HRP) allocation entry points.

This module owns the top-level allocation functions:
- hrp: Compute HRP portfolio weights from prices
- schur_hrp: Compute Schur Complementary Allocation weights from prices

For backward compatibility the supporting building blocks stay importable from
``pyhrp.hrp``, but they are owned (and exported) by the modules that define them:
- build_tree, Dendrogram: see :mod:`pyhrp.dendrogram`
- compute_returns, compute_cov, compute_corr: see :mod:`pyhrp.covariance`
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .algos import risk_parity, schur_risk_parity
from .cluster import Cluster
from .covariance import check_finite_matrix, compute_returns
from .covariance import compute_corr as compute_corr
from .covariance import compute_cov as compute_cov
from .dendrogram import Dendrogram as Dendrogram
from .dendrogram import build_tree as build_tree

__all__ = ["hrp", "schur_hrp"]

Method = Literal["single", "complete", "average", "ward"]


def _prepare(
    prices: pl.DataFrame, node: Cluster | None, method: Method, bisection: bool
) -> tuple[pl.DataFrame, Cluster]:
    """Turn prices into the covariance matrix and the cluster tree the allocators need.

    The correlation matrix is only computed when no ``node`` is supplied, since it
    is used for nothing but building the tree.

    Args:
        prices (pl.DataFrame): Asset price time series (columns are assets, rows are dates)
        node (Cluster, optional): Root of a prebuilt cluster tree, used as-is if given
        method (Method): Linkage method passed to :func:`build_tree`
        bisection (bool): Whether to use bisection for tree construction

    Returns:
        tuple[pl.DataFrame, Cluster]: The (finite) covariance matrix and the tree root
    """
    returns = compute_returns(prices)
    cov = check_finite_matrix(compute_cov(returns), name="covariance matrix")
    if node is None:
        node = build_tree(compute_corr(returns), method=method, bisection=bisection).root
    return cov, node


def hrp(
    prices: pl.DataFrame,
    node: Cluster | None = None,
    method: Method = "ward",
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
    cov, node = _prepare(prices, node, method, bisection)
    return risk_parity(root=node, cov=cov)


def schur_hrp(
    prices: pl.DataFrame,
    node: Cluster | None = None,
    method: Method = "ward",
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
    cov, node = _prepare(prices, node, method, bisection)
    return schur_risk_parity(root=node, cov=cov, gamma=gamma)
