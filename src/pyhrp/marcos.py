"""Replicate the implementation of HRP by Marcos Lopez de Prado using this package

The original implementation by Marcos Lopez de Prado is using recursive bisection
on a ranked list of columns of the covariance matrix
To get to this list Lopez de Prado is using a matrix quasi-diagonalization
induced by the order (from left to right) of the dendrogram.
Based on that we build a tree reflecting the recursive bisection.
With that tree and the covariance matrix we go back to the hrp algorithm"""

from __future__ import annotations

import pandas as pd

from .cluster import Cluster, build_cluster
from .hrp import root


def marcos(prices: pd.DataFrame, node=None, method=None) -> Cluster:
    """The algorithm as implemented in the book by Marcos Lopez de Prado"""
    # convert into returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # compute covariance matrix and correlation matrices (both as DataFrames)
    cov, cor = returns.cov(), returns.corr()

    # Compute the root node of the tree
    method = method or "single"
    # you can either use a pre-computed node or you can construct a new dendrogram
    _root = node or root(cor.values, method=method, bisection=True).root

    # build the cluster
    return build_cluster(node=_root, cov=cov)
