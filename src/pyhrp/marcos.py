"""Replicate the implementation of HRP by Marcos Lopez de Prado using this package

The original implementation by Marcos Lopez de Prado is using recursive bisection
on a ranked list of columns of the covariance matrix
To get to this list Lopez de Prado is using a matrix quasi-diagonalization
induced by the order (from left to right) of the dendrogram.
Based on that we build a tree reflecting the recursive bisection.
With that tree and the covariance matrix we go back to the hrp algorithm"""

from __future__ import annotations

import pandas as pd
import scipy.cluster.hierarchy as sch

from .hrp import build_cluster, dist, linkage, tree


def bisection(ids):
    """
    Compute the graph underlying the recursive bisection of Marcos Lopez de Prado

    :param ids: A (ranked) set of indixes
    :return: The root ClusterNode of this tree
    """

    def split(ids):
        """split the vector ids in two parts, split in the middle"""
        if len(ids) < 2:
            raise AssertionError
        num = len(ids)
        return ids[: num // 2], ids[num // 2 :]

    if len(ids) < 1:
        raise AssertionError
    if len(ids) != len(set(ids)):
        raise AssertionError

    if len(ids) == 1:
        return sch.ClusterNode(id=ids[0])

    left, right = split(ids)
    return sch.ClusterNode(id=0, left=bisection(ids=left), right=bisection(ids=right))


def marcos(prices, node=None, method=None):
    """The algorithm as implemented in the book by Marcos Lopez de Prado"""

    # make sure the prices are a DataFrame
    if not isinstance(prices, pd.DataFrame):
        raise AssertionError

    # convert into returns
    returns = prices.pct_change().dropna(axis=0, how="all")

    # compute covariance matrix and correlation matrices (both as DataFrames)
    cov, cor = returns.cov(), returns.corr()

    # Compute the root node of the tree
    method = method or "ward"
    node = node or tree(linkage(dist(cor.values), method=method))

    # this is an interesting step
    ids = node.pre_order()
    # apply bisection, root is now a ClusterNode of the graph
    root = bisection(ids=ids)

    # It's not clear to me why Marcos is going down this route.
    # Rather than sticking with the graph computed above.
    return build_cluster(node=root, cov=cov)
