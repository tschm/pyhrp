"""risk parity for clusters"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hrp import root


def hrp(prices, node=None, method="ward", bisection=False) -> Cluster:
    """
    Computes the root node for the hierarchical risk parity portfolio
    :param cov: This is the covariance matrix that shall be used
    :param node: Optional. This is the rootnode of the graph describing the dendrogram
    :return: the root cluster of the risk parity portfolio
    """
    returns = prices.pct_change().dropna(axis=0, how="all")
    cov, cor = returns.cov(), returns.corr()
    node = node or root(cor.values, method=method, bisection=bisection).root

    return build_cluster(node, cov)


def build_cluster(node, cov) -> Cluster:
    """compute a cluster"""
    if node.is_leaf():
        # a node is a leaf if has no further relatives downstream.
        # no leaves, no branches, ...
        asset = cov.keys().to_list()[node.id]
        return Cluster(assets={asset: 1.0}, variance=cov[asset][asset])

    # drill down on the left
    cluster_left = build_cluster(node.left, cov)
    # drill down on the right
    cluster_right = build_cluster(node.right, cov)
    # combine left and right into a new cluster
    return risk_parity(cluster_left, cluster_right, cov=cov)


def risk_parity(cluster_left, cluster_right, cov) -> Cluster:
    """
    Given two clusters compute in a bottom-up approach their parent.

    :param cluster_left: left cluster
    :param cluster_right: right cluster
    :param cov: (global) covariance matrix. Will pick the correct sub-matrix

    """

    # combine two clusters

    def parity(v_left, v_right):
        """
        Compute the weights for a risk parity portfolio of two assets
        :param v_left: Variance of the "left" portfolio
        :param v_right: Variance of the "right" portfolio
        :return: w, 1-w the weights for the left and the right portfolio.
                 It is w*v_left == (1-w)*v_right hence w = v_right / (v_right + v_left)
        """
        return v_right / (v_left + v_right), v_left / (v_left + v_right)

    # split is s.t. v_left * alpha_left == v_right * alpha_right and alpha + beta = 1
    alpha_left, alpha_right = parity(cluster_left.variance, cluster_right.variance)

    # assets in the cluster are the assets of the left and right cluster
    # further downstream
    assets = {
        **(alpha_left * cluster_left.weights).to_dict(),
        **(alpha_right * cluster_right.weights).to_dict(),
    }

    weights = np.array(list(assets.values()))
    covariance = cov[assets.keys()].loc[assets.keys()]

    var = np.linalg.multi_dot((weights, covariance, weights))

    return Cluster(
        assets=assets,
        variance=var,
        left=cluster_left,
        right=cluster_right,
    )


@dataclass(frozen=True)
class Cluster:
    """
    Clusters are the nodes of the graphs we build.
    Each cluster is aware of the left and the right cluster
    it is connecting to.
    """

    assets: dict[str, float]
    variance: float
    left: Cluster = None
    right: Cluster = None

    def __post_init__(self):
        """check input"""

        if self.variance <= 0:
            raise AssertionError

    def is_leaf(self):
        """true if this cluster is a leaf, e.g. no clusters follow downstream"""
        return self.left is None and self.right is None

    @property
    def weights(self):
        """weight series"""
        return pd.Series(self.assets, name="Weights").sort_index()
