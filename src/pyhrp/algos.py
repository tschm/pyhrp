from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cluster import Cluster


def risk_parity(root: Cluster, cov: pd.DataFrame) -> Cluster:
    """compute a cluster"""
    if root.is_leaf:
        # a node is a leaf if has no further relatives downstream.
        # no leaves, no branches, ...
        asset = cov.keys().to_list()[root.value]
        root.portfolio[asset] = 1.0
        root.portfolio.variance = cov[asset][asset]
        return root

    # drill down on the left
    root.left = risk_parity(root.left, cov)
    # drill down on the right
    root.right = risk_parity(root.right, cov)

    # combine left and right into a new cluster
    return _parity(root, cov=cov)


def _parity(cluster, cov) -> Cluster:
    """
    Given two clusters compute in a bottom-up approach their parent.

    :param cluster: left cluster
    :param cov: covariance matrix. Will pick the correct sub-matrix

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
    alpha_left, alpha_right = parity(cluster.left.portfolio.variance, cluster.right.portfolio.variance)

    # assets in the cluster are the assets of the left and right cluster
    # further downstream
    assets = {
        **(alpha_left * cluster.left.portfolio.weights).to_dict(),
        **(alpha_right * cluster.right.portfolio.weights).to_dict(),
    }

    weights = np.array(list(assets.values()))

    covariance = cov[assets.keys()].loc[assets.keys()]

    var = np.linalg.multi_dot((weights, covariance, weights))

    cluster.portfolio.variance = var
    for asset, weight in assets.items():
        cluster.portfolio[asset] = weight

    return cluster


def one_over_n(root: Cluster) -> dict[int, Any] | None:
    # print(root.levels)
    w = 1
    portfolios = {}
    for n, level in enumerate(root.levels):
        for node in level:
            for leaf in node.leaves:
                root.portfolio[leaf.value] = w / node.leaf_count

        portfolios[n] = root.portfolio.copy()
        w *= 0.5

    return portfolios
