from __future__ import annotations

from copy import deepcopy
from typing import Generator

import pandas as pd

from .cluster import Cluster, Portfolio


def risk_parity(root: Cluster, cov: pd.DataFrame) -> Cluster:
    """compute a cluster"""
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


def _parity(cluster, cov) -> Cluster:
    """
    Given two clusters compute in a bottom-up approach their parent.

    :param cluster: left cluster
    :param cov: covariance matrix. Will pick the correct sub-matrix

    """
    # split is s.t. v_left * alpha_left == v_right * alpha_right and alpha + beta = 1
    v_left = cluster.left.portfolio.variance(cov)
    v_right = cluster.right.portfolio.variance(cov)

    alpha_left, alpha_right = v_right / (v_left + v_right), v_left / (v_left + v_right)

    # assets in the cluster are the assets of the left and right cluster
    # further downstream
    assets = {
        **(alpha_left * cluster.left.portfolio.weights).to_dict(),
        **(alpha_right * cluster.right.portfolio.weights).to_dict(),
    }

    for asset, weight in assets.items():
        cluster.portfolio[asset] = weight

    return cluster


def one_over_n(dendrogram) -> Generator[tuple[int, Portfolio]]:
    root = dendrogram.root
    assets = dendrogram.assets

    w = 1

    for n, level in enumerate(root.levels):
        for node in level:
            for leaf in node.leaves:
                root.portfolio[assets[leaf.value]] = w / node.leaf_count

        w *= 0.5
        yield n, deepcopy(root.portfolio)
