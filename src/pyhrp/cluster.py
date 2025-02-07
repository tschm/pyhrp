"""risk parity for clusters"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def risk_parity(cluster_left, cluster_right, cov):
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

    if not set(cluster_left.assets).isdisjoint(set(cluster_right.assets)):
        raise AssertionError

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
        right=cluster_right,  # , node=node
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
    left: object = None
    right: object = None

    def __post_init__(self):
        """check input"""

        if self.variance <= 0:
            raise AssertionError
        if self.left is None:
            # if there is no left, there can't be a right
            if self.right is not None:
                raise AssertionError
        else:
            # left is not None, hence both left and right have to be clusters
            if not isinstance(self.left, Cluster):
                raise AssertionError
            if not isinstance(self.right, Cluster):
                raise AssertionError
            if not set(self.left.assets.keys()).isdisjoint(set(self.right.assets.keys())):
                raise AssertionError

    def is_leaf(self):
        """true if this cluster is a leaf, e.g. no clusters follow downstream"""
        return self.left is None and self.right is None

    @property
    def weights(self):
        """weight series"""
        return pd.Series(self.assets, name="Weights").sort_index()
