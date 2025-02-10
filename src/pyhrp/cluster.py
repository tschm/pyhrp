from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch


class Cluster(sch.ClusterNode):
    """
    Clusters are the nodes of the graphs we build.
    Each cluster is aware of the left and the right cluster
    it is connecting to.
    """

    def __init__(self, id, left: Cluster = None, right: Cluster = None, **kwargs):
        super().__init__(id, left, right, **kwargs)

        self.__assets = {}
        self.__variance = None

    # Property for 'variance'
    @property
    def variance(self):
        """Getter for variance."""
        return self.__variance

    # Setter for 'variance'
    @variance.setter
    def variance(self, value):
        """Setter for variance. It allows setting the value."""
        # You can add validation or logic here
        if value < 0:
            raise ValueError("Variance must be non-negative!")
        self.__variance = value

    def __getitem__(self, item):
        return self.__assets[item]

    def __setitem__(self, key, value):
        self.__assets[key] = value

    @property
    def weights(self):
        """weight series"""
        return pd.Series(self.__assets, name="Weights").sort_index()

    def risk_parity(self, cov) -> Cluster:
        """compute a cluster"""
        if self.is_leaf():
            # a node is a leaf if has no further relatives downstream.
            # no leaves, no branches, ...
            asset = cov.keys().to_list()[self.id]
            self[asset] = 1.0
            self.variance = cov[asset][asset]
            return self

        # drill down on the left
        self.left = self.left.risk_parity(cov)
        # drill down on the right
        self.right = self.right.risk_parity(cov)

        # combine left and right into a new cluster
        return self._parity(cov=cov)

    def _parity(self, cov) -> Cluster:
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
        alpha_left, alpha_right = parity(self.left.variance, self.right.variance)

        # assets in the cluster are the assets of the left and right cluster
        # further downstream
        assets = {
            **(alpha_left * self.left.weights).to_dict(),
            **(alpha_right * self.right.weights).to_dict(),
        }

        weights = np.array(list(assets.values()))

        covariance = cov[assets.keys()].loc[assets.keys()]

        var = np.linalg.multi_dot((weights, covariance, weights))

        self.variance = var
        for asset, weight in assets.items():
            self[asset] = weight

        return self
