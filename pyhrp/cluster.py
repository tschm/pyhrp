import numpy as np
import pandas as pd
from pyhrp.linalg import variance, sub


def risk_parity(cluster_left, cluster_right, cov):
    # combine two clusters

    def rp(v_left, v_right):
        """
        Compute the weights for a risk parity portfolio of two assets
        :param v_left: Variance of the "left" portfolio
        :param v_right: Variance of the "right" portfolio
        :return: w, 1-w the weights for the left and the right portfolio. It is w*v_left == (1-w)*v_right and hence w = v_right / (v_right + v_left)
        """
        return v_right / (v_left + v_right), v_left / (v_left + v_right)

    assert set(cluster_left.assets).isdisjoint(set(cluster_right.assets))

    # the split is such that v_left * alpha_left == v_right * alpha_right and alpha + beta = 1
    alpha_left, alpha_right = rp(cluster_left.variance, cluster_right.variance)

    # update the weights
    weights = np.concatenate([alpha_left * cluster_left.weights, alpha_right * cluster_right.weights])

    # assets in the cluster are the assets of the left and right cluster further downstream
    assets = cluster_left.assets + cluster_right.assets

    var = variance(w=weights, cov=sub(cov, idx=assets))

    return Cluster(assets=assets, variance=var, weights=weights, left=cluster_left, right=cluster_right)


class Cluster(object):
    def __init__(self, assets, variance, weights, left=None, right=None):
        assert len(assets) == len(weights)
        assert isinstance(weights, np.ndarray)
        assert np.all(weights > 0)
        assert variance >= 0
        # test that the weights are close to 1.0
        assert np.isclose(np.sum(weights), 1.0)

        self.__assets = assets
        self.__variance = variance
        self.__weights = weights

        self.__left = left
        self.__right = right

        if left is None:
            # if there is no left, there can't be a right
            assert right is None
        else:
            # left is not None, hence both left and right have to be clusters
            assert isinstance(left, Cluster)
            assert isinstance(right, Cluster)

            assert left.assets + right.assets == self.__assets
            assert set(left.assets).isdisjoint(set(right.assets))

    @property
    def weights(self):
        return self.__weights

    @property
    def variance(self):
        return self.__variance

    @property
    def assets(self):
        return self.__assets

    @property
    def left(self):
        return self.__left

    @property
    def right(self):
        return self.__right

    def is_leaf(self):
        return self.left is None and self.right is None

    def weights_series(self, index=None):
        a = pd.Series(index=self.__assets, data=self.__weights, name="Weights")
        a.index.name = "Position"

        if index is not None:
            a.rename(lambda x: index[x], axis="index", inplace=True)
            a.index.name = "Asset"

        return a.sort_index()
