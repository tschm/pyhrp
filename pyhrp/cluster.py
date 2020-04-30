import numpy as np
import pandas as pd


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

    # assets in the cluster are the assets of the left and right cluster further downstream
    assets = {**(alpha_left * cluster_left.weights).to_dict(), **(alpha_right * cluster_right.weights).to_dict()}

    w = np.array(list(assets.values()))
    c = cov[assets.keys()].loc[assets.keys()]

    var = np.linalg.multi_dot((w, c, w))

    return Cluster(assets=assets, variance=var, left=cluster_left, right=cluster_right)


class Cluster(object):
    def __init__(self, assets, variance, left=None, right=None):
        w = np.array(list(assets.values()))

        assert np.all(w > 0)
        assert variance >= 0

        # test that the weights are close to 1.0
        assert np.isclose(np.sum(w), 1.0)

        self.__assets = assets
        self.__variance = variance
        self.__left = left
        self.__right = right

        if left is None:
            # if there is no left, there can't be a right
            assert right is None
        else:
            # left is not None, hence both left and right have to be clusters
            assert isinstance(left, Cluster)
            assert isinstance(right, Cluster)
            assert set(left.assets.keys()).isdisjoint(set(right.assets.keys()))

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

    @property
    def weights(self):
        return pd.Series(self.assets, name="Weights").sort_index()
