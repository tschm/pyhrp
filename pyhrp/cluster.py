import numpy as np
import pandas as pd


class Cluster(object):
    def __init__(self, assets, variance, weights, left=None, right=None):
        assert len(assets) == len(weights)
        assert isinstance(weights, np.ndarray)
        assert np.all(weights > 0)
        assert variance >= 0

        if left is None:
            # if there is no left, there can't be a right
            assert right is None
        else:
            # left is not None, hence both left and right have to be clusters
            assert isinstance(left, Cluster)
            assert isinstance(right, Cluster)

        self.__assets = assets
        self.__variance = variance
        self.__weights = weights

        self.__left = left
        self.__right = right

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

    @property
    def weights_series(self):
        return pd.Series(index=self.__assets, data=self.__weights).sort_index()
