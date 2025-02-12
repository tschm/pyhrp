from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

import pandas as pd
from binarytree import Node


@dataclass
class Portfolio:
    _variance: float = None
    _weights: dict[int, float] = field(default_factory=dict)

    @property
    def variance(self):
        return self._variance

    @variance.setter
    def variance(self, value: float):
        if value < 0:
            raise ValueError("Variance cannot be negative.")
        self._variance = value

    def __getitem__(self, item):
        return self._weights[item]

    def __setitem__(self, key, value):
        self._weights[key] = value

    @property
    def weights(self):
        """weight series"""
        return pd.Series(self._weights, name="Weights").sort_index()

    def copy(self):
        variance = self.variance
        weights = copy(self._weights)

        return Portfolio(variance, weights)


class Cluster(Node):
    """
    Clusters are the nodes of the graphs we build.
    Each cluster is aware of the left and the right cluster
    it is connecting to.
    """

    def __init__(self, value, left: Cluster | None = None, right: Cluster | None = None, **kwargs):
        super().__init__(value=value, left=left, right=right, **kwargs)
        self.portfolio = Portfolio()

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    @property
    def leaves(self):
        """
        Give a set of all reachable leaf nodes.
        """
        if self.is_leaf:
            return [self]
        else:
            return self.left.leaves + self.right.leaves
