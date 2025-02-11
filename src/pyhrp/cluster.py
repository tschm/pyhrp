from __future__ import annotations

import pandas as pd
import scipy.cluster.hierarchy as sch


class Cluster(sch.ClusterNode):
    """
    Clusters are the nodes of the graphs we build.
    Each cluster is aware of the left and the right cluster
    it is connecting to.
    """

    def __init__(self, id, left: Cluster | None = None, right: Cluster | None = None, **kwargs):
        super().__init__(id, left, right, **kwargs)

        self.__assets = {}
        self.__variance = None

    def __hash__(self):
        return hash(self.id)  # Use an attribute of the object to generate the hash

    def __eq__(self, other: object) -> bool:
        """Equality based on cluster ID"""
        if not isinstance(other, Cluster):
            return False
        return self.id == other.id

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

    @property
    def leaves(self):
        """
        Give a set of all reachable leaf nodes.
        """
        if self.is_leaf():
            return {self}
        else:
            return set(self.left.leaves).union(self.right.leaves)
