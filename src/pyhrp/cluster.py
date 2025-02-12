from __future__ import annotations

import pandas as pd
from binarytree import Node


class Cluster(Node):
    """
    Clusters are the nodes of the graphs we build.
    Each cluster is aware of the left and the right cluster
    it is connecting to.
    """

    def __init__(self, id, left: Cluster | None = None, right: Cluster | None = None, **kwargs):
        super().__init__(value=id, left=left, right=right, **kwargs)

        self.__assets = {}
        self.__variance = None

    @property
    def id(self):
        return self.value

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    # def __hash__(self):
    #    return hash(self.id)  # Use an attribute of the object to generate the hash

    # def __eq__(self, other: object) -> bool:
    #    """Equality based on cluster ID"""
    #    if not isinstance(other, Cluster):
    #        return False
    #    return self.id == other.id

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
        if self.is_leaf:
            return [self]
        else:
            return self.left.leaves + self.right.leaves
