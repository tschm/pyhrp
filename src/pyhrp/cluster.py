from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from binarytree import Node


@dataclass(frozen=True)
class Asset:
    mu: float = None
    name: str = None

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Asset):
            return False
        return self.name == other.name

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Asset):
            return False
        return self.name < other.name


@dataclass
class Portfolio:
    # _variance: float = None
    _weights: dict[Asset, float] = field(default_factory=dict)

    @property
    def assets(self) -> list[Asset]:
        return list(self._weights.keys())

    def variance(self, cov: pd.DataFrame) -> float:
        c = cov[self.assets].loc[self.assets].values
        w = self.weights[self.assets].values
        return np.linalg.multi_dot((w, c, w))

    def __getitem__(self, item) -> float:
        return self._weights[item]

    def __setitem__(self, key, value: float):
        self._weights[key] = value

    @property
    def weights(self):
        """weight series"""
        return pd.Series(self._weights, name="Weights").sort_index()

    # @property
    def weights_vs_name(self, names):
        w = {asset.name: weight for asset, weight in self.weights.items()}
        return pd.Series({name: w[name] for name in names})

    def plot(self, names):
        # Plot the weights using pandas' built-in plotting, without needing to import matplotlib
        # a = self.weights_vs_name(names)
        a = self.weights_vs_name(names)

        ax = a.plot(kind="bar", color="skyblue")

        # Set x-axis labels and rotations
        ax.set_xticklabels(names, rotation=90, fontsize=8)
        return ax


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

        Note that the leaves method of the Node class implemented in BinaryTree
        is not respecting the 'correct' order of the nodes.
        """
        if self.is_leaf:
            return [self]
        else:
            return self.left.leaves + self.right.leaves
