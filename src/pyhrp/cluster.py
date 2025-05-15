"""
Data structures for hierarchical risk parity portfolio optimization.

This module defines the core data structures used in the hierarchical risk parity algorithm:
- Asset: Represents a financial asset in a portfolio
- Portfolio: Manages a collection of assets and their weights
- Cluster: Represents a node in the hierarchical clustering tree
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from binarytree import Node


@dataclass(frozen=True)
class Asset:
    """
    Represents a financial asset in a portfolio.

    Attributes:
        mu (float, optional): Expected return of the asset
        name (str, optional): Name of the asset
    """

    mu: float = None
    name: str = None

    def __hash__(self) -> int:
        """
        Hash function for Asset objects.

        Returns:
            int: Hash value based on the asset name
        """
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        """
        Equality comparison for Asset objects.

        Args:
            other (Any): Object to compare with

        Returns:
            bool: True if other is an Asset with the same name
        """
        if not isinstance(other, Asset):
            return False
        return self.name == other.name

    def __lt__(self, other: Any) -> bool:
        """
        Less than comparison for Asset objects.

        Args:
            other (Any): Object to compare with

        Returns:
            bool: True if this asset's name is lexicographically less than other's name
        """
        if not isinstance(other, Asset):
            return False
        return self.name < other.name


@dataclass
class Portfolio:
    _weights: dict[Asset, float] = field(default_factory=dict)

    @property
    def assets(self) -> list[Asset]:
        """
        Get all assets in the portfolio.

        Returns:
            list[Asset]: List of assets in the portfolio
        """
        return list(self._weights.keys())

    def variance(self, cov: pd.DataFrame) -> float:
        """
        Calculate the variance of the portfolio.

        Args:
            cov (pd.DataFrame): Covariance matrix

        Returns:
            float: Portfolio variance
        """
        c = cov[self.assets].loc[self.assets].values
        w = self.weights[self.assets].values
        return np.linalg.multi_dot((w, c, w))

    def __getitem__(self, item: Asset) -> float:
        """
        Get the weight of an asset.

        Args:
            item (Asset): The asset to get the weight for

        Returns:
            float: The weight of the asset
        """
        return self._weights[item]

    def __setitem__(self, key: Asset, value: float) -> None:
        """
        Set the weight of an asset.

        Args:
            key (Asset): The asset to set the weight for
            value (float): The weight to set
        """
        self._weights[key] = value

    @property
    def weights(self) -> pd.Series:
        """
        Get all weights as a pandas Series.

        Returns:
            pd.Series: Series of weights indexed by assets
        """
        return pd.Series(self._weights, name="Weights").sort_index()

    def weights_by_name(self, names: list[str]) -> pd.Series:
        """
        Get weights as a pandas Series indexed by asset names.

        Args:
            names (list[str]): List of asset names to include

        Returns:
            pd.Series: Series of weights indexed by asset names
        """
        w = {asset.name: weight for asset, weight in self.weights.items()}
        return pd.Series({name: w[name] for name in names})

    def plot(self, names: list[str]):
        """
        Plot the portfolio weights.

        Args:
            names (list[str]): List of asset names to include in the plot

        Returns:
            matplotlib.axes.Axes: The plot axes
        """
        a = self.weights_by_name(names)

        ax = a.plot(kind="bar", color="skyblue")

        # Set x-axis labels and rotations
        ax.set_xticklabels(names, rotation=90, fontsize=8)
        return ax


class Cluster(Node):
    """
    Represents a cluster in the hierarchical clustering tree.

    Clusters are the nodes of the graphs we build.
    Each cluster is aware of the left and the right cluster
    it is connecting to. Each cluster also has an associated portfolio.

    Attributes:
        portfolio (Portfolio): The portfolio associated with this cluster
    """

    def __init__(self, value: int, left: Cluster | None = None, right: Cluster | None = None, **kwargs):
        """
        Initialize a new Cluster.

        Args:
            value (int): The identifier for this cluster
            left (Cluster, optional): The left child cluster
            right (Cluster, optional): The right child cluster
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(value=value, left=left, right=right, **kwargs)
        self.portfolio = Portfolio()

    @property
    def is_leaf(self) -> bool:
        """
        Check if this cluster is a leaf node (has no children).

        Returns:
            bool: True if this is a leaf node, False otherwise
        """
        return self.left is None and self.right is None

    @property
    def leaves(self) -> list[Cluster]:
        """
        Get all reachable leaf nodes in the correct order.

        Note that the leaves method of the Node class implemented in BinaryTree
        is not respecting the 'correct' order of the nodes.

        Returns:
            list[Cluster]: List of all leaf nodes reachable from this cluster
        """
        if self.is_leaf:
            return [self]
        else:
            return self.left.leaves + self.right.leaves
