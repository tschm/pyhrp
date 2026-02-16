"""Data structures for hierarchical risk parity portfolio optimization.

This module defines the core data structures used in the hierarchical risk parity algorithm:
- Portfolio: Manages a collection of asset weights (strings identify assets)
- Cluster: Represents a node in the hierarchical clustering tree
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .treelib import Node


@dataclass
class Portfolio:
    """Container for portfolio asset weights.

    This lightweight class stores and manipulates a mapping from asset names to
    their portfolio weights, and provides convenience helpers for analysis and
    visualization.

    Attributes:
        _weights (dict[str, float]): Internal mapping from asset symbol to weight.
    """

    _weights: dict[str, float] = field(default_factory=dict)

    @property
    def assets(self) -> list[str]:
        """List of asset names present in the portfolio.

        Returns:
            list[str]: Asset identifiers in insertion order (Python 3.7+ dict order).
        """
        return list(self._weights.keys())

    def variance(self, cov: pd.DataFrame) -> float:
        """Calculate the variance of the portfolio.

        Args:
            cov (pd.DataFrame): Covariance matrix

        Returns:
            float: Portfolio variance
        """
        c = cov[self.assets].loc[self.assets].values
        w = self.weights[self.assets].values
        return float(np.linalg.multi_dot((w, c, w)))

    def __getitem__(self, item: str) -> float:
        """Return the weight for a given asset.

        Args:
            item (str): Asset name/symbol.

        Returns:
            float: The weight associated with the asset.

        Raises:
            KeyError: If the asset is not present in the portfolio.
        """
        return self._weights[item]

    def __setitem__(self, key: str, value: float) -> None:
        """Set or update the weight for an asset.

        Args:
            key (str): Asset name/symbol.
            value (float): Portfolio weight for the asset.
        """
        self._weights[key] = value

    @property
    def weights(self) -> pd.Series:
        """Get all weights as a pandas Series.

        Returns:
            pd.Series: Series of weights indexed by assets
        """
        return pd.Series(self._weights, name="Weights").sort_index()

    def plot(self, names: list[str]) -> Axes:
        """Plot the portfolio weights.

        Args:
            names (list[str]): List of asset names to include in the plot

        Returns:
            matplotlib.axes.Axes: The plot axes
        """
        a = self.weights.loc[names]

        ax = a.plot(kind="bar", color="skyblue")

        # Set x-axis labels and rotations
        ax.set_xticklabels(names, rotation=90, fontsize=8)
        return ax


class Cluster(Node):
    """Represents a cluster in the hierarchical clustering tree.

    Clusters are the nodes of the graphs we build.
    Each cluster is aware of the left and the right cluster
    it is connecting to. Each cluster also has an associated portfolio.

    Attributes:
        portfolio (Portfolio): The portfolio associated with this cluster
    """

    def __init__(self, value: int, left: Cluster | None = None, right: Cluster | None = None, **kwargs: Any) -> None:
        """Initialize a new Cluster.

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
        """Check if this cluster is a leaf node (has no children).

        Returns:
            bool: True if this is a leaf node, False otherwise
        """
        return self.left is None and self.right is None

    @property
    def leaves(self) -> list[Cluster]:
        """Get all reachable leaf nodes in the correct order.

        Note that the leaves method of the Node class implemented in BinaryTree
        is not respecting the 'correct' order of the nodes.

        Returns:
            list[Cluster]: List of all leaf nodes reachable from this cluster
        """
        if self.is_leaf:
            return [self]
        else:
            if self.left is None:
                raise ValueError("Expected left child to exist for non-leaf cluster")  # noqa: TRY003
            if self.right is None:
                raise ValueError("Expected right child to exist for non-leaf cluster")  # noqa: TRY003
            left_leaves: list[Cluster] = self.left.leaves  # type: ignore[assignment]
            right_leaves: list[Cluster] = self.right.leaves  # type: ignore[assignment]
            return left_leaves + right_leaves
