"""Data structures for hierarchical risk parity portfolio optimization.

This module defines the core data structures used in the hierarchical risk parity algorithm:
- Portfolio: Manages a collection of asset weights (strings identify assets)
- Cluster: Represents a node in the hierarchical clustering tree
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .treelib import Node

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["Cluster", "Portfolio"]

# One non-leaf node of a laid-out tree: the node and the half-open slices
# [lo, mid) and [mid, hi) of the leaf order its left and right subtrees cover.
Span = tuple["Cluster", int, int, int]


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

    def variance(self, cov: pl.DataFrame) -> float:
        """Calculate the variance of the portfolio.

        Args:
            cov (pl.DataFrame): Covariance matrix where columns and rows correspond
                to assets in the same order as columns list.

        Returns:
            float: Portfolio variance
        """
        assets = self.assets
        index = {name: i for i, name in enumerate(cov.columns)}
        row_indices = [index[a] for a in assets]
        cov_matrix = cov.to_numpy()
        c = cov_matrix[np.ix_(row_indices, row_indices)]
        w = np.array([self._weights[a] for a in assets])
        return float(w @ c @ w)

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
    def weights(self) -> dict[str, float]:
        """Get all weights as a dict sorted alphabetically by asset name.

        Returns:
            dict[str, float]: Mapping from asset name to weight, sorted by name.
        """
        return dict(sorted(self._weights.items()))

    def plot(self, names: list[str]) -> go.Figure:
        """Plot the portfolio weights as a bar chart.

        Args:
            names (list[str]): List of asset names to include in the plot

        Returns:
            go.Figure: The plotly figure

        Note:
            The plotly dependency is imported lazily so importing the allocation
            core (which pulls in this module) stays plotly-free.
        """
        import plotly.graph_objects as go

        w = self.weights
        values = [w[n] for n in names]
        fig = go.Figure(go.Bar(x=names, y=values, marker_color="steelblue"))
        fig.update_layout(xaxis={"tickangle": -90})
        return fig


class Cluster(Node[int]):
    """Represents a cluster in the hierarchical clustering tree.

    Clusters are the nodes of the graphs we build.
    Each cluster is aware of the left and the right cluster
    it is connecting to. Each cluster also has an associated portfolio.

    Attributes:
        portfolio (Portfolio): The portfolio associated with this cluster

    After an allocation a node does not hold its portfolio, only its split: the
    share of its weight that goes to the left child. The portfolio is built from
    the splits below the node when it is first read, and then kept. Holding every
    node's portfolio would cost the sum of all cluster sizes, which is quadratic in
    the number of assets for the chain-like trees single linkage builds.
    """

    def __init__(self, value: int, left: Cluster | None = None, right: Cluster | None = None) -> None:
        """Initialize a new Cluster.

        Args:
            value (int): The identifier for this cluster
            left (Cluster, optional): The left child cluster
            right (Cluster, optional): The right child cluster
        """
        super().__init__(value=value, left=left, right=right)
        self._portfolio: Portfolio | None = Portfolio()
        self._share = 1.0
        self._assets: Sequence[str] = ()

    @property
    def portfolio(self) -> Portfolio:
        """The portfolio of this cluster, built from the allocation's splits on first access."""
        if self._portfolio is None:
            leaves, spans = self._layout()
            w = np.ones(len(leaves))
            _apply_shares(w, spans)
            self._portfolio = _portfolio(leaves, w, self._assets)
        return self._portfolio

    @portfolio.setter
    def portfolio(self, value: Portfolio) -> None:
        """Set the portfolio explicitly."""
        self._portfolio = value

    def _set_allocation(self, share: float, assets: Sequence[str]) -> None:
        """Record this node's left share and drop its portfolio, to be rebuilt on access."""
        self._share = share
        self._assets = assets
        self._portfolio = None

    def _layout(self) -> tuple[list[Cluster], list[Span]]:
        """Lay the subtree out over its leaf order.

        Returns the leaves left to right and, for every non-leaf node in
        post-order (children before parents), the slices of that leaf order its
        two subtrees cover. Iterative and linear in the subtree size; every
        non-leaf node is validated before anything is returned.
        """
        preorder: list[Cluster] = []
        stack: list[Cluster] = [self]
        while stack:
            node = stack.pop()
            preorder.append(node)
            if not node.is_leaf:
                left, right = node._child_clusters()
                stack.append(right)
                stack.append(left)

        # Leaf counts bottom-up, then each node's first leaf position top-down.
        count: dict[int, int] = {}
        for node in reversed(preorder):
            if node.is_leaf:
                count[id(node)] = 1
            else:
                left, right = node._child_clusters()
                count[id(node)] = count[id(left)] + count[id(right)]
        start = {id(self): 0}
        leaves: list[Cluster] = []
        spans: list[Span] = []
        for node in preorder:
            lo = start[id(node)]
            if node.is_leaf:
                leaves.append(node)
            else:
                left, right = node._child_clusters()
                mid = lo + count[id(left)]
                start[id(left)] = lo
                start[id(right)] = mid
                spans.append((node, lo, mid, lo + count[id(node)]))
        spans.reverse()
        return leaves, spans

    # Override narrows the return type to list[Cluster] and validates tree integrity;
    # the traversal order (left to right) matches Node.leaves.
    @property
    def leaves(self) -> list[Cluster]:
        """Get all reachable leaf nodes in left-to-right dendrogram order.

        Returns:
            list[Cluster]: List of all leaf nodes reachable from this cluster
        """
        # Iterative for the same reason as Node.leaves, and still validating each
        # non-leaf node through _child_clusters() so a malformed tree raises here
        # rather than yielding a silently short leaf list.
        result: list[Cluster] = []
        stack: list[Cluster] = [self]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                result.append(node)
                continue
            left, right = node._child_clusters()
            # Right first, so the left subtree is emitted first.
            stack.append(right)
            stack.append(left)
        return result

    def _child_clusters(self) -> tuple[Cluster, Cluster]:
        """Return the validated (left, right) child clusters of a non-leaf node.

        Raises:
            ValueError: If either child is missing on a non-leaf cluster.
            TypeError: If either child is not a Cluster.
        """
        if self.left is None:
            msg = "Expected left child to exist for non-leaf cluster"
            raise ValueError(msg)
        if self.right is None:
            msg = "Expected right child to exist for non-leaf cluster"
            raise ValueError(msg)
        if not isinstance(self.left, Cluster):
            msg = f"Expected left child to be a Cluster for node {self.value}"
            raise TypeError(msg)
        if not isinstance(self.right, Cluster):
            msg = f"Expected right child to be a Cluster for node {self.value}"
            raise TypeError(msg)
        return self.left, self.right


def _apply_shares(w: np.ndarray, spans: list[Span]) -> None:
    """Scale ``w`` bottom-up by the left share of every node, in place.

    ``w`` starts at one per leaf; afterwards it holds the subtree's portfolio in
    leaf order. The products are formed bottom-up, the order in which the
    allocators form them.
    """
    for node, lo, mid, hi in spans:
        w[lo:mid] *= node._share
        w[mid:hi] *= 1.0 - node._share


def _portfolio(leaves: list[Cluster], w: np.ndarray, assets: Sequence[str]) -> Portfolio:
    """The Portfolio holding weight ``w[i]`` for the asset of ``leaves[i]``, in leaf order."""
    portfolio = Portfolio()
    for leaf, weight in zip(leaves, w.tolist(), strict=True):
        portfolio[assets[int(leaf.value)]] = weight
    return portfolio
