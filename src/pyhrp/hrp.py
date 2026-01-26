"""Hierarchical Risk Parity (HRP) algorithm implementation.

This module implements the core HRP algorithm and related functions:
- hrp: Main function to compute HRP portfolio weights
- build_tree: Function to build hierarchical cluster tree from correlation matrix
- Dendrogram: Class to store and visualize hierarchical clustering results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from .algos import risk_parity
from .cluster import Cluster


def hrp(
    prices: pd.DataFrame,
    node: Cluster = None,
    method: Literal["single", "complete", "average", "ward"] = "ward",
    bisection: bool = False,
) -> Cluster:
    """Compute the hierarchical risk parity portfolio weights.

    This is the main entry point for the HRP algorithm. It calculates returns from prices,
    builds a hierarchical clustering tree if not provided, and applies risk parity weights.

    Args:
        prices (pd.DataFrame): Asset price time series
        node (Cluster, optional): Root node of the hierarchical clustering tree.
            If None, a tree will be built from the correlation matrix.
        method (Literal["single", "complete", "average", "ward"]): Linkage method to use for distance calculation
            - "single": minimum distance between points (nearest neighbor)
            - "complete": maximum distance between points (furthest neighbor)
            - "average": average distance between all points
            - "ward": Ward variance minimization
        bisection (bool): Whether to use bisection method for tree construction

    Returns:
        Cluster: The root cluster with portfolio weights assigned according to HRP
    """
    returns = prices.pct_change().dropna(axis=0, how="all")
    cov, cor = returns.cov(), returns.corr()
    node = node or build_tree(cor, method=method, bisection=bisection).root

    return risk_parity(root=node, cov=cov)


@dataclass(frozen=True)
class Dendrogram:
    """Container for hierarchical clustering dendrogram data and visualization.

    This class stores the results of hierarchical clustering and provides methods
    for accessing and visualizing the dendrogram structure.

    Attributes:
        root (Cluster): The root node of the hierarchical clustering tree
        assets (list[Asset]): List of assets included in the clustering
        linkage (np.ndarray | None): Linkage matrix in scipy format for plotting
        distance (np.ndarray | None): Distance matrix used for clustering
        method (str | None): Linkage method used for clustering
    """

    root: Cluster
    assets: list[str]
    distance: pd.DataFrame | None = None
    linkage: np.ndarray | None = None
    method: str | None = None

    def __post_init__(self):
        """Validate dataclass fields after initialization.

        Ensures that the optional distance matrix, when provided, is a pandas
        DataFrame aligned with the asset order, and verifies that the number of
        leaves in the cluster tree matches the number of assets.
        """
        # ---- Optional: validate distance index/columns ----
        if self.distance is not None:
            if not isinstance(self.distance, pd.DataFrame):
                raise TypeError("distance must be a pandas DataFrame.")  # noqa: TRY003

            # Optionally check if distance matches assets
            if not self.distance.index.equals(pd.Index(self.assets)) or not self.distance.columns.equals(
                pd.Index(self.assets)
            ):
                raise ValueError("Distance matrix index/columns must align with assets.")  # noqa: TRY003

        # Check the number of leaves and assets
        if len(self.root.leaves) != len(self.assets):
            raise ValueError("Number of leaves does not match number of assets.")  # noqa: TRY003

    def plot(self, **kwargs):
        """Plot the dendrogram."""
        sch.dendrogram(self.linkage, leaf_rotation=90, leaf_font_size=8, labels=self.assets, **kwargs)

    @property
    def ids(self):
        """Node values in the order left -> right as they appear in the dendrogram."""
        return [node.value for node in self.root.leaves]

    @property
    def names(self):
        """The asset names as induced by the order of ids."""
        return [self.assets[i] for i in self.ids]


def _compute_distance_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert correlation matrix to distance matrix."""
    c = corr.values
    dist = np.sqrt(np.clip((1.0 - c) / 2.0, a_min=0.0, a_max=1.0))
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(data=dist, index=corr.index, columns=corr.index)


def build_tree(
    cor: pd.DataFrame, method: Literal["single", "complete", "average", "ward"] = "ward", bisection: bool = False
) -> Dendrogram:
    """Build hierarchical cluster tree from correlation matrix.

    This function converts a correlation matrix to a distance matrix, performs
    hierarchical clustering, and returns a Dendrogram object containing the
    resulting tree structure.

    Args:
        cor (pd.DataFrame): Correlation matrix of asset returns
        method (Literal["single", "complete", "average", "ward"]): Linkage method for hierarchical clustering
            - "single": minimum distance between points (nearest neighbor)
            - "complete": maximum distance between points (furthest neighbor)
            - "average": average distance between all points
            - "ward": Ward variance minimization
        bisection (bool): Whether to use bisection method for tree construction

    Returns:
        Dendrogram: Object containing the hierarchical clustering tree, with:
            - root: Root cluster node
            - linkage: Linkage matrix for plotting
            - assets: List of assets
            - method: Clustering method used
            - distance: Distance matrix
    """
    # Create distance matrix and linkage
    assert isinstance(cor, pd.DataFrame), "Correlation matrix must be a pandas DataFrame."
    dist = _compute_distance_matrix(cor)
    links = sch.linkage(ssd.squareform(dist), method=method)

    # Convert scipy tree to our Cluster format
    def to_cluster(node: sch.ClusterNode) -> Cluster:
        """Convert a scipy ClusterNode to our Cluster format.

        Args:
            node (sch.ClusterNode): A node from scipy's hierarchical clustering

        Returns:
            Cluster: Equivalent node in our Cluster format
        """
        if node.left is not None:
            left = to_cluster(node.left)
            right = to_cluster(node.right)
            return Cluster(value=node.id, left=left, right=right)
        return Cluster(value=node.id)

    root = to_cluster(sch.to_tree(links, rd=False))

    # Apply bisection if requested
    if bisection:

        def bisect_tree(ids: list[int]) -> Cluster:
            """Build tree by recursive bisection.

            This function recursively splits the list of IDs in half and creates
            a binary tree where each node represents a split.

            Args:
                ids (list[int]): List of leaf node IDs to organize into a tree

            Returns:
                Cluster: Root node of the constructed tree
            """
            nonlocal nnn

            if len(ids) == 1:
                return Cluster(value=ids[0])

            mid = len(ids) // 2
            left_ids, right_ids = ids[:mid], ids[mid:]

            left = bisect_tree(left_ids)
            right = bisect_tree(right_ids)

            nnn += 1
            return Cluster(value=nnn, left=left, right=right)

        # Rebuild tree using bisection
        leaf_ids = [node.value for node in root.leaves]
        nnn = max(leaf_ids)
        root = bisect_tree(leaf_ids)

        # Convert back to linkage format for plotting
        links = []

        def get_linkage(node: Cluster) -> None:
            """Convert tree structure back to linkage matrix format.

            This function traverses the tree and builds a linkage matrix compatible
            with scipy's hierarchical clustering format for visualization.

            Args:
                node (Cluster): Current node being processed
            """
            if node.left is not None:
                get_linkage(node.left)
                get_linkage(node.right)
                links.append(
                    [
                        node.left.value,
                        node.right.value,
                        float(node.size),
                        len(node.left.leaves) + len(node.right.leaves),
                    ]
                )

        get_linkage(root)
        links = np.array(links)

    return Dendrogram(root=root, linkage=links, method=method, distance=dist, assets=cor.columns)
