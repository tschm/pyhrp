"""the hrp algorithm"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from .algos import risk_parity
from .cluster import Asset, Cluster


def hrp(
    prices, node=None, method: Literal["single", "complete", "average", "ward"] = "ward", bisection=False
) -> Cluster:
    """
    Computes the root node for the hierarchical risk parity portfolio
    :param node: Optional. This is the rootnode of the graph describing the dendrogram
    :param method: Linkage method to use for distance calculation
           - "single": minimum distance between points (nearest neighbor)
           - "complete": maximum distance between points (furthest neighbor)
           - "average": average distance between all points
           - "ward": Ward variance minimization
    :param bisection: Optional. Whether to use bisection method
    :return: the root cluster of the risk parity portfolio
    """
    returns = prices.pct_change().dropna(axis=0, how="all")

    cov, cor = returns.cov(), returns.corr()
    node = node or build_tree(cor, method=method, bisection=bisection).root

    return risk_parity(root=node, cov=cov)


@dataclass(frozen=True)
class Dendrogram:
    """Simple container for dendrogram data and plotting"""

    root: Cluster
    assets: list[Asset]
    linkage: np.ndarray | None = None
    distance: np.ndarray | None = None
    method: str | None = None

    def __post_init__(self):
        if not len(self.root.leaves) == len(self.assets):
            raise ValueError("Inconsistent number of assets and leaves")

    def plot(self, **kwargs):
        """Plot the dendrogram"""
        labels = [asset.name for asset in self.assets]
        sch.dendrogram(self.linkage, leaf_rotation=90, leaf_font_size=8, labels=labels, **kwargs)

    @property
    def ids(self):
        """node values in the order left -> right as they appear in the dendrogram"""
        return [node.value for node in self.root.leaves]

    @property
    def names(self):
        """The asset names as induced by the order of ids"""
        return [self.assets[i].name for i in self.ids]


def _compute_distance_matrix(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix"""
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, a_min=0.0, a_max=1.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def build_tree(
    cor: pd.DataFrame, method: Literal["single", "complete", "average", "ward"] = "ward", bisection: bool = False
) -> Dendrogram:
    """
    Build hierarchical cluster tree from correlation matrix

    Args:
        cor: Correlation matrix
        method: Clustering method
        bisection: Whether to use bisection method

    Returns:
        root: Root cluster node
        linkage: Linkage matrix for plotting
    """
    # Create distance matrix and linkage
    dist = _compute_distance_matrix(cor.values)
    links = sch.linkage(ssd.squareform(dist), method=method)

    # Convert scipy tree to our Cluster format
    def to_cluster(node: sch.ClusterNode) -> Cluster:
        if node.left is not None:
            left = to_cluster(node.left)
            right = to_cluster(node.right)
            return Cluster(value=node.id, left=left, right=right)
        return Cluster(value=node.id)

    root = to_cluster(sch.to_tree(links, rd=False))

    # Apply bisection if requested
    if bisection:

        def bisect_tree(ids):
            """Build tree by recursive bisection"""
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

        def get_linkage(node):
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

    return Dendrogram(root=root, linkage=links, method=method, distance=dist, assets=list(cor.columns))
