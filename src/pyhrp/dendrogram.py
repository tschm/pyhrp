"""Hierarchical clustering tree construction and the Dendrogram container.

This module builds the hierarchical clustering tree consumed by the HRP
allocation entry points and stores it in a :class:`Dendrogram`:
- build_tree: Build a hierarchical cluster tree from a correlation matrix
- Dendrogram: Container for the clustering result and its visualization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from .cluster import Cluster

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["Dendrogram", "build_tree"]


@dataclass(frozen=True)
class Dendrogram:
    """Container for hierarchical clustering dendrogram data and visualization.

    This class stores the results of hierarchical clustering and provides methods
    for accessing and visualizing the dendrogram structure.

    Attributes:
        root (Cluster): The root node of the hierarchical clustering tree
        assets (list[str]): Names of assets included in the clustering
        linkage (np.ndarray | None): Linkage matrix in scipy format for plotting
        distance (pl.DataFrame | None): Distance matrix used for clustering
        method (str | None): Linkage method used for clustering
    """

    root: Cluster
    assets: list[str]
    distance: pl.DataFrame | None = None
    linkage: np.ndarray | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        """Validate dataclass fields after initialization.

        Ensures that the optional distance matrix, when provided, is a polars
        DataFrame with columns aligned to the asset list, and verifies that the
        number of leaves in the cluster tree matches the number of assets.
        """
        if self.distance is not None:
            if not isinstance(self.distance, pl.DataFrame):
                raise TypeError("distance must be a polars DataFrame.")

            if self.distance.columns != list(self.assets):
                raise ValueError("Distance matrix index/columns must align with assets.")

        if len(self.root.leaves) != len(self.assets):
            raise ValueError("Number of leaves does not match number of assets.")

    def plot(self, **kwargs: object) -> go.Figure:
        """Build and return a plotly dendrogram figure.

        Delegates to :func:`pyhrp.plot.plot_dendrogram`; the plotly dependency
        is imported lazily so importing the allocation core stays plotly-free.
        """
        from .plot import plot_dendrogram

        return plot_dendrogram(self, **kwargs)

    @property
    def ids(self) -> list[int]:
        """Node values in the order left -> right as they appear in the dendrogram."""
        return [node.value for node in self.root.leaves]

    @property
    def names(self) -> list[str]:
        """The asset names as induced by the order of ids."""
        return [self.assets[i] for i in self.ids]


def _compute_distance_matrix(corr: pl.DataFrame) -> pl.DataFrame:
    """Convert correlation matrix to distance matrix."""
    c = corr.to_numpy()
    dist = np.sqrt(np.clip((1.0 - c) / 2.0, a_min=0.0, a_max=1.0))
    np.fill_diagonal(dist, 0.0)
    cols = corr.columns
    return pl.DataFrame(dict(zip(cols, dist, strict=True)))


def _bisect_tree(ids: list[int], next_id: int) -> tuple[Cluster, int]:
    """Build tree by recursive bisection."""
    if not ids:
        raise ValueError("ids must contain at least one node id.")
    if len(ids) == 1:
        return Cluster(value=ids[0]), next_id

    mid = len(ids) // 2
    left_ids, right_ids = ids[:mid], ids[mid:]
    left, next_id = _bisect_tree(left_ids, next_id)
    right, next_id = _bisect_tree(right_ids, next_id)
    next_id += 1
    return Cluster(value=next_id, left=left, right=right), next_id


def _get_linkage(node: Cluster) -> list[list[float]]:
    """Convert tree structure back to linkage matrix format."""
    links_list: list[list[float]] = []
    if node.left is not None and node.right is not None:
        if not isinstance(node.left, Cluster):
            raise TypeError("Expected left child to be a Cluster")  # pragma: no cover
        if not isinstance(node.right, Cluster):
            raise TypeError("Expected right child to be a Cluster")  # pragma: no cover
        links_list.extend(_get_linkage(node.left))
        links_list.extend(_get_linkage(node.right))
        links_list.append(
            [
                float(node.left.value),
                float(node.right.value),
                float(node.size),
                float(len(node.left.leaves) + len(node.right.leaves)),
            ]
        )
    return links_list


def _check_finite_correlations(cor: pl.DataFrame, c: np.ndarray) -> None:
    """Raise if the correlation matrix contains non-finite values.

    Names the offending assets when the non-finite values sit on the diagonal,
    since a constant (zero-variance) price series is the usual cause.
    """
    bad = [col for col, diag in zip(cor.columns, np.diagonal(c), strict=True) if not np.isfinite(diag)]
    if bad:
        msg = (
            f"Correlation matrix contains non-finite values for assets {bad}; "
            "constant (zero-variance) price series produce NaN correlations."
        )
        raise ValueError(msg)
    if not np.isfinite(c).all():
        msg = "Correlation matrix contains non-finite values."
        raise ValueError(msg)


def _validate_correlation_matrix(cor: pl.DataFrame) -> None:
    """Validate the correlation matrix accepted by :func:`build_tree`.

    Raises:
        TypeError: If ``cor`` is not a polars DataFrame.
        ValueError: If it has fewer than two assets or contains non-finite values.
    """
    if not isinstance(cor, pl.DataFrame):
        raise TypeError("Correlation matrix must be a polars DataFrame.")
    if len(cor.columns) < 2:
        msg = "Correlation matrix must contain at least two assets."
        raise ValueError(msg)
    _check_finite_correlations(cor, cor.to_numpy())


def _to_cluster(node: sch.ClusterNode) -> Cluster:
    """Convert a scipy ClusterNode tree into our Cluster format.

    Args:
        node (sch.ClusterNode): A node from scipy's hierarchical clustering.

    Returns:
        Cluster: Equivalent node in our Cluster format.
    """
    if node.left is not None and node.right is not None:
        return Cluster(value=node.id, left=_to_cluster(node.left), right=_to_cluster(node.right))
    return Cluster(value=node.id)


def build_tree(
    cor: pl.DataFrame, method: Literal["single", "complete", "average", "ward"] = "ward", bisection: bool = False
) -> Dendrogram:
    """Build hierarchical cluster tree from correlation matrix.

    This function converts a correlation matrix to a distance matrix, performs
    hierarchical clustering, and returns a Dendrogram object containing the
    resulting tree structure.

    Args:
        cor (pl.DataFrame): Correlation matrix of asset returns (columns are assets)
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

    Examples:
        >>> import polars as pl
        >>> from pyhrp.dendrogram import build_tree
        >>> cor = pl.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]})
        >>> dg = build_tree(cor, method="ward")
        >>> dg.root.leaf_count
        2
    """
    _validate_correlation_matrix(cor)
    dist = _compute_distance_matrix(cor)
    links = sch.linkage(ssd.squareform(dist.to_numpy(), checks=False), method=method)

    root = _to_cluster(sch.to_tree(links, rd=False))

    # Apply bisection if requested
    if bisection:
        # Rebuild tree using bisection
        leaf_ids: list[int] = [int(node.value) for node in root.leaves]
        root, _ = _bisect_tree(ids=leaf_ids, next_id=max(leaf_ids))
        links = np.array(_get_linkage(root))

    return Dendrogram(root=root, linkage=links, method=method, distance=dist, assets=cor.columns)
