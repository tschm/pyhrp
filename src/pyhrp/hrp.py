"""the hrp algorithm"""

from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

import pyhrp.node as node


# Define a NamedTuple
class Dendrogram(NamedTuple):
    root: sch.ClusterNode
    linkage: np.ndarray
    distance: np.ndarray
    bisection: bool
    method: str

    def plot(self, ax=None, **kwargs):
        """Plot a dendrogram using matplotlib"""
        if ax is None:
            _, ax = plt.subplots(figsize=(25, 20))
        sch.dendrogram(self.linkage, ax=ax, **kwargs)

        return ax


def _node_to_linkage(root, n):
    """
    Convert a hierarchical clustering tree (root node) back into a linkage matrix.

    Parameters:
    root: The root node of the hierarchical clustering tree.
    n: The number of original data points.

    Returns:
    linkage_matrix: A (n-1) x 4 numpy array representing the linkage matrix.
    """
    linkage_matrix = []
    current_id = n  # Start assigning IDs for merged clusters from n

    def _traverse(node):
        nonlocal current_id
        if node.is_leaf():
            return node.id  # Return the leaf node's ID

        # Recursively traverse the left and right children
        left_id = _traverse(node.left)
        right_id = _traverse(node.right)

        # Record the merge step
        linkage_matrix.append([left_id, right_id, node.count, node.count])

        # Assign a new ID to the merged cluster
        merged_id = current_id
        current_id += 1
        return merged_id

    # Start the traversal
    _traverse(root)
    M = np.array(linkage_matrix)

    return M


def _dist(cor):
    """
    Compute the correlation based distance matrix d,
    compare with page 239 of the first book by Marcos
    :param cor: the n x n correlation matrix
    :return: The matrix d indicating the distance between column i and i.
             Note that all the diagonal entries are zero.

    """
    # https://stackoverflow.com/questions/18952587/
    matrix = np.sqrt(np.clip((1.0 - cor) / 2.0, a_min=0.0, a_max=1.0))
    np.fill_diagonal(matrix, val=0.0)
    return ssd.squareform(matrix)


def _linkage(dist_vec, method="ward", **kwargs) -> np.ndarray:
    """
    Based on distance matrix compute the underlying links
    :param dist_vec: The distance vector based on the correlation matrix
    :param method: "single", "ward", etc.
    :return: links  The links describing the graph (useful to draw the dendrogram)
                    and basis for constructing the tree object
    """
    # compute the root node of the dendrogram
    return sch.linkage(dist_vec, method=method, **kwargs)


def _tree(links, bisection: bool = False) -> node.Node:
    """
    Compute the root ClusterNode.

    :param links: The linkage matrix compiled by the linkage function.
    :param bisection: If True, apply the bisection method to sort the leaves.
    :return: The root node. From there, it's possible to reach the entire graph.
    """
    # Convert the linkage matrix to a tree
    root = sch.to_tree(links, rd=False)

    # Apply the bisection method if requested
    if bisection:
        # Get the leaf IDs in pre-order traversal order
        leaf_ids = root.pre_order()
        # Reconstruct the tree using the bisection method
        root = node.bisection(ids=leaf_ids, n=len(leaf_ids))

    return node.Node(id=root.id, left=root.left, right=root.right)


def root(cor, method="ward", bisection=False) -> Dendrogram:
    distance = _dist(cor)
    links = _linkage(distance, method=method)
    root = _tree(links, bisection=bisection)

    if bisection:
        links = _node_to_linkage(root, n=cor.shape[0])

    return Dendrogram(root=root, linkage=links, distance=distance, bisection=bisection, method=method)
