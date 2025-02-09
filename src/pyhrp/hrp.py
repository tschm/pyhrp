"""the hrp algorithm"""

from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd


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

    @staticmethod
    def build(cor, method="ward", bisection=False):
        distance = _dist(cor)
        links = sch.linkage(distance, method=method)
        root = _tree(links, bisection=bisection)

        if bisection:
            links = _node_to_linkage(root, n=cor.shape[0])

        return Dendrogram(root=root, linkage=links, distance=distance, bisection=bisection, method=method)


class Node:
    def __init__(self, id=None, left=None, right=None):
        self.id = id
        self.left = left
        self.right = right

    def pre_order(self):
        if self.left:
            return self.left.pre_order() + self.right.pre_order()
        else:
            return [self.id]

    def __repr__(self):
        return f"Node(id={self.id}, left={self.left}, right={self.right})"

    def is_leaf(self):
        return self.left is None and self.right is None

    def __len__(self) -> int:
        if self.left:
            return len(self.left) + len(self.right)
        else:
            return 1


def _bisection(ids, n: int) -> Node:
    """
    Compute the graph underlying the recursive bisection of Marcos Lopez de Prado.
    Ensures that the pre-order traversal of the tree remains unchanged.

    :param ids: A (ranked) set of indices (leaf nodes).
    :param n: The current ID to assign to the newly created cluster node.
    :return: The root ClusterNode of this tree.
    """

    def split(ids):
        """Split the vector ids into two parts, split in the middle."""
        num = len(ids)
        return ids[: num // 2], ids[num // 2 :]

    # Base case: if there's only one ID, return a leaf node
    if len(ids) == 1:
        return Node(id=ids[0])

    # Split the IDs into left and right halves
    left, right = split(ids)

    # Recursively construct the left and right subtrees
    left_node = _bisection(ids=left, n=n + 1)
    right_node = _bisection(ids=right, n=n + 1 + len(left))

    # Create a new cluster node with the current ID and the left/right subtrees
    return Node(id=n, left=left_node, right=right_node)


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
        linkage_matrix.append([left_id, right_id, float(len(node)), len(node)])

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


def _tree(links, bisection: bool = False) -> Node:
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
        root = _bisection(ids=leaf_ids, n=len(leaf_ids))

    return Node(id=root.id, left=root.left, right=root.right)
