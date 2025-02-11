"""the hrp algorithm"""

from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from .cluster import Cluster


def hrp(prices, node=None, method="ward", bisection=False) -> Cluster:
    """
    Computes the root node for the hierarchical risk parity portfolio
    :param node: Optional. This is the rootnode of the graph describing the dendrogram
    :param method: Optional. Which method to use for the dendrogram
    :param bisection: Optional. Whether to use bisection method
    :return: the root cluster of the risk parity portfolio
    """
    returns = prices.pct_change().dropna(axis=0, how="all")
    cov, cor = returns.cov(), returns.corr()
    node = node or Dendrogram.build(cor.values, method=method, bisection=bisection).root

    return node.risk_parity(cov)


class Dendrogram(NamedTuple):
    root: Cluster
    linkage: np.ndarray
    distance: np.ndarray
    bisection: bool
    method: str

    @staticmethod
    def build(cor, method="ward", bisection=False) -> Dendrogram:
        """
        Build a dendrogram from a correlation matrix
        """

        def _dist():
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
            return matrix
            # return ssd.squareform(matrix)

        def _tree() -> Cluster:
            """
            Compute the root ClusterNode.

            :param links: The linkage matrix compiled by the linkage function.
            :param bisection: If True, apply the bisection method to sort the leaves.
            :return: The root node. From there, it's possible to reach the entire graph.
            """

            def _bisection(ids) -> Cluster:
                """
                Compute the graph underlying the recursive bisection of Marcos Lopez de Prado.
                Ensures that the pre-order traversal of the tree remains unchanged.

                :param ids: A (ranked) set of indices (leaf nodes).
                :return: The root ClusterNode of this tree.
                """
                nonlocal nnn

                def split(ids):
                    """Split the vector ids into two parts, split in the middle."""
                    num = len(ids)
                    return ids[: num // 2], ids[num // 2 :]

                # Base case: if there's only one ID, return a leaf node
                if len(ids) == 1:
                    return Cluster(id=ids[0])

                # Split the IDs into left and right halves
                left, right = split(ids)

                # nnn += 1
                # Recursively construct the left and right subtrees
                left_node = _bisection(ids=left)
                # nnn += 1
                right_node = _bisection(ids=right)

                nnn += 1
                # Create a new cluster node with the current ID and the left/right subtrees
                return Cluster(id=nnn, left=left_node, right=right_node)

            # Convert the linkage matrix to a tree
            root = sch.to_tree(links, rd=False)

            # Apply the bisection method if requested
            if bisection:
                nnn = len(root.pre_order()) - 1
                # Get the leaf IDs in pre-order traversal order
                leaf_ids = root.pre_order()
                # Reconstruct the tree using the bisection method
                root = _bisection(ids=leaf_ids)

            return root
            # return Cluster(id=root.id, left=root.left, right=root.right)

        def _node_to_linkage(root: Cluster) -> np.ndarray:
            """
            Convert a hierarchical clustering tree (root node) back into a linkage matrix.
            Needed to plot the dendrogram

            Parameters:
            root: The root node of the hierarchical clustering tree.

            Returns:
            linkage_matrix: A n x 4 numpy array representing the linkage matrix.
            """
            linkage_matrix = []

            def _traverse(node):
                if node.left is not None:
                    # Recursively traverse the left and right children
                    _traverse(node.left)
                    _traverse(node.right)

                    # dist = node.left.distance(other=node.right, distance_matrix=distance_matrix)
                    dist = float(node.count)

                    # Record the merge step
                    linkage_matrix.append([node.left.id, node.right.id, dist, node.count])

            # Start the traversal
            _traverse(root)
            M = np.array(linkage_matrix)
            # print(M)
            return M

        def convert_to_nodes(cluster_node):
            """
            Recursively converts a ClusterNode tree to a tree of Nodes.

            Args:
            - cluster_node: The root of the ClusterNode tree.

            Returns:
            - Node: A tree of Node instances with the same structure as the input ClusterNode tree.
            """
            # Base case: if the cluster_node is a leaf node (id is not None)
            # if cluster_node is not None:
            #    return Cluster(id=cluster_node.id, distance=cluster_node.distance, count=cluster_node.count)

            # Recursive case: convert the left and right branches (subtrees)
            if cluster_node.left is not None:
                left_node = convert_to_nodes(cluster_node.left)

                # if cluster_node.right is not None:
                right_node = convert_to_nodes(cluster_node.right)

                # Create a new Node instance for the current cluster_node and assign the converted left and right nodes
                return Cluster(left=left_node, right=right_node, id=cluster_node.id, count=cluster_node.count)

            return Cluster(id=cluster_node.id, count=cluster_node.count)

        distance = _dist()
        links = sch.linkage(ssd.squareform(distance), method=method)
        root = _tree()

        # convert all nodes into Cluster
        root = convert_to_nodes(root)

        assert isinstance(root, Cluster), f"Root {type(root)}"

        if bisection:
            links = _node_to_linkage(root=root)

        return Dendrogram(root=root, linkage=links, distance=distance, bisection=bisection, method=method)

    def plot(self, ax=None, **kwargs):
        """Plot a dendrogram using matplotlib"""
        if ax is None:
            _, ax = plt.subplots(figsize=(25, 20))
        sch.dendrogram(self.linkage, ax=ax, **kwargs)

        return ax
