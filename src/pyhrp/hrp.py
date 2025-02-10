"""the hrp algorithm"""

from __future__ import annotations

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd


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
            return ssd.squareform(matrix)

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

                nnn += 1
                # Recursively construct the left and right subtrees
                left_node = _bisection(ids=left)
                nnn += 1
                right_node = _bisection(ids=right)

                nnn += 1
                # Create a new cluster node with the current ID and the left/right subtrees
                return Cluster(id=nnn, left=left_node, right=right_node)

            # Convert the linkage matrix to a tree
            root = sch.to_tree(links, rd=False)

            # Apply the bisection method if requested
            if bisection:
                nnn = len(root.pre_order())
                # Get the leaf IDs in pre-order traversal order
                leaf_ids = root.pre_order()
                # Reconstruct the tree using the bisection method
                root = _bisection(ids=leaf_ids)

            return root
            # return Cluster(id=root.id, left=root.left, right=root.right)

        def _node_to_linkage(n):
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
                linkage_matrix.append([left_id, right_id, float(node.count), node.count])

                # Assign a new ID to the merged cluster
                merged_id = current_id
                current_id += 1
                return merged_id

            # Start the traversal
            _traverse(root)
            M = np.array(linkage_matrix)

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
        links = sch.linkage(distance, method=method)
        root = _tree()

        # convert all nodes into Cluster
        root = convert_to_nodes(root)

        assert isinstance(root, Cluster), f"Root {type(root)}"

        if bisection:
            links = _node_to_linkage(n=cor.shape[0])

        return Dendrogram(root=root, linkage=links, distance=distance, bisection=bisection, method=method)

    def plot(self, ax=None, **kwargs):
        """Plot a dendrogram using matplotlib"""
        if ax is None:
            _, ax = plt.subplots(figsize=(25, 20))
        sch.dendrogram(self.linkage, ax=ax, **kwargs)

        return ax


class Cluster(sch.ClusterNode):
    """
    Clusters are the nodes of the graphs we build.
    Each cluster is aware of the left and the right cluster
    it is connecting to.
    """

    def __init__(self, id, left: Cluster = None, right: Cluster = None, **kwargs):
        super().__init__(id, left, right, **kwargs)

        self.__assets = {}
        self.__variance = None

    # Property for 'variance'
    @property
    def variance(self):
        """Getter for variance."""
        return self.__variance

    # Setter for 'variance'
    @variance.setter
    def variance(self, value):
        """Setter for variance. It allows setting the value."""
        # You can add validation or logic here
        if value < 0:
            raise ValueError("Variance must be non-negative!")
        self.__variance = value

    def __getitem__(self, item):
        return self.__assets[item]

    def __setitem__(self, key, value):
        self.__assets[key] = value

    @property
    def weights(self):
        """weight series"""
        return pd.Series(self.__assets, name="Weights").sort_index()

    def risk_parity(self, cov) -> Cluster:
        """compute a cluster"""
        if self.is_leaf():
            # a node is a leaf if has no further relatives downstream.
            # no leaves, no branches, ...
            asset = cov.keys().to_list()[self.id]
            self[asset] = 1.0
            self.variance = cov[asset][asset]
            return self

        # drill down on the left
        self.left = self.left.risk_parity(cov)
        # drill down on the right
        self.right = self.right.risk_parity(cov)

        # combine left and right into a new cluster
        return self._parity(cov=cov)

    def _parity(self, cov) -> Cluster:
        """
        Given two clusters compute in a bottom-up approach their parent.

        :param cluster: left cluster
        :param cov: covariance matrix. Will pick the correct sub-matrix

        """

        # combine two clusters

        def parity(v_left, v_right):
            """
            Compute the weights for a risk parity portfolio of two assets
            :param v_left: Variance of the "left" portfolio
            :param v_right: Variance of the "right" portfolio
            :return: w, 1-w the weights for the left and the right portfolio.
                     It is w*v_left == (1-w)*v_right hence w = v_right / (v_right + v_left)
            """
            return v_right / (v_left + v_right), v_left / (v_left + v_right)

        # split is s.t. v_left * alpha_left == v_right * alpha_right and alpha + beta = 1
        alpha_left, alpha_right = parity(self.left.variance, self.right.variance)

        # assets in the cluster are the assets of the left and right cluster
        # further downstream
        assets = {
            **(alpha_left * self.left.weights).to_dict(),
            **(alpha_right * self.right.weights).to_dict(),
        }

        weights = np.array(list(assets.values()))

        covariance = cov[assets.keys()].loc[assets.keys()]

        var = np.linalg.multi_dot((weights, covariance, weights))

        self.variance = var
        for asset, weight in assets.items():
            self[asset] = weight

        return self
