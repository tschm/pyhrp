import numpy as np
import scipy.cluster.hierarchy as sch

from pyhrp.cluster import Cluster, risk_parity

import scipy.spatial.distance as ssd


def dist(cor):
    """
    Compute the correlation based distance matrix d, compare with page 239 of the first book by Marcos
    :param cor: the n x n correlation matrix
    :return: The matrix d indicating the distance between column i and i. Note that all the diagonal entries are zero.

    """
    # https://stackoverflow.com/questions/18952587/
    matrix = np.sqrt(np.clip((1.0 - cor) / 2., a_min=0.0, a_max=1.0))
    np.fill_diagonal(matrix, val=0.0)
    return ssd.squareform(matrix)


def linkage(dist, method="ward", **kwargs):
    """
    Based on distance matrix compute the underlying links
    :param dist: The distance vector based on the correlation matrix
    :param method: "single", "ward", etc.
    :return: links  The links describing the graph (useful to draw the dendrogram) and basis for constructing the tree object
    """
    # compute the root node of the dendrogram
    return sch.linkage(dist, method=method, **kwargs)


def tree(linkage):
    """
    Compute the root ClusterNode.
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ClusterNode.html
    :param links: The Linkage matrix compiled by the linkage function above
    :return: The root node. From there it's possible to reach the entire graph
    """
    return sch.to_tree(linkage, rd=False)


def _hrp(node, cov):
    if node.is_leaf():
        # a node is a leaf if has no further relatives downstream. No leaves, no branches...
        asset = cov.keys().to_list()[node.id]
        return Cluster(assets={asset: 1.0}, variance=cov[asset][asset])
    else:
        cluster_left = _hrp(node.left, cov)
        cluster_right = _hrp(node.right, cov)
        return risk_parity(cluster_left, cluster_right, cov=cov)


def hrp(prices, node=None, method="single"):
    """
    Computes the expected variance and the weights for the hierarchical risk parity portfolio
    :param cov: This is the covariance matrix that shall be used
    :param node: Optional. This is the rootnode of the graph describing the dendrogram
    :return: variance, weights
    """
    returns = prices.pct_change().dropna(axis=0, how="all")
    cov, cor = returns.cov(), returns.corr()
    links = linkage(dist(cor.values), method=method)
    node = node or tree(links)

    return _hrp(node, cov)

