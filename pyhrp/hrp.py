import numpy as np
import scipy.cluster.hierarchy as sch

from pyhrp.cluster import Cluster, risk_parity
from pyhrp.linalg import dist


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

#
# def bisection(ids):
#     """
#     Compute the graph underlying the recursive bisection of Marcos Lopez de Prado
#
#     :param ids: A (ranked) set of indixes
#     :return: The root ClusterNode of this tree
#     """
#
#     def split(ids):
#         # split the vector ids in two parts, split in the middle
#         assert len(ids) >= 2
#         n = len(ids)
#         return ids[:n // 2], ids[n // 2:]
#
#     assert len(ids) >= 1
#
#     if len(ids) == 1:
#         return sch.ClusterNode(id=ids[0])
#
#     left, right = split(ids)
#     return sch.ClusterNode(id=nr.randint(low=100000, high=200000), left=bisection(ids=left), right=bisection(ids=right))


# def __hrp(node, cov, weights):
#     if node.is_leaf():
#         # a node is a leaf if has no further relatives downstream. No leaves, no branches...
#         return cov[node.id][node.id], weights
#     else:
#         # compute the variance of the left branch
#         v_left, _ = __hrp(node.left, cov, weights)
#
#         # compute the variance of the right branch
#         v_right, _ = __hrp(node.right, cov, weights)
#
#         # compute the split factors alpha_left and alpha_right
#         # the split is such that v_left * alpha_left == v_right * alpha_right and alpha + beta = 1
#         alpha_left, alpha_right = risk_parity(v_left, v_right)
#
#         # compile a list of reachable leafs from the left node and from the right node
#         # this could be done with an expensive recursive function but scipy's tree provides a powerful pre_order
#         left, right = node.left.pre_order(), node.right.pre_order()
#
#         # update the weights linked to those leafs
#         weights[left], weights[right] = alpha_left * weights[left], alpha_right * weights[right]
#
#         # return the variance for the node and the updated weights
#         return variance(w=weights[left + right], cov=sub(cov, idx=left + right)), weights


def _hrp2(node, cov):
    if node.is_leaf():
        # a node is a leaf if has no further relatives downstream. No leaves, no branches...
        asset = cov.keys().to_list()[node.id]
        return Cluster(assets={asset: 1.0}, variance=cov[asset][asset])
    else:
        cluster_left = _hrp2(node.left, cov)
        cluster_right = _hrp2(node.right, cov)
        return risk_parity(cluster_left, cluster_right, cov=cov)


# def hrp_feed(cov, node=None):
#     """
#     Computes the expected variance and the weights for the hierarchical risk parity portfolio
#     :param cov: This is the covariance matrix that shall be used
#     :param node: Optional. This is the rootnode of the graph describing the dendrogram
#     :return: variance, weights
#     """
#     if node is None:
#         cor = correlation_from_covariance(cov)
#         node = tree(linkage(dist(cor)))
#
#     return __hrp(node, cov, weights=np.ones(cov.shape[1]))


def hrp_feed2(prices, node=None, method="single"):
    """
    Computes the expected variance and the weights for the hierarchical risk parity portfolio
    :param cov: This is the covariance matrix that shall be used
    :param node: Optional. This is the rootnode of the graph describing the dendrogram
    :return: variance, weights
    """
    returns = prices.pct_change().dropna(axis=0, how="all")
    cov, cor = returns.cov(), returns.corr()
    node = node or tree(linkage(dist(cor.values), method=method))

    return _hrp2(node, cov)

