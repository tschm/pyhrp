import numpy as np
import numpy.random as nr
from pyhrp.linalg import variance, sub, correlation_from_covariance, dist

import scipy.cluster.hierarchy as sch


def linkage(dist, method="ward", **kwargs):
    """
    Based on distance matrix compute the underlying links
    :param dist: The distance metric based on the correlation matrix
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


def risk_parity(v_left, v_right):
    """
    Compute the weights for a risk parity portfolio of two assets
    :param v_left: Variance of the "left" portfolio
    :param v_right: Variance of the "right" portfolio
    :return: w, 1-w the weights for the left and the right portfolio. It is w*v_left == (1-w)*v_right and hence w = v_right / (v_right + v_left)
    """
    return v_right / (v_left + v_right), 1 - v_right / (v_left + v_right)


def bisection(ids):
    """
    Compute the graph underlying the recursive bisection of Marcos Lopez de Prado

    :param ids: A (ranked) set of indixes
    :return: The root ClusterNode of this tree
    """
    def split(ids):
        # split the vector ids in two parts, split in the middle
        assert len(ids) >= 2
        n = len(ids)
        return ids[:n // 2], ids[n // 2:]

    assert len(ids) >= 1

    if len(ids) == 1:
        return sch.ClusterNode(id=ids[0])

    left, right = split(ids)
    return sch.ClusterNode(id=nr.randint(low=100000, high=200000), left=bisection(ids=left), right=bisection(ids=right))


def __hrp(node, cov, weights):

    if node.is_leaf():
        # a node is a leaf if has no further relatives downstream. No leaves, no branches...
        return cov[node.id][node.id], weights
    else:
        # compute the variance of the left branch
        v_left, _ = __hrp(node.left, cov, weights)

        # compute the variance of the right branch
        v_right, _ = __hrp(node.right, cov, weights)

        # compute the split factors alpha[0] and alpha[1]
        # the split is such that v_left * alpha_left == v_right * alpha_right
        # and alpha + beta = 1
        alpha_left, alpha_right = risk_parity(v_left, v_right)

        # compile a list of reachable leafs from the left node and from the right node
        # this could be done with an expensive recursive function but scipy's tree provides a powerful pre_order
        left, right = node.left.pre_order(), node.right.pre_order()

        # update the weights linked to those leafs
        weights[left], weights[right] = alpha_left * weights[left], alpha_right * weights[right]

        # return the variance for the node and the updated weights
        return variance(w=weights[left + right], cov=sub(cov, idx=left + right)), weights


def hrp_feed(cov, node=None):
    """
    Computes the expected variance and the weights for the hierarchical risk parity portfolio
    :param cov: This is the covariance matrix that shall be used
    :param node: Optional. This is the rootnode of the graph describing the dendrogram
    :return: variance, weights
    """
    if node is None:
        cor = correlation_from_covariance(cov)
        node = tree(linkage(dist(cor)))

    return __hrp(node, cov, weights=np.ones(cov.shape[1]))


def marcos(cov, node=None):
    if node is None:
        cor = correlation_from_covariance(cov)
        node = tree(linkage(dist(cor), method="single"))
        # get the order of the leaves (Marcos calls this quasi-diagonalization)
        ids = node.pre_order()
        # apply bisection, root is now a ClusterNode of the graph
        root = bisection(ids=ids)
        # It's not clear to me why Marcos is going down this route. Rather than sticking with the graph computed above.

    return __hrp(node=root, cov=cov, weights=np.ones(cov.shape[1]))
