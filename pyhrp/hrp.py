import numpy as np
from pyhrp.linalg import variance, sub

import scipy.cluster.hierarchy as sch


def root(dist, method="ward"):
    """
    Based on distance matrix compute the underlying graph (dendrogram)
    :param dist: The distance metric based on the correlation matrix
    :param method: "single", "ward", etc.
    :return: rootnode, links  The rootnode of the graph and the links describing the graph (useful to draw the dendrogram)
    """
    # compute the root node of the dendrogram
    links = sch.linkage(dist, method=method, optimal_ordering=True)
    return sch.to_tree(links, rd=False), links


def risk_parity(v_left, v_right):
    """
    Compute the weights for a risk parity portfolio of two assets
    :param v_left: Variance of the "left" portfolio
    :param v_right: Variance of the "right" portfolio
    :return: w, 1-w the weights for the left and the right portfolio. It is w*v_left == (1-w)*v_right and hence w = v_right / (v_right + v_left)
    """
    return v_right / (v_left + v_right), 1 - v_right / (v_left + v_right)


def __hrp(node, cov, weights):

    if node.is_leaf():
        return cov[node.id][node.id], weights
    else:
        # compute the variance of the left branch
        v_left, weights = __hrp(node.left, cov, weights)

        # compute the variance of the right branch
        v_right, weights = __hrp(node.right, cov, weights)

        # compute the split factors alpha[0] and alpha[1]
        # the split is such that v_left * alpha == v_right * beta
        # shouldn't the split be v_left * alpha^2 == v_right * beta^2
        # and alpha + beta = 1
        alpha, beta = risk_parity(v_left, v_right)

        # compile a list of reachable leafs from the left node and from the right node
        # this could be done with an expensive recursive function but scipy's tree provides a powerful pre_order
        left, right = node.left.pre_order(lambda x: x.id), node.right.pre_order(lambda x: x.id)

        # update the weights linked to those leafs
        weights[left], weights[right] = alpha * weights[left], beta * weights[right]

        # return the variance for the node and the updated weights
        return variance(w=weights[left + right], cov=sub(cov, idx=left + right)), weights


def hrp_feed(node, cov):
    """
    Computes the expected variance and the weights for the hierarchical risk parity portfolio
    :param node: This is the rootnode of the graph describing the dendrogram
    :param cov: This is the covariance matrix that shall be used
    :return: variance, weights
    """
    return __hrp(node, cov, weights=np.ones(cov.shape[1]))

