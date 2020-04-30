# here we implement the HRP algorithm without the cluster concept. Rather we reach around a weight vector.
# It's not(!) exactly possible to the a smart post-analysis.

import numpy as np
import pandas as pd

from pyhrp.hrp import dist, tree, linkage


def __rp(v_left, v_right):
    """
    Compute the weights for a risk parity portfolio of two assets
    :param v_left: Variance of the "left" portfolio
    :param v_right: Variance of the "right" portfolio
    :return: w, 1-w the weights for the left and the right portfolio. It is w*v_left == (1-w)*v_right and hence w = v_right / (v_right + v_left)
    """
    return v_right / (v_left + v_right), v_left / (v_left + v_right)


def __hrp(node, cov, weights):
    if node.is_leaf():
        # a node is a leaf if has no further relatives downstream. No leaves, no branches...
        return cov[node.id][node.id], weights
    else:
        # compute the variance of the left branch
        v_left, _ = __hrp(node.left, cov, weights)

        # compute the variance of the right branch
        v_right, _ = __hrp(node.right, cov, weights)

        # compute the split factors alpha_left and alpha_right
        # the split is such that v_left * alpha_left == v_right * alpha_right and alpha + beta = 1
        alpha_left, alpha_right = __rp(v_left, v_right)

        # compile a list of reachable leafs from the left node and from the right node
        # this could be done with an expensive recursive function but scipy's tree provides a powerful pre_order
        left, right = node.left.pre_order(), node.right.pre_order()

        # update the weights linked to those leafs
        weights[left], weights[right] = alpha_left * weights[left], alpha_right * weights[right]

        # return the variance for the node and the updated weights
        w = weights[left + right]
        c = cov[left + right, :][:, left + right]

        return np.linalg.multi_dot((w,c,w)), weights


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

    return __hrp(node, cov.values, weights=np.ones(cov.shape[1]))
