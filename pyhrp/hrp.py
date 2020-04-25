import numpy as np
from pyhrp.linalg import bilinear, sub


def split(v_left, v_right):
    alpha = 1 - v_left / (v_left + v_right)
    return alpha, 1 - alpha


def __hrp(node, cov, weights):

    if node.is_leaf():
        return cov[node.id][node.id], weights
    else:
        # compute the variance of the left branch
        v_left, _ = __hrp(node.left, cov, weights)

        # compute the variance of the right branch
        v_right, _ = __hrp(node.right, cov, weights)

        # compute the split factors alpha[0] and alpha[1]
        # the split is such that v_left * alpha == v_right * beta
        # shouldn't the split be v_left * alpha^2 == v_right * beta^2
        # and alpha + beta = 1
        alpha, beta = split(v_left, v_right)

        # compile a list of reachable leafs from the left node and from the right node
        # this could be done with an expensive recursive function but scipy's tree provides a powerful pre_order
        left = node.left.pre_order(lambda x: x.id)
        right = node.right.pre_order(lambda x: x.id)

        # update the weights linked to those leafs
        weights[left] = alpha * weights[left]
        weights[right] = beta * weights[right]

        # this is the list of all reachable leafs from both nodes
        idx = left + right

        # return the variance for the node and the updated weights
        return bilinear(x=weights[idx], A=sub(cov, idx=idx)), weights


def hrp_feed(node, cov):
    return __hrp(node, cov, weights=np.ones(cov.shape[1]))

