from itertools import chain
import numpy as np

from pyhrp.linalg import bilinear, sub


def split(v_left, v_right):
    alpha = 1 - v_left / (v_left + v_right)
    return np.array([alpha, 1 - alpha])


def __hrp(node, cov, weights):
    def successors(node):
        # get the list of ids following a node downstream
        if node.is_leaf():
            return [node.id]

        return list(chain.from_iterable([[node.id], successors(node.left), successors(node.right)]))

    if node.is_leaf():
        # weights[node.id] = 1.0
        return cov[node.id][node.id], weights
    else:
        v_left, _ = __hrp(node.left, cov, weights)

        # compute the variance of the right branch
        v_right, _ = __hrp(node.right, cov, weights)

        # compute the split factor
        alpha = split(v_left, v_right)

        # update the weights on the left and the ones on the right
        left = successors(node=node.left)
        right = successors(node=node.right)

        weights[left] = alpha[0] * weights[left]
        weights[right] = alpha[1] * weights[right]

        idx = np.array(left + right)

        # look only at all the leafs
        idx = idx[idx < cov.shape[0]]

        v = bilinear(x=weights[idx], A=sub(cov, idx=idx))

        # return the variance for the node and the updated weights
        return v, weights


def hrp_feed(node, cov):
    weights = np.ones(2 * cov.shape[1] - 1)
    v, weights = __hrp(node, cov, weights=weights)
    return v, weights[0:cov.shape[1]]
