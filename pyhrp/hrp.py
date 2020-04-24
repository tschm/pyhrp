import numpy as np
from itertools import chain


def dist(cov):
    # see https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    def correlation_from_covariance(covariance):
        v = np.sqrt(np.diag(covariance))
        return covariance / np.outer(v, v)

    cor = correlation_from_covariance(cov)
    # This is DANGEROUS! 1.0 - 1.0 could easily be < 0
    dist = ((1.0 - cor) / 2.)
    # problem here with 1.0 - corr...
    dist[dist < 0] = 0.0
    return dist ** .5


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
        # have we reached a node ending in two leafs?
        if node.left.is_leaf() and node.right.is_leaf():
            ids = [node.left.id, node.right.id]
            c = cov[ids, :][:, ids]
            # risk-parity
            d = 1 / np.diag(c)
            weights[ids] = d / d.sum()
            return np.linalg.multi_dot([weights[ids], c, weights[ids]]), weights

        else:

            # compute the variance of the left branch
            v1, _ = __hrp(node.left, cov, weights)
            # compute the variance of the right branch
            v2, _ = __hrp(node.right, cov, weights)

            # compute the split factor
            alpha = 1 - v1 / (v1 + v2)

            # update the weights on the left and the ones on the right
            left = successors(node=node.left)
            right = successors(node=node.right)

            weights[left] = alpha * weights[left]
            weights[right] = (1 - alpha) * weights[right]

            # return the variance for the node and the updated weights
            return alpha ** 2 * v1 + ((1 - alpha) ** 2) * v2, weights


def hrp_feed(node, cov):
    weights = np.ones(2 * cov.shape[1] - 1)
    v, weights = __hrp(node, cov, weights=weights)
    return v, weights[0:cov.shape[1]]
