import numpy as np


def correlation_from_covariance(covariance):
    # see https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    v = np.sqrt(np.diag(covariance))
    return covariance / np.outer(v, v)


def bilinear(A, x):
    return np.linalg.multi_dot((x, A, x))


def sub(A, idx):
    return A[idx, :][:, idx]


def dist(cov):
    cor = correlation_from_covariance(cov)
    # This is DANGEROUS! 1.0 - 1.0 could easily be < 0
    dist = ((1.0 - cor) / 2.)

    # problem here with 1.0 - corr...
    dist[dist < 0] = 0.0
    return dist ** .5
