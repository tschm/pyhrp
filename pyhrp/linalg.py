import numpy as np
import scipy.spatial.distance as ssd


def correlation_from_covariance(cov):
    """
    Compute a correlation from a covariance matrix.
    :param cov: The n x n covariance matrix
    :return: The n x n correlation matrix
    """
    # the vector of volatilities
    v = np.sqrt(np.diag(cov))
    return cov / np.outer(v, v)


def variance(cov, w):
    """
    Compute the variance w^T cov w
    :param cov: The covariance (sub) n x n matrix cov
    :param w: The (column-) weight-vector w of length n
    :return: The variance w^T * cov * w
    """
    # compute w^T * cov * w
    return np.linalg.multi_dot((w, cov, w))


def sub(cov, idx):
    """
    Get the square sub-matrix of cov induced by the indices idx
    :param cov: The matrix cov
    :param idx: The desired rows and columns of the matrix cov
    :return: the square sub matrix of cov
    """
    # get square sub-matrix of A
    return cov[idx, :][:, idx]


def dist(cor):
    """
    Compute the correlation based distance matrix d, compare with page 239 of the first book by Marcos
    :param cor: the n x n correlation matrix
    :return: The matrix d indicating the distance between column i and i. Note that all the diagonal entries are zero.

    """
    # This is DANGEROUS! 1.0 - 1.0 could easily be < 0, hence we clip the result before we take the square root
    # if isinstance(cor, pd.DataFrame):
    #    return pd.DataFrame(index=cor.index, columns=cor.keys(), data=np.sqrt(np.clip((1.0 - cor) / 2., a_min=0.0, a_max=1.0)))

    return np.sqrt(np.clip((1.0 - cor) / 2., a_min=0.0, a_max=1.0))
    # https://stackoverflow.com/questions/18952587/
    #return ssd.squareform(((1 - cor) / 2) ** 0.5)