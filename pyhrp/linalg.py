import numpy as np
import pandas as pd
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
