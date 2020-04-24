"""
This code is an almost identical copy from
https://github.com/robertmartin8/PyPortfolioOpt

It's main purpose here is to serve as a tool to check the results hrp.py is computing.

The ``hierarchical_portfolio`` module seeks to implement one of the recent advances in
portfolio optimisation – the application of hierarchical clustering models in allocation.

All of the hierarchical classes have a similar API to ``EfficientFrontier``, though since
many hierarchical models currently don't support different objectives, the actual allocation
happens with a call to `optimize()`.

Currently implemented:

- ``HRPOpt`` implements the Hierarchical Risk Parity (HRP) portfolio. Code reproduced with
  permission from Marcos Lopez de Prado (2016).
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd



class HRPOpt(object):
    """
    A HRPOpt object (inheriting from BaseOptimizer) constructs a hierarchical
    risk parity portfolio.

    Instance variables:

    - Inputs
        - ``returns`` - pd.DataFrame

    - Output:

        - ``weights`` - np.ndarray
        - ``clusters`` - linkage matrix corresponding to clustered assets.

    Public methods:

    - ``optimize()`` calculates weights using HRP
    """

    def __init__(self, returns):
        """
        :param returns: asset historical returns
        :type returns: pd.DataFrame
        :raises TypeError: if ``returns`` is not a dataframe
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")

        self.returns = returns
        self.clusters = None


    @staticmethod
    def _get_cluster_var(cov, cluster_items):
        """
        Compute the variance per cluster

        :param cov: covariance matrix
        :type cov: np.ndarray
        :param cluster_items: tickers in the cluster
        :type cluster_items: list
        :return: the variance per cluster
        :rtype: float
        """
        # Compute variance per cluster
        cov_slice = cov.loc[cluster_items, cluster_items]
        weights = 1 / np.diag(cov_slice)  # Inverse variance weights
        weights /= weights.sum()
        return np.linalg.multi_dot((weights, cov_slice, weights))

    @staticmethod
    def _get_quasi_diag(link):
        """
        Sort clustered items by distance

        :param link: linkage matrix after clustering
        :type link: np.ndarray
        :return: sorted list of indices
        :rtype: list
        """
        link = link.astype(int)
        # The new clusters formed
        c = np.arange(link.shape[0]) + link[-1, 3]
        root_id = c[-1]
        d = dict(list(zip(c, link[:, 0:2].tolist())))

        # Unpacks the linkage matrix recursively.
        def recursive_unlink(curr, d):
            """ Start this with curr = root integer """
            if curr in d:
                return [
                    node for parent in d[curr] for node in recursive_unlink(parent, d)
                ]
            else:
                return [curr]

        return recursive_unlink(root_id, d)

    @staticmethod
    def _raw_hrp_allocation(cov, ordered_tickers):
        """
        Given the clusters, compute the portfolio that minimises risk by
        recursively traversing the hierarchical tree from the top.

        :param cov: covariance matrix
        :type cov: np.ndarray
        :param ordered_tickers: list of tickers ordered by distance
        :type ordered_tickers: str list
        :return: raw portfolio weights
        :rtype: pd.Series
        """
        w = pd.Series(1, index=ordered_tickers)
        cluster_items = [ordered_tickers]  # initialize all items in one cluster

        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k]
                for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]  # bi-section
            # For each pair, optimise locally.
            for i in range(0, len(cluster_items), 2):
                first_cluster = cluster_items[i]
                second_cluster = cluster_items[i + 1]
                # Form the inverse variance portfolio for this pair
                first_variance = HRPOpt._get_cluster_var(cov, first_cluster)
                second_variance = HRPOpt._get_cluster_var(cov, second_cluster)
                alpha = 1 - first_variance / (first_variance + second_variance)
                w[first_cluster] *= alpha  # weight 1
                w[second_cluster] *= 1 - alpha  # weight 2
        return w

    def optimize(self):
        """
        Construct a hierarchical risk parity portfolio

        :return: weights for the HRP portfolio
        :rtype: dict
        """
        corr, cov = self.returns.corr(), self.returns.cov()

        # Compute distance matrix, with ClusterWarning fix as
        # per https://stackoverflow.com/questions/18952587/
        dist = ssd.squareform(((1 - corr) / 2) ** 0.5)

        self.clusters = sch.linkage(dist, "single")
        sort_ix = HRPOpt._get_quasi_diag(self.clusters)
        ordered_tickers = corr.index[sort_ix].tolist()
        hrp = HRPOpt._raw_hrp_allocation(cov, ordered_tickers)
        return dict(hrp.sort_index())

