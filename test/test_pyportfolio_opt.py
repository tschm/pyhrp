# import numpy as np
# import pytest
# import pandas as pd
#
# from pypfopt import HRPOpt
#
# from pyhrp.graph import dendrogram
# from pyhrp.hrp import linkage, tree, hrp_feed
# from pyhrp.linalg import correlation_from_covariance, dist, variance
# from test.config import get_data, resource
# import pandas.testing as pdt
# import scipy.spatial.distance as ssd
# import numpy.testing as nt
# import scipy.cluster.hierarchy as sch
#
#
# def _raw_hrp_allocation(cov, ordered_tickers):
#     """
#     Given the clusters, compute the portfolio that minimises risk by
#     recursively traversing the hierarchical tree from the top.
#
#     :param cov: covariance matrix
#     :type cov: np.ndarray
#     :param ordered_tickers: list of tickers ordered by distance
#     :type ordered_tickers: str list
#     :return: raw portfolio weights
#     :rtype: pd.Series
#     """
#     w = pd.Series(1, index=ordered_tickers)
#     cluster_items = [ordered_tickers]  # initialize all items in one cluster
#
#     while len(cluster_items) > 0:
#         cluster_items = [
#             i[j:k]
#             for i in cluster_items
#             for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
#             if len(i) > 1
#         ]  # bi-section
#         # For each pair, optimise locally.
#         for i in range(0, len(cluster_items), 2):
#             first_cluster = cluster_items[i]
#             second_cluster = cluster_items[i + 1]
#             print(first_cluster, second_cluster)
#
#             # Form the inverse variance portfolio for this pair
#             first_variance = HRPOpt._get_cluster_var(cov, first_cluster)
#             second_variance = HRPOpt._get_cluster_var(cov, second_cluster)
#             print(first_variance, second_variance)
#
#             alpha = 1 - first_variance / (first_variance + second_variance)
#             w[first_cluster] *= alpha  # weight 1
#             w[second_cluster] *= 1 - alpha  # weight 2
#     return w
#
#
# def test_hrp_portfolio():
#     df = get_data()
#     returns = df.pct_change().dropna(how="all")
#     hrp = HRPOpt(returns)
#     w = hrp.optimize()
#     x = pd.Series(w)
#     print(x["JPM"])
#     print(x["BAC"])
#     print(x["JPM"]/x["BAC"])
#     print(x["SHLD"])
#     assert isinstance(w, dict)
#     assert set(w.keys()) == set(df.columns)
#     np.testing.assert_almost_equal(sum(w.values()), 1)
#     assert all([i >= 0 for i in w.values()])
#
#     cov = returns.cov()
#     cor = returns.corr()
#
#     d = ssd.squareform(((1 - cor) / 2) ** 0.5)
#     print(d)
#     print(cov.shape)
#     print(d.shape)
#     dd = ssd.squareform(dist(cor))
#     nt.assert_allclose(d, dd)
#
#     #nt.assert_allclose(linkage(dist=dist(cor)), linkage(dist=ssd.squareform(dist(cor))))
#
#     #assert False
#
#     d = ssd.squareform(dist(cor))
#
#     links = linkage(d, method="single")
#     root = tree(linkage=links)
#
#     # make sure you can recompute the clusters
#     # nt.assert_allclose(links, hrp.clusters)
#
#     #var, weights = hrp_feed(node=root, cov=cov.values)
#     #weights = pd.Series(index=cov.index, data=weights).sort_index()
#     #print(weights)
#
#     #sort_ix = HRPOpt._get_quasi_diag(hrp.clusters)
#     #print(sort_ix)
#     #print(tree(links).pre_order())
#
#     #links = linkage(d, method="single", optimal_ordering=False)
#     assert sch.to_tree(hrp.clusters, rd=False).pre_order() == HRPOpt._get_quasi_diag(hrp.clusters)
#     assert False
#
#     # plot the dendrogram
#     ax = dendrogram(links, labels=cov.keys())
#     ax.get_figure().savefig(resource("dendrogram.png"))
#     hrp.plot_dendrogram(show_tickers=True, filename=resource("dendrogram2.png"))
#     print(weights["JPM"])
#     print(weights["BAC"])
#     print(weights["JPM"]/weights["BAC"])
#     #print(tree(links).pre_order())
#     print(weights["SHLD"])
#     print(var)
#     #print(cov)
#     print(variance(w=weights, cov=cov))
#     sort_ix = HRPOpt._get_quasi_diag(hrp.clusters)
#
#     ordered_tickers = cor.index[sort_ix].tolist()
#
#     print(_raw_hrp_allocation(cov, ordered_tickers))
#
#     assert False
#
