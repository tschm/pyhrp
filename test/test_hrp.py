import numpy as np
import pandas as pd

from pyhrp.hrp import linkage, tree, hrp, dist
from test.config import resource, get_data


# def test_hrp():
#     # use a small covariance matrix
#     cov = np.array([[1, 0.5, 0], [0.5, 2, 0.0], [0, 0, 3]])
#
#     # we compute the root of a graph here
#     # The root points to left and right and has an id attribute.
#     link = linkage(dist(correlation_from_covariance(cov)), 'single')
#     root = tree(link)
#
#     v, w = hrp_feed(node=root, cov=cov)
#     nt.assert_allclose(v, np.linalg.multi_dot([w, cov, w]))
#     nt.assert_allclose(w.sum(), 1.0)
#
#     # risk parity in the branch...
#     nt.assert_approx_equal(w[0], w[1] * 2)


# def test_hrp2():
#     # use a small covariance matrix
#     cov = np.array([[1, 0.5, 0], [0.5, 2, 0.0], [0, 0, 3]])
#
#     # we compute the root of a graph here
#     # The root points to left and right and has an id attribute.
#     link = linkage(dist(correlation_from_covariance(cov)), 'single')
#     root = tree(link)
#
#     root = hrp_feed2(node=root, cov=cov)
#
#     nt.assert_allclose(root.variance,
#                        np.linalg.multi_dot([root.weights_series().sort_index().values, cov, root.weights_series().values]))
#     nt.assert_allclose(root.weights.sum(), 1.0)
#
#     # risk parity in the branch...
#     nt.assert_approx_equal(root.weights_series()[0], root.weights_series()[1] * 2)
#
#     # you can now drill into the subclusters
#     assert root.left.assets == [2]
#     assert root.right.assets == [0, 1]
#     print(root.right.assets)
#     print(root.right.weights)
#     print(root.right.weights)
#
#     nt.assert_allclose(root.right.weights, np.array([2.0 / 3.0, 1.0 / 3.0]))
def test_dist():
    a = np.array([[1.0, 0.2 / np.sqrt(2.0)], [0.2/np.sqrt(2.0), 1.0]])
    np.testing.assert_allclose(dist(a), np.array([6.552017e-01]), rtol=1e-6, atol=1e-6)


def test_quasi_diag():
    prices = get_data().truncate(before="2017-01-01")

    # compute returns
    returns = prices.pct_change().dropna(axis=0, how="all").fillna(0.0)

    np.testing.assert_allclose(returns.cov().values, np.genfromtxt(resource("covariance2.csv")))
    np.testing.assert_allclose(returns.corr().values, np.genfromtxt(resource("correlation2.csv")))

    cor = returns.corr().values
    links = linkage(dist(cor), method="single")

    # uncomment this line if you want to generate a new test resource
    # np.savetxt(resource("links.csv"), links, delimiter=",")

    np.testing.assert_array_almost_equal(links, np.loadtxt(resource("links.csv"), delimiter=','))

    node = tree(links)

    ids = node.pre_order()
    assert ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]

    ordered_tickers = prices.keys()[ids].to_list()
    print(ordered_tickers)
    assert ordered_tickers == ['UAA',
                               'WMT',
                               'SBUX',
                               'AMD',
                               'RRC',
                               'GE',
                               'T',
                               'XOM',
                               'BABA',
                               'AAPL',
                               'AMZN',
                               'MA',
                               'GOOG',
                               'FB',
                               'PFE',
                               'GM',
                               'BAC',
                               'JPM',
                               'SHLD',
                               'BBY']


def test_hrp():
    prices = get_data()

    root = hrp(prices=prices, method="single")

    # uncomment this line if you want generating a new file
    root2 = hrp(prices=prices, method="ward")
    root2.weights.to_csv(resource("weights_hrp2.csv"), header=False)

    x = pd.read_csv(resource("weights_hrp.csv"), squeeze=True, index_col=0, header=None)
    x.name = "Weights"
    x.index.name = None

    pd.testing.assert_series_equal(x, root.weights, check_exact=False)