import numpy as np

from pyhrp.hrp import hrp_feed, risk_parity, linkage, tree
import numpy.testing as nt

from pyhrp.linalg import dist, correlation_from_covariance


def test_dist():
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])
    a = dist(correlation_from_covariance(cov))
    nt.assert_allclose(a, np.array([[0.000000e+00, 6.552017e-01], [6.552017e-01, 0.0]]), rtol=1e-6, atol=1e-6)


def test_hrp():
    # use a small covariance matrix
    cov = np.array([[1, 0.5, 0], [0.5, 2, 0.0], [0, 0, 3]])

    # we compute the rootnode of a graph here
    # The rootnode points to left and right and has an id attribute.
    link = linkage(dist(correlation_from_covariance(cov)), 'single')
    rootnode = tree(link)

    v, w = hrp_feed(node=rootnode, cov=cov)
    nt.assert_allclose(v, np.linalg.multi_dot([w, cov, w]))
    nt.assert_allclose(w.sum(), 1.0)

    # risk parity in the branch...
    nt.assert_approx_equal(w[0], w[1] * 2)

    wi = np.array([2, 1]) / 3

    v1 = 0.8888888888888888
    v2 = 3.0
    nt.assert_approx_equal(1 * wi[0] ** 2 + 2 * wi[1] ** 2 + wi[0] * wi[1], v1)
    nt.assert_approx_equal(cov[2][2] * 1.0, v2)


def test_risk_parity():
    v1, v2 = 3, 5
    x1, x2 = risk_parity(v_left=v1, v_right=v2)
    nt.assert_approx_equal(x1, 5.0/8.0)
    nt.assert_approx_equal(x2, 3.0/8.0)
    nt.assert_approx_equal(x1 + x2, 1.0)

