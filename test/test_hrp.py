import numpy as np

from pyhrp.hrp import hrp_feed, risk_parity, linkage, tree, hrp_feed2
import numpy.testing as nt

from pyhrp.linalg import dist, correlation_from_covariance


def test_hrp():
    # use a small covariance matrix
    cov = np.array([[1, 0.5, 0], [0.5, 2, 0.0], [0, 0, 3]])

    # we compute the root of a graph here
    # The root points to left and right and has an id attribute.
    link = linkage(dist(correlation_from_covariance(cov)), 'single')
    root = tree(link)

    v, w = hrp_feed(node=root, cov=cov)
    nt.assert_allclose(v, np.linalg.multi_dot([w, cov, w]))
    nt.assert_allclose(w.sum(), 1.0)

    # risk parity in the branch...
    nt.assert_approx_equal(w[0], w[1] * 2)



def test_hrp2():
    # use a small covariance matrix
    cov = np.array([[1, 0.5, 0], [0.5, 2, 0.0], [0, 0, 3]])

    # we compute the root of a graph here
    # The root points to left and right and has an id attribute.
    link = linkage(dist(correlation_from_covariance(cov)), 'single')
    root = tree(link)

    root = hrp_feed2(node=root, cov=cov)

    nt.assert_allclose(root.variance, np.linalg.multi_dot([root.weights_series.values, cov, root.weights_series.values]))
    nt.assert_allclose(root.weights.sum(), 1.0)

    # risk parity in the branch...
    nt.assert_approx_equal(root.weights_series[0], root.weights_series[1] * 2)

    # you can now drill into the subclusters
    assert root.left.assets == [2]
    assert root.right.assets == [0, 1]
    assert nt.assert_allclose(root.right.weights, np.array([2.0, 1.0])/3.0)



def test_risk_parity():
    v1, v2 = 3, 5
    x1, x2 = risk_parity(v_left=v1, v_right=v2)
    nt.assert_approx_equal(x1, 5.0/8.0)
    nt.assert_approx_equal(x2, 3.0/8.0)
    nt.assert_approx_equal(x1 + x2, 1.0)

