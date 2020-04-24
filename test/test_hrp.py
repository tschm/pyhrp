import numpy as np

from pyhrp.cluster import root
from pyhrp.hrp import hrp_feed, split
import numpy.testing as nt

from pyhrp.linalg import dist


def test_dist():
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])
    a = dist(cov)
    nt.assert_allclose(a, np.array([[0.000000e+00, 6.552017e-01], [6.552017e-01, 0.0]]), rtol=1e-6, atol=1e-6)


def test_hrp():
    # use a small covariance matrix
    cov = np.array([[1, 0.5, 0], [0.5, 2, 0.0], [0, 0, 3]])

    # we compute the rootnode of a graph here
    # The rootnode points to left and right and has an id attribute.
    rootnode, link = root(dist(cov), 'single')

    v, w = hrp_feed(rootnode, cov=cov)
    nt.assert_allclose(v, np.linalg.multi_dot([w, cov, w]))
    nt.assert_allclose(w.sum(), 1.0)

    # risk parity in the branch...
    nt.assert_approx_equal(w[0], w[1]*2)

    wi = np.array([2, 1])/3

    v1 = 0.8888888888888888
    v2 = 3.0
    nt.assert_approx_equal(1*wi[0]**2 + 2*wi[1]**2 + wi[0]*wi[1], v1)
    nt.assert_approx_equal(cov[2][2]*1.0, v2)

    alpha, beta = split(v1, v2)
    nt.assert_approx_equal(alpha, 0.7714285714285715)
    nt.assert_approx_equal(v1*alpha, v2*beta)


def test_split():
    v1 = 3
    v2 = 5
    x1,x2 = split(v_left=v1, v_right=v2)
    nt.assert_allclose(np.array([x1,x2]), np.array([5.0, 3.0])/8.0)
    nt.assert_approx_equal(x1 + x2, 1.0)
    

def test_add():
    x = [1,2,3]
    y = [4,5]
    assert x+y == [1,2,3,4,5]


