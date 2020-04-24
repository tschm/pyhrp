import numpy as np

from pyhrp.cluster import root
from pyhrp.hrp import dist, hrp_feed
import numpy.testing as nt


def test_dist():
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])
    a = dist(cov)
    nt.assert_allclose(a, np.array([[0.000000e+00, 6.552017e-01], [6.552017e-01, 0.0]]), rtol=1e-6, atol=1e-6)


def test_hrp():
    # use a small covariance matrix
    cov = np.array([[1, 0.2, 0], [0.2, 2, 0.0], [0, 0, 3]])

    # we compute the rootnode of a graph here
    # The rootnode points to left and right and has an id attribute.
    rootnode, link = root(dist(cov), 'ward')

    v, w = hrp_feed(rootnode, cov=cov)
    nt.assert_allclose(v, np.linalg.multi_dot([w, cov, w]))
    nt.assert_allclose(w.sum(), 1.0)
