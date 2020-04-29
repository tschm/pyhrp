import numpy as np
import pandas as pd

from pyhrp.linalg import dist


def test_dist():
    a = np.array([[1.0, 0.2 / np.sqrt(2.0)], [0.2/np.sqrt(2.0), 1.0]])
    np.testing.assert_allclose(dist(a), np.array([6.552017e-01]), rtol=1e-6, atol=1e-6)
