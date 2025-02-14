from __future__ import annotations

import numpy as np
import pandas as pd

from pyhrp.algos import risk_parity
from pyhrp.cluster import Asset, Cluster


def test_riskparity():
    a = Asset(name="A")
    b = Asset(name="B")
    left = Cluster(value=1)
    left.portfolio[a] = 1.0

    right = Cluster(value=0)
    right.portfolio[b] = 1.0

    cov = np.array([[2.0, 1.0], [1.0, 4.0]])
    cov = pd.DataFrame(data=cov, index=[b, a], columns=[b, a])

    cl = Cluster(value=2, left=left, right=right)

    cluster = risk_parity(cl, cov=cov)

    np.testing.assert_allclose(cluster.portfolio.weights.values, np.array([1.0, 2.0]) / 3.0)
    np.testing.assert_almost_equal(cluster.portfolio.variance(cov), 1.7777777777777777)
