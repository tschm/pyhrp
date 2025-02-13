from __future__ import annotations

import numpy as np
import pandas as pd

from pyhrp.algos import risk_parity
from pyhrp.cluster import Cluster


def test_riskparity():
    left = Cluster(value=1)
    left.portfolio["A"] = 1.0
    left.portfolio.variance = 4.0

    right = Cluster(value=0)
    right.portfolio["B"] = 1.0
    right.portfolio.variance = 2.0

    cov = np.array([[2.0, 1.0], [1.0, 4.0]])
    cov = pd.DataFrame(data=cov, index=["B", "A"], columns=["B", "A"])

    cl = Cluster(value=3, left=left, right=right)

    cluster = risk_parity(cl, cov=cov)

    np.testing.assert_allclose(cluster.portfolio.weights, np.array([1.0, 2.0]) / 3.0)
    np.testing.assert_almost_equal(cluster.portfolio.variance, 1.7777777777777777)
    np.testing.assert_almost_equal(
        cluster.portfolio.variance,
        (1.0 / 3.0) ** 2 * 4 + (2.0 / 3.0) ** 2 * 2.0 + 2.0 * (1.0 / 3.0) * (2.0 / 3.0),
    )
    np.testing.assert_almost_equal(cluster.portfolio.variance, (4.0 / 9.0) + (8.0 / 9.0) + (4.0 / 9.0))
