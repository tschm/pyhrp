"""Tests for the Cluster class and related functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyhrp.algos import risk_parity
from pyhrp.cluster import Cluster


def test_riskparity() -> None:
    """Test the risk parity algorithm implementation.

    This test verifies:
    1. Creation of Asset and Cluster objects
    2. Portfolio assignment to clusters
    3. Risk parity calculation with a simple covariance matrix
    4. Resulting portfolio weights and variance calculation
    """
    # Create left cluster with asset A
    left = Cluster(value=1)
    left.portfolio["A"] = 1.0

    # Create right cluster with asset B
    right = Cluster(value=0)
    right.portfolio["B"] = 1.0

    # Create covariance matrix
    # [[2.0, 1.0],
    #  [1.0, 4.0]]
    cov = np.array([[2.0, 1.0], [1.0, 4.0]])
    cov = pd.DataFrame(data=cov, index=["B", "A"], columns=["B", "A"])

    # Create parent cluster
    cl = Cluster(value=2, left=left, right=right)

    # Apply risk parity algorithm
    cluster = risk_parity(cl, cov=cov)

    # Verify the resulting portfolio weights
    # Expected weights: [1/3, 2/3]
    np.testing.assert_allclose(cluster.portfolio.weights.values, np.array([1.0, 2.0]) / 3.0)

    # Verify the resulting portfolio variance
    np.testing.assert_almost_equal(cluster.portfolio.variance(cov), 1.7777777777777777)
