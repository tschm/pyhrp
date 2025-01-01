from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyhrp.cluster import Cluster, risk_parity


def test_cluster_simple():
    c = Cluster(assets={"A": 0.2, "B": 0.8}, variance=1)
    assert c.is_leaf()


def test_negative_variance():
    with pytest.raises(AssertionError):
        Cluster(assets={"A": -0.2, "B": 0.8}, variance=-1)


def test_only_left():
    with pytest.raises(AssertionError):
        Cluster(
            assets={"A": 0.5, "B": 0.5},
            variance=1,
            left=Cluster(assets={"C": 1.0}, variance=1),
        )


def test_wrong_type():
    with pytest.raises(AssertionError):
        Cluster(assets={"A": 0.5, "B": 0.5}, variance=1, left=5, right=5)


def test_left_right():
    left = Cluster(assets={"A": 0.2, "B": 0.8}, variance=2.0)
    right = Cluster(assets={"C": 0.2, "D": 0.5, "F": 0.3}, variance=3.0)

    c = Cluster(
        assets={"A": 0.1, "B": 0.4, "C": 0.1, "D": 0.25, "F": 0.15},
        variance=2.5,
        left=left,
        right=right,
    )

    assert c.left.is_leaf
    assert c.right.is_leaf

    assert c.left
    assert c.right

    assert c.assets == {"A": 0.1, "B": 0.4, "C": 0.1, "D": 0.25, "F": 0.15}
    assert c.variance == 2.5
    assert c.right.variance == 3.0
    assert c.left.variance == 2.0

    pd.testing.assert_series_equal(c.weights, pd.Series(c.assets), check_names=False)


def test_riskparity():
    left = Cluster(assets={"A": 1.0}, variance=4)
    right = Cluster(assets={"B": 1.0}, variance=2)
    cov = np.array([[2.0, 1.0], [1.0, 4.0]])
    cov = pd.DataFrame(data=cov, index=["B", "A"], columns=["B", "A"])

    cluster = risk_parity(cluster_left=left, cluster_right=right, cov=cov)

    assert isinstance(cluster, Cluster)
    # risk parity implies that left cluster will get 33%
    np.testing.assert_allclose(cluster.weights, np.array([1.0, 2.0]) / 3.0)
    np.testing.assert_almost_equal(cluster.variance, 1.7777777777777777)
    np.testing.assert_almost_equal(
        cluster.variance,
        (1.0 / 3.0) ** 2 * 4 + (2.0 / 3.0) ** 2 * 2.0 + 2.0 * (1.0 / 3.0) * (2.0 / 3.0),
    )
    np.testing.assert_almost_equal(cluster.variance, (4.0 / 9.0) + (8.0 / 9.0) + (4.0 / 9.0))
