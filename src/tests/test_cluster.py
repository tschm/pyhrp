from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyhrp.algos import risk_parity
from pyhrp.cluster import Cluster


def test_cluster_simple():
    c = Cluster(id=2)
    assert c.is_leaf


def test_negative_variance():
    with pytest.raises(ValueError):
        c = Cluster(id=1)
        c.portfolio.variance = -1


def test_left_right():
    left = Cluster(id=1)
    left.portfolio["A"] = 0.2
    left.portfolio["B"] = 0.8
    left.variance = 2.0

    assert left.portfolio["A"] == 0.2
    assert left.portfolio["B"] == 0.8

    right = Cluster(id=2)
    right.portfolio["C"] = 0.2
    right.portfolio["D"] = 0.5
    right.portfolio["F"] = 0.3
    right.variance = 3.0

    c = Cluster(
        id=3,
        left=left,
        right=right,
    )

    assert c.left.is_leaf
    assert c.right.is_leaf

    assert c.left
    assert c.right

    assert c.right.variance == 3.0
    assert c.left.variance == 2.0

    assert c.leaves == [left, right]

    # assert not left.__eq__(right)
    # assert left.__ne__(right)

    # assert not left.__eq__(right)
    # assert not left.__eq__("Peter Maffay")


def test_riskparity():
    left = Cluster(id=1)
    left.portfolio["A"] = 1.0
    left.portfolio.variance = 4.0

    right = Cluster(id=0)
    # assets={"A": 1.0}, variance=4)
    right.portfolio["B"] = 1.0
    right.portfolio.variance = 2.0

    # right = Cluster(assets={"B": 1.0}, variance=2)
    cov = np.array([[2.0, 1.0], [1.0, 4.0]])
    cov = pd.DataFrame(data=cov, index=["B", "A"], columns=["B", "A"])

    cl = Cluster(id=3, left=left, right=right)

    cluster = risk_parity(cl, cov=cov)

    np.testing.assert_allclose(cluster.portfolio.weights, np.array([1.0, 2.0]) / 3.0)
    np.testing.assert_almost_equal(cluster.portfolio.variance, 1.7777777777777777)
    np.testing.assert_almost_equal(
        cluster.portfolio.variance,
        (1.0 / 3.0) ** 2 * 4 + (2.0 / 3.0) ** 2 * 2.0 + 2.0 * (1.0 / 3.0) * (2.0 / 3.0),
    )
    np.testing.assert_almost_equal(cluster.portfolio.variance, (4.0 / 9.0) + (8.0 / 9.0) + (4.0 / 9.0))
