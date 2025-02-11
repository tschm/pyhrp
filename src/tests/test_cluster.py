from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyhrp.cluster import Cluster


def test_cluster_simple():
    c = Cluster(id=2)
    assert c.is_leaf()


def test_negative_variance():
    with pytest.raises(ValueError):
        c = Cluster(id=1)
        c.variance = -1


def test_left_right():
    left = Cluster(id=1)
    left["A"] = 0.2
    left["B"] = 0.8
    left.variance = 2.0

    assert left["A"] == 0.2
    assert left["B"] == 0.8

    right = Cluster(id=2)
    right["C"] = 0.2
    right["D"] = 0.5
    right["F"] = 0.3
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

    assert c.leaves == {1, 2}


def test_riskparity():
    left = Cluster(id=1)
    left["A"] = 1.0
    left.variance = 4.0

    right = Cluster(id=0)
    # assets={"A": 1.0}, variance=4)
    right["B"] = 1.0
    right.variance = 2.0

    # right = Cluster(assets={"B": 1.0}, variance=2)
    cov = np.array([[2.0, 1.0], [1.0, 4.0]])
    cov = pd.DataFrame(data=cov, index=["B", "A"], columns=["B", "A"])

    cl = Cluster(id=3, left=left, right=right)

    cluster = cl.risk_parity(cov=cov)

    np.testing.assert_allclose(cluster.weights, np.array([1.0, 2.0]) / 3.0)
    np.testing.assert_almost_equal(cluster.variance, 1.7777777777777777)
    np.testing.assert_almost_equal(
        cluster.variance,
        (1.0 / 3.0) ** 2 * 4 + (2.0 / 3.0) ** 2 * 2.0 + 2.0 * (1.0 / 3.0) * (2.0 / 3.0),
    )
    np.testing.assert_almost_equal(cluster.variance, (4.0 / 9.0) + (8.0 / 9.0) + (4.0 / 9.0))


def test_distance(distance):
    left = Cluster(id=0)
    right = Cluster(id=10)

    x = left.distance(distance_matrix=distance, other=right, method="average")
    assert x == pytest.approx(0.6377218246354981)

    x = left.distance(distance_matrix=distance, other=right, method="single")
    assert x == pytest.approx(0.6377218246354981)

    x = left.distance(distance_matrix=distance, other=right, method="complete")
    assert x == pytest.approx(0.6377218246354981)

    x = left.distance(distance_matrix=distance, other=right, method="ward")
    assert x == pytest.approx(0.450937426710419)

    with pytest.raises(ValueError):
        left.distance(distance_matrix=distance, other=right, method="dunno")
