from __future__ import annotations

import pytest

from pyhrp.hrp import Cluster


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

    # assert c.assets == {"A": 0.1, "B": 0.4, "C": 0.1, "D": 0.25, "F": 0.15}
    # assert c.variance == 2.5
    assert c.right.variance == 3.0
    assert c.left.variance == 2.0

    # pd.testing.assert_series_equal(c.weights, pd.Series(c.assets), check_names=False)
