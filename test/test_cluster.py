import numpy as np
import pandas as pd
from pyhrp.cluster import Cluster, risk_parity
import pytest


def test_cluster_simple():
    c = Cluster(assets=[3, 5], variance=1, weights=np.array([0.2, 0.8]))


def test_negative_weight():
    with pytest.raises(AssertionError):
        Cluster(assets=[3, 5], variance=1, weights=np.array([-0.2, 1.2]))


def test_negative_variance():
    with pytest.raises(AssertionError):
        Cluster(assets=[3, 5], variance=-1, weights=np.array([0.2, 0.8]))


def test_weights_as_list():
    with pytest.raises(AssertionError):
        Cluster(assets=[3, 5], variance=1, weights=[0.2, 0.8])


def test_mismatch_length():
    with pytest.raises(AssertionError):
        Cluster(assets=[3], variance=1, weights=np.array([0.2, 0.8]))


def test_leverage():
    with pytest.raises(AssertionError):
        Cluster(assets=[3], variance=1, weights=np.array([0.5, 0.8]))


def test_only_left():
    with pytest.raises(AssertionError):
        Cluster(assets=[3, 5], variance=1, weights=np.array([0.5, 0.8]),
                left=Cluster(assets=[6], variance=1, weights=np.array([1.0])))


def test_wrong_type():
    with pytest.raises(AssertionError):
        Cluster(assets=[3, 5], variance=1, weights=np.array([1.0, 0.0]), left=5, right=5)


def test_left_right():
    left = Cluster(assets=[4, 2], variance=2.0, weights=np.array([0.2, 0.8]))
    right = Cluster(assets=[0, 1, 5], variance=3.0, weights=np.array([0.2, 0.5, 0.3]))

    c = Cluster(assets=[4, 2, 0, 1, 5], variance=2.5, weights=np.array([0.1, 0.4, 0.1, 0.25, 0.15]), left=left,
                right=right)

    assert c.left.is_leaf()
    assert c.right.is_leaf()

    assert c.left
    assert c.right

    assert c.assets == [4, 2, 0, 1, 5]
    assert c.variance == 2.5
    assert c.right.variance == 3.0
    assert c.left.variance == 2.0
    np.testing.assert_array_equal(c.weights, np.array([0.1, 0.4, 0.1, 0.25, 0.15]))

    pd.testing.assert_series_equal(c.weights_series(index=["A", "B", "C", "D", "E", "F"]),
                                   pd.Series({"A": 0.1, "B": 0.25, "C": 0.4, "E": 0.1, "F": 0.15}), check_names=False)


def test_riskparity():
    left = Cluster(assets=[1], variance=4, weights=np.array([1.0]))
    right = Cluster(assets=[0], variance=2, weights=np.array([1.0]))
    cov = np.array([[2.0, 1.0], [1.0, 4.0]])

    cluster = risk_parity(cluster_left=left, cluster_right=right, cov=cov)

    assert isinstance(cluster, Cluster)
    # risk parity implies that left cluster will get 33%
    np.testing.assert_allclose(cluster.weights, np.array([1.0, 2.0])/3.0)
    np.testing.assert_almost_equal(cluster.variance, 1.7777777777777777)
    np.testing.assert_almost_equal(cluster.variance, (1.0/3.0)**2 * 4 + (2.0/3.0)**2 * 2.0 + 2.0*(1.0/3.0)*(2.0/3.0))
    np.testing.assert_almost_equal(cluster.variance, (4.0 / 9.0) + (8.0 / 9.0) + (4.0 / 9.0))