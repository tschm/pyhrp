"""Tests for the Cluster class and related functions."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pyhrp.algos import _parity, risk_parity
from pyhrp.cluster import Cluster
from pyhrp.treelib import Node


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

    # Covariance matrix: cov(B,B)=2, cov(A,A)=4, cov(A,B)=cov(B,A)=1
    # Columns: B, A — row 0 = B's row, row 1 = A's row
    cov = pl.DataFrame({"B": [2.0, 1.0], "A": [1.0, 4.0]})

    # Create parent cluster
    cl = Cluster(value=2, left=left, right=right)

    # Apply risk parity algorithm
    cluster = risk_parity(cl, cov=cov)

    # Verify the resulting portfolio weights (alphabetically sorted: A, B)
    # Expected weights: [1/3, 2/3]
    np.testing.assert_allclose(
        np.array(list(cluster.portfolio.weights.values())),
        np.array([1.0, 2.0]) / 3.0,
    )

    # Verify the resulting portfolio variance
    np.testing.assert_almost_equal(cluster.portfolio.variance(cov), 1.7777777777777777)


def test_risk_parity_non_cluster_left() -> None:
    """TypeError is raised when risk_parity encounters a non-Cluster left child."""
    root = Cluster(value=10)
    root.left = Node(1)
    root.right = Cluster(value=2)
    cov = pl.DataFrame({"A": [1.0]})
    with pytest.raises(TypeError, match="Expected left child to be a Cluster"):
        risk_parity(root, cov)


def test_risk_parity_non_cluster_right() -> None:
    """TypeError is raised when risk_parity encounters a non-Cluster right child."""
    root = Cluster(value=10)
    root.left = Cluster(value=1)
    root.right = Node(2)
    cov = pl.DataFrame({"A": [1.0]})
    with pytest.raises(TypeError, match="Expected right child to be a Cluster"):
        risk_parity(root, cov)


def test_parity_non_cluster_left() -> None:
    """TypeError is raised when _parity encounters a non-Cluster left child."""
    cluster = Cluster(value=10)
    cluster.left = Node(1)
    cluster.right = Cluster(value=2)
    cov = pl.DataFrame({"A": [1.0]})
    with pytest.raises(TypeError, match="Expected left child to be a Cluster"):
        _parity(cluster, cov)


def test_parity_non_cluster_right() -> None:
    """TypeError is raised when _parity encounters a non-Cluster right child."""
    cluster = Cluster(value=10)
    cluster.left = Cluster(value=1)
    cluster.right = Node(2)
    cov = pl.DataFrame({"A": [1.0]})
    with pytest.raises(TypeError, match="Expected right child to be a Cluster"):
        _parity(cluster, cov)


def test_leaves_only_right_child() -> None:
    """ValueError is raised when a non-leaf Cluster has only a right child."""
    c = Cluster(value=10)
    c.right = Cluster(value=1)
    with pytest.raises(ValueError, match="Expected left child to exist for non-leaf cluster"):
        _ = c.leaves


def test_leaves_only_left_child() -> None:
    """ValueError is raised when a non-leaf Cluster has only a left child."""
    c = Cluster(value=10)
    c.left = Cluster(value=1)
    with pytest.raises(ValueError, match="Expected right child to exist for non-leaf cluster"):
        _ = c.leaves
