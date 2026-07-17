"""Tests for the Portfolio and Cluster data structures."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import pytest

from pyhrp.cluster import Cluster, Portfolio
from pyhrp.treelib import Node


class TestPortfolio:
    """Tests for the Portfolio weight container."""

    def test_portfolio(self) -> None:
        """Test portfolio creation, plotting, and variance calculation.

        This test verifies:
        1. Portfolio creation and weight assignment
        2. Portfolio plotting functionality
        3. Portfolio variance calculation with a covariance matrix
        """
        p = Portfolio()
        a = "A"
        b = "B"
        c = "C"

        p[a] = 0.4
        p[b] = 0.3
        p[c] = 0.3

        # Diagonal covariance matrix (columns: A, B, C — rows in same order)
        cov = pl.DataFrame({"A": [2.0, 0.0, 0.0], "B": [0.0, 3.0, 0.0], "C": [0.0, 0.0, 4.0]})

        fig: go.Figure = p.plot(names=["A", "B", "C"])
        assert fig is not None

        # Expected variance: 0.4^2 * 2 + 0.3^2 * 3 + 0.3^2 * 4 = 0.95
        assert p.variance(cov) == pytest.approx(0.95)

    def test_getset_item(self) -> None:
        """Test the __getitem__ and __setitem__ methods of Portfolio class."""
        p = Portfolio()
        a = "A"

        p[a] = 0.4
        assert p[a] == 0.4


class TestCluster:
    """Tests for the Cluster tree node."""

    def test_node(self) -> None:
        """Test the basic functionality of the Cluster class.

        This test verifies:
        1. Creation of leaf nodes
        2. Basic properties of leaf nodes (value, size, is_leaf)
        3. Creation of a parent node with children
        4. Properties of parent nodes (value, size, is_leaf, leaves)
        """
        # Create two leaf nodes
        left = Cluster(value=0)
        right = Cluster(value=1)

        # Verify leaf node properties
        assert left.value == 0
        assert right.value == 1
        assert left.size == 1
        assert right.size == 1

        assert left.is_leaf
        assert right.is_leaf

        # Create a parent node with the two leaf nodes as children
        node = Cluster(value=5, left=left, right=right)

        # Verify parent node properties
        assert node.value == 5
        assert node.left.value == 0
        assert node.right.value == 1
        assert node.size == 3

        assert not node.is_leaf
        assert node.leaves == [left, right]

    def test_leaves_only_right_child(self) -> None:
        """ValueError is raised when a non-leaf Cluster has only a right child."""
        c = Cluster(value=10)
        c.right = Cluster(value=1)
        with pytest.raises(ValueError, match="Expected left child to exist for non-leaf cluster"):
            _ = c.leaves

    def test_leaves_only_left_child(self) -> None:
        """ValueError is raised when a non-leaf Cluster has only a left child."""
        c = Cluster(value=10)
        c.left = Cluster(value=1)
        with pytest.raises(ValueError, match="Expected right child to exist for non-leaf cluster"):
            _ = c.leaves

    def test_leaves_non_cluster_left(self) -> None:
        """TypeError is raised when leaves encounters a non-Cluster left child."""
        cluster = Cluster(2)
        cluster.left = Node(0)  # type: ignore[assignment]
        cluster.right = Cluster(1)
        with pytest.raises(TypeError, match="Expected left child to be a Cluster"):
            _ = cluster.leaves

    def test_leaves_non_cluster_right(self) -> None:
        """TypeError is raised when leaves encounters a non-Cluster right child."""
        cluster = Cluster(2)
        cluster.left = Cluster(0)
        cluster.right = Node(1)  # type: ignore[assignment]
        with pytest.raises(TypeError, match="Expected right child to be a Cluster"):
            _ = cluster.leaves
