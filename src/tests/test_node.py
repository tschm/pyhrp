"""Tests for the Cluster class."""

from pyhrp.cluster import Cluster


def test_node() -> None:
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
