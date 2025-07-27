"""Tests for the treelib module."""

from pyhrp.treelib import Node


def test_node_init() -> None:
    """Test Node initialization."""
    # Test creating a node with just a value
    node = Node(value=1)
    assert node.value == 1
    assert node.left is None
    assert node.right is None

    # Test creating a node with left and right children
    left = Node(value=2)
    right = Node(value=3)
    parent = Node(value=1, left=left, right=right)
    assert parent.value == 1
    assert parent.left is left
    assert parent.right is right


def test_is_leaf() -> None:
    """Test the is_leaf property."""
    # A node with no children is a leaf
    leaf = Node(value=1)
    assert leaf.is_leaf is True

    # A node with only a left child is not a leaf
    left_only = Node(value=1, left=Node(value=2))
    assert left_only.is_leaf is False

    # A node with only a right child is not a leaf
    right_only = Node(value=1, right=Node(value=3))
    assert right_only.is_leaf is False

    # A node with both children is not a leaf
    parent = Node(value=1, left=Node(value=2), right=Node(value=3))
    assert parent.is_leaf is False


def test_leaves() -> None:
    """Test the leaves property."""
    # A leaf node returns itself as the only leaf
    leaf = Node(value=1)
    assert leaf.leaves == [leaf]

    # Create a simple tree
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    leaf4 = Node(value=4)
    leaf5 = Node(value=5)
    node2 = Node(value=2, left=leaf4, right=leaf5)
    leaf3 = Node(value=3)
    root = Node(value=1, left=node2, right=leaf3)

    # Check leaves
    assert root.leaves == [leaf4, leaf5, leaf3]
    assert node2.leaves == [leaf4, leaf5]
    assert leaf3.leaves == [leaf3]


def test_levels() -> None:
    """Test the levels property."""
    # Create a simple tree
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    leaf4 = Node(value=4)
    leaf5 = Node(value=5)
    node2 = Node(value=2, left=leaf4, right=leaf5)
    leaf3 = Node(value=3)
    root = Node(value=1, left=node2, right=leaf3)

    # Check levels
    levels = root.levels
    assert len(levels) == 3
    assert levels[0] == [root]
    assert levels[1] == [node2, leaf3]
    assert levels[2] == [leaf4, leaf5]

    # Test a single node
    single = Node(value=1)
    assert single.levels == [[single]]


def test_leaf_count() -> None:
    """Test the leaf_count property."""
    # A leaf node has a leaf count of 1
    leaf = Node(value=1)
    assert leaf.leaf_count == 1

    # Create a simple tree
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    leaf4 = Node(value=4)
    leaf5 = Node(value=5)
    node2 = Node(value=2, left=leaf4, right=leaf5)
    leaf3 = Node(value=3)
    root = Node(value=1, left=node2, right=leaf3)

    # Check leaf counts
    assert root.leaf_count == 3
    assert node2.leaf_count == 2
    assert leaf3.leaf_count == 1


def test_size() -> None:
    """Test the size property."""
    # A single node has size 1
    leaf = Node(value=1)
    assert leaf.size == 1

    # Create a simple tree
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    leaf4 = Node(value=4)
    leaf5 = Node(value=5)
    node2 = Node(value=2, left=leaf4, right=leaf5)
    leaf3 = Node(value=3)
    root = Node(value=1, left=node2, right=leaf3)

    # Check sizes
    assert root.size == 5
    assert node2.size == 3
    assert leaf3.size == 1


def test_iter() -> None:
    """Test the __iter__ method."""
    # Create a simple tree
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    leaf4 = Node(value=4)
    leaf5 = Node(value=5)
    node2 = Node(value=2, left=leaf4, right=leaf5)
    leaf3 = Node(value=3)
    root = Node(value=1, left=node2, right=leaf3)

    # Collect nodes in level-order traversal
    nodes = list(root)

    # Check level-order traversal
    assert len(nodes) == 5
    assert nodes[0] is root
    assert nodes[1] is node2
    assert nodes[2] is leaf3
    assert nodes[3] is leaf4
    assert nodes[4] is leaf5

    # Test iteration on a single node
    single = Node(value=1)
    assert list(single) == [single]