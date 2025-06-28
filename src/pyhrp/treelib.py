"""
A lightweight binary tree implementation to replace the binarytree dependency.

This module provides a simple Node class that can be used to create binary trees.
It implements only the functionality needed by the pyhrp package.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, TypeVar, Union

# Type for node values
NodeValue = Union[int, float, str]
T = TypeVar("T", bound="Node")


class Node:
    """
    A binary tree node with left and right children.

    This class implements the minimal functionality needed from the binarytree.Node class
    that is used in the pyhrp package.

    Attributes:
        value: The value of the node
        left: The left child node
        right: The right child node
    """

    def __init__(self, value: NodeValue, left: Node | None = None, right: Node | None = None):
        """
        Initialize a new Node.

        Args:
            value: The value of the node
            left: The left child node
            right: The right child node
        """
        self.value = value
        self.left = left
        self.right = right

    @property
    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf node (has no children).

        Returns:
            bool: True if this is a leaf node, False otherwise
        """
        return self.left is None and self.right is None

    @property
    def leaves(self) -> list[Node]:
        """
        Get all leaf nodes in the tree rooted at this node.

        Returns:
            List[Node]: List of all leaf nodes
        """
        if self.is_leaf:
            return [self]

        result = []
        if self.left:
            result.extend(self.left.leaves)
        if self.right:
            result.extend(self.right.leaves)

        return result

    @property
    def levels(self) -> list[list[Node]]:
        """
        Get nodes by level in the tree.

        Returns:
            List[List[Node]]: List of lists of nodes at each level
        """
        result = []
        current_level = [self]

        while current_level:
            result.append(current_level)
            next_level = []

            for node in current_level:
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)

            current_level = next_level

        return result

    @property
    def leaf_count(self) -> int:
        """
        Count the number of leaf nodes in the tree.

        Returns:
            int: Number of leaf nodes
        """
        return len(self.leaves)

    @property
    def size(self) -> int:
        """
        Count the total number of nodes in the tree.

        Returns:
            int: Total number of nodes
        """
        size = 1  # Count this node
        if self.left:
            size += self.left.size
        if self.right:
            size += self.right.size
        return size

    def __iter__(self) -> Iterator[Node]:
        """
        Iterate through all nodes in the tree in level-order.

        Returns:
            Iterator[Node]: Iterator over all nodes
        """
        queue: Deque[Node] = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
