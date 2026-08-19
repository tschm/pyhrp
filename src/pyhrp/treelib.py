"""A lightweight binary tree implementation to replace the binarytree dependency.

This module provides a simple Node class that can be used to create binary trees.
It implements only the functionality needed by the pyhrp package.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar

# Type for node values
NodeValue = int | float | str

T = TypeVar("T", bound=NodeValue)

__all__ = ["Node"]


class Node(Generic[T]):
    """A binary tree node with left and right children.

    This class implements the minimal functionality needed from the binarytree.Node class
    that is used in the pyhrp package.

    Attributes:
        value: The value of the node
        left: The left child node
        right: The right child node
    """

    def __init__(self, value: T, left: Node[T] | None = None, right: Node[T] | None = None) -> None:
        """Initialize a new Node.

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
        """Check if this node is a leaf node (has no children).

        Returns:
            bool: True if this is a leaf node, False otherwise
        """
        return self.left is None and self.right is None

    @property
    def leaves(self) -> Sequence[Node[T]]:
        """Get all leaf nodes in the tree rooted at this node.

        Returns:
            List[Node]: List of all leaf nodes
        """
        # Iterative depth-first walk with an explicit stack: the leaf order is the
        # same left-to-right order the recursive form produced, but the traversal
        # depth is bounded by the heap rather than by sys.getrecursionlimit(). A
        # chain-degenerate cluster tree is as deep as it is wide, so recursing here
        # put a ceiling on the number of assets the package could handle.
        result: list[Node[T]] = []
        stack: list[Node[T]] = [self]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                result.append(node)
                continue
            # Right first, so the left subtree is popped and emitted first.
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

        return result

    @property
    def levels(self) -> list[list[Node[T]]]:
        """Get nodes by level in the tree.

        Returns:
            List[List[Node]]: List of lists of nodes at each level
        """
        result: list[list[Node[T]]] = []
        current_level: list[Node[T]] = [self]

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
        """Count the number of leaf nodes in the tree.

        Returns:
            int: Number of leaf nodes
        """
        return len(self.leaves)

    @property
    def size(self) -> int:
        """Count the total number of nodes in the tree.

        Returns:
            int: Total number of nodes
        """
        # Counts via __iter__, which is already an iterative level-order walk, so
        # this inherits its freedom from the recursion limit.
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[Node[T]]:
        """Iterate through all nodes in the tree in level-order.

        Returns:
            Iterator[Node]: Iterator over all nodes
        """
        queue: deque[Node[T]] = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
