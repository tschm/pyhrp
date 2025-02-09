class Node:
    def __init__(self, id=None, left=None, right=None):
        self.id = id
        self.left = left
        self.right = right

    def pre_order(self):
        if self.left:
            return self.left.pre_order() + self.right.pre_order()
        else:
            return [self.id]

    def __repr__(self):
        return f"Node(id={self.id}, left={self.left}, right={self.right})"

    def is_leaf(self):
        return self.left is None and self.right is None

    @property
    def count(self):
        if self.left:
            return self.left.count + self.right.count
        else:
            return 1.0


def bisection(ids, n: int) -> Node:
    """
    Compute the graph underlying the recursive bisection of Marcos Lopez de Prado.
    Ensures that the pre-order traversal of the tree remains unchanged.

    :param ids: A (ranked) set of indices (leaf nodes).
    :param n: The current ID to assign to the newly created cluster node.
    :return: The root ClusterNode of this tree.
    """

    def split(ids):
        """Split the vector ids into two parts, split in the middle."""
        num = len(ids)
        return ids[: num // 2], ids[num // 2 :]

    # Base case: if there's only one ID, return a leaf node
    if len(ids) == 1:
        return Node(id=ids[0])

    # Split the IDs into left and right halves
    left, right = split(ids)

    # Recursively construct the left and right subtrees
    left_node = bisection(ids=left, n=n + 1)
    right_node = bisection(ids=right, n=n + 1 + len(left))

    # Create a new cluster node with the current ID and the left/right subtrees
    return Node(id=n, left=left_node, right=right_node)
