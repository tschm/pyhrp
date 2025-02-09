from pyhrp.hrp import Node


def test_node():
    left = Node(id=0)
    right = Node(id=1)

    assert left.id == 0
    assert right.id == 1
    assert len(left) == 1
    assert len(right) == 1

    assert left.pre_order() == [0]
    assert right.pre_order() == [1]

    assert left.is_leaf()
    assert right.is_leaf()

    node = Node(id=5, left=left, right=right)

    assert node.id == 5
    assert node.left.id == 0
    assert node.right.id == 1
    assert len(node.left) == 1
    assert len(node.right) == 1
    assert len(node) == 2
    assert node.pre_order() == [0, 1]

    assert not node.is_leaf()
