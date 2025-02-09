from pyhrp.hrp import Node, _bisection


def test_node():
    left = Node(id=0)
    right = Node(id=1)

    assert left.id == 0
    assert right.id == 1
    assert left.count == 1
    assert right.count == 1

    assert left.pre_order() == [0]
    assert right.pre_order() == [1]

    assert left.is_leaf()
    assert right.is_leaf()

    node = Node(id=5, left=left, right=right)

    assert node.id == 5
    assert node.left.id == 0
    assert node.right.id == 1
    assert node.left.count == 1
    assert node.right.count == 1
    assert node.count == 2
    assert node.pre_order() == [0, 1]

    assert not node.is_leaf()

    root = _bisection(ids=[0, 1], n=2)
    assert isinstance(root, Node)
    assert root.id == 2
    assert root.left.id == 0
    assert root.right.id == 1
    assert root.pre_order() == [0, 1]


def test_bisection():
    root = _bisection(ids=[2, 3, 1, 0, 4, 6, 5, 7], n=8)
    assert root.pre_order() == [2, 3, 1, 0, 4, 6, 5, 7]


def test_repr():
    left = Node(id=0)
    assert repr(left) == "Node(id=0, left=None, right=None)"
