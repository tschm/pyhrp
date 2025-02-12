from pyhrp.cluster import Cluster


def test_node():
    left = Cluster(value=0)
    right = Cluster(value=1)

    assert left.value == 0
    assert right.value == 1
    assert left.size == 1
    assert right.size == 1

    assert left.preorder == [left]
    assert right.preorder == [right]

    assert left.is_leaf
    assert right.is_leaf

    node = Cluster(value=5, left=left, right=right)

    assert node.value == 5
    assert node.left.value == 0
    assert node.right.value == 1
    assert node.size == 3
    assert node.preorder == [node, left, right]

    assert not node.is_leaf
