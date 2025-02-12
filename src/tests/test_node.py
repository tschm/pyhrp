from pyhrp.cluster import Cluster


def test_node():
    left = Cluster(id=0)
    right = Cluster(id=1)

    assert left.id == 0
    assert right.id == 1
    assert left.size == 1
    assert right.size == 1

    assert left.preorder == [left]
    assert right.preorder == [right]

    assert left.is_leaf
    assert right.is_leaf

    node = Cluster(id=5, left=left, right=right)

    assert node.id == 5
    assert node.left.id == 0
    assert node.right.id == 1
    assert node.size == 3
    assert node.preorder == [node, left, right]

    assert not node.is_leaf
