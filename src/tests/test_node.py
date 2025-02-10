from pyhrp.cluster import Cluster


def test_node():
    left = Cluster(id=0)
    right = Cluster(id=1)

    assert left.id == 0
    assert right.id == 1
    assert left.count == 1
    assert right.count == 1

    assert left.pre_order() == [0]
    assert right.pre_order() == [1]

    assert left.is_leaf()
    assert right.is_leaf()

    node = Cluster(id=5, left=left, right=right)

    assert node.id == 5
    assert node.left.id == 0
    assert node.right.id == 1
    assert node.count == 2
    assert node.pre_order() == [0, 1]

    assert not node.is_leaf()
