import pytest

from pyhrp.algos import one_over_n
from pyhrp.cluster import Asset
from pyhrp.cluster import Cluster as Node
from pyhrp.hrp import Dendrogram


def test_one_over_n():
    root = Node(10)
    root.left = Node(11)
    root.right = Node(0)
    a = Asset(name="A")
    # root.right.asset = a

    root.left.left = Node(1)
    b = Asset(name="B")
    # root.left.left.asset = b

    root.left.right = Node(2)
    c = Asset(name="C")
    # root.left.right.asset = c
    print(root)

    dendrogram = Dendrogram(root=root, assets=[a, b, c])

    for level, portfolio in one_over_n(dendrogram):
        print(f"Level: {level}")
        print(portfolio)


def test_wrong_number_of_nodes():
    root = Node(10)
    root.left = Node(11)
    root.right = Node(0)
    root.left.left = Node(1)
    root.left.right = Node(2)

    a = Asset(name="A")
    b = Asset(name="B")
    c = Asset(name="C")
    d = Asset(name="D")

    with pytest.raises(ValueError):
        Dendrogram(root=root, assets=[a, b, c, d])
