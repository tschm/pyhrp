from pyhrp.algos import generic, one, one_over_n
from pyhrp.cluster import Asset
from pyhrp.cluster import Cluster as Node


def test_one_over_n():
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.right.asset = Asset(name="A")

    root.left.left = Node(4)
    root.left.left.asset = Asset(name="B")

    root.left.right = Node(5)
    root.left.right.asset = Asset(name="C")

    for level, portfolio in one_over_n(root).items():
        print(f"Level: {level}")
        print(portfolio)


def test_generic():
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.right.asset = Asset(name="A")

    root.left.left = Node(4)
    root.left.left.asset = Asset(name="B")

    root.left.right = Node(5)
    root.left.right.asset = Asset(name="C")

    for level, portfolio in generic(root, fct=one).items():
        print(f"Level: {level}")
        print(portfolio)
