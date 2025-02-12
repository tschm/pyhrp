from pyhrp.algos import one_over_n


def test_one_over_n():
    from pyhrp.cluster import Cluster as Node

    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)

    print(root)

    for level, portfolio in one_over_n(root).items():
        print(f"Level: {level}")
        print(portfolio.weights)
