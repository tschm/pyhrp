def test_one_over_n():
    from pyhrp.cluster import Cluster as Node

    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)

    print(root)
    print(root.levels)
    print(root.leaves)
    for node in root.leaves:
        print(node.height)

    # root.leaves
    # [Node(3), Node(4), Node(5)]

    # root.levels
    # [[Node(1)], [Node(2), Node(3)], [Node(4), Node(5)]]

    w = 1
    for n, level in enumerate(root.levels):
        for node in level:
            for leaf in node.leaves:
                root.portfolio[leaf.id] = w / node.leaf_count

        print(root.portfolio.weights)

        w *= 0.5
