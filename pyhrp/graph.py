import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import numpy.random as nr


def dendrogram(links, ax=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(figsize=(25, 20))
    sch.dendrogram(links, ax=ax, **kwargs)
    return ax


def bisection(ids):
    def split(ids):
        # split the vector ids in two parts, split in the middle
        assert len(ids) >= 2
        n = len(ids)
        return ids[:n // 2], ids[n // 2:]

    assert len(ids) >= 1

    if len(ids) == 1:
        return sch.ClusterNode(id=ids[0])

    left, right = split(ids)
    return sch.ClusterNode(id=nr.randint(low=100000, high=200000), left=bisection(ids=left), right=bisection(ids=right))


if __name__ == '__main__':
    ids = [4, 3, 0, 1, 2]
    root = bisection([4,3,2,5,0,6,1])
    print(root.pre_order())
    print(root.id)
    print(root.left.id)
    print(root.left.is_leaf())
    print(root.right.left.id)
    print(root.right.right.id)

    #print(root.left.right.id)
    #print(root.left.left.id)
    #print(root.right.id)