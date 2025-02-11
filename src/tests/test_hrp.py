import numpy as np

from pyhrp.hrp import Dendrogram


def test_linkage(returns, resource_dir):
    dendrogram = Dendrogram.build(cor=returns.corr().values, method="single", bisection=False)
    ids = dendrogram.root.pre_order()
    assert ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]

    np.testing.assert_array_almost_equal(dendrogram.linkage, np.loadtxt(resource_dir / "links.csv", delimiter=","))


def test_bisection(returns, resource_dir):
    dendrogram = Dendrogram.build(cor=returns.corr().values, method="single", bisection=True)
    ids = dendrogram.root.pre_order()
    # The order doesn't change when using bisection
    assert ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]


def test_plot(returns):
    # compute covariance matrix and correlation matrices (both as DataFrames)
    cor = returns.corr()

    # you can either use a pre-computed node or you can construct a new dendrogram
    dendrogram = Dendrogram.build(cor.values, method="single", bisection=True)

    dendrogram.plot()

    import matplotlib.pyplot as plt

    plt.show()
