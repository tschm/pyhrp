import matplotlib.pyplot as plt

from pyhrp.algos import one_over_n
from pyhrp.hrp import build_tree


def test_one_over_n(returns):
    cor = returns.corr()
    dendrogram = build_tree(cor=cor, method="ward")
    dendrogram.plot()
    plt.show()

    for level, portfolio in one_over_n(dendrogram):
        print(f"Level: {level}")
        print(portfolio)
