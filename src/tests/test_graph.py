from pyhrp.graph import dendrogram
from pyhrp.hrp import root


def test_graph(prices):
    returns = prices.pct_change().dropna(axis=0, how="all")

    # compute covariance matrix and correlation matrices (both as DataFrames)
    cor = returns.corr()

    # you can either use a pre-computed node or you can construct a new dendrogram
    links = root(cor.values, method="single", bisection=True).linkage

    dendrogram(links=links)
