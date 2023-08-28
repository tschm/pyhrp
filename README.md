# [pyhrp](https://tschm.github.io/pyhrp/book)

A recursive implementation of the Hierarchical Risk Parity (hrp) approach
by Marcos Lopez de Prado.
We take heavily advantage of the scipy.cluster.hierarchy package.

Here's a simple example

```python
import pandas as pd
from pyhrp.hrp import dist, linkage, tree, _hrp

prices = pd.read_csv("test/resources/stock_prices.csv", index_col=0, parse_dates=True)

returns = prices.pct_change().dropna(axis=0, how="all")
cov, cor = returns.cov(), returns.corr()
links = linkage(dist(cor.values), method='ward')
node = tree(links)

rootcluster = _hrp(node, cov)

ax = dendrogram(links, orientation="left")
ax.get_figure().savefig("dendrogram.png")
```

For your convenience you can bypass the construction of the covariance and
correlation matrix, the links and the node, e.g. the root of the tree (dendrogram).

```python
import pandas as pd
from pyhrp.hrp import hrp

prices = pd.read_csv("test/resources/stock_prices.csv", index_col=0, parse_dates=True)
root = hrp(prices=prices)
```

You may expect a weight series here but instead the `hrp` function returns a
`Cluster` object. The `Cluster` simplifies all further post-analysis.

```python
print(cluster.weights)
print(cluster.variance)
# You can drill into the graph by going downstream
print(cluster.left)
print(cluster.right)

## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
make install
```

to replicate the virtual environment we have defined in [pyproject.toml](pyproject.toml)
and locked in [poetry.lock](poetry.lock).

## Jupyter

We install [JupyterLab](https://jupyter.org) on fly within the aforementioned
virtual environment. Executing

```bash
make jupyter
```

will install and start the jupyter lab.
