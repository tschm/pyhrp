# [pyhrp](https://tschm.github.io/pyhrp/book)

[![PyPI version](https://badge.fury.io/py/pyhrp.svg)](https://badge.fury.io/py/pyhrp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/pyhrp?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/pyhrp)
[![Coverage Status](https://coveralls.io/repos/github/tschm/pyhrp/badge.png?branch=main)](https://coveralls.io/github/tschm/pyhrp?branch=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/tschm/pyhrp/main.svg)](https://results.pre-commit.ci/latest/github/tschm/pyhrp/main)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/tschm/pyhrp)

A recursive implementation of the Hierarchical Risk Parity (hrp) approach
by Marcos Lopez de Prado.
We take heavily advantage of the scipy.cluster.hierarchy package.

![Comparing 'ward' with 'single' and bisection](https://raw.githubusercontent.com/tschm/pyhrp/main/demo.png)

Here's a simple example

```python
>>> import pandas as pd
>>> from pyhrp.hrp import build_tree
>>> from pyhrp.algos import risk_parity
>>> from pyhrp.cluster import Asset

>>> prices = pd.read_csv("src/tests/resources/stock_prices.csv", index_col=0, parse_dates=True)
>>> prices.columns = [Asset(name=column) for column in prices.columns]

>>> returns = prices.pct_change().dropna(axis=0, how="all")
>>> cov, cor = returns.cov(), returns.corr()

# Compute the dendrogram based on the correlation matrix and Ward's metric
>>> dendrogram = build_tree(cor, method='ward')
>>> dendrogram.plot()

# Compute the weights on the dendrogram
>>> root = risk_parity(root=dendrogram.root, cov=cov)
>>> ax = root.portfolio.plot(names=dendrogram.names)
```

For your convenience you can bypass the construction of the covariance and
correlation matrix, and the construction of the dendrogram.

```python
>>> from pyhrp.hrp import hrp

>>> root = hrp(prices=prices, method="ward", bisection=False)
```

You may expect a weight series here but instead the `hrp` function returns a
`Node` object. The `node` simplifies all further post-analysis.

```python
>>> weights = root.portfolio.weights
>>> variance = root.portfolio.variance(cov)

# You can drill deeper into the tree
>>> left = root.left
>>> right = root.right
```

## uv

Starting with

```bash
make install
```

will install [uv](https://github.com/astral-sh/uv) and create
the virtual environment defined in
pyproject.toml and locked in uv.lock.

## marimo

We install [marimo](https://marimo.io) on the fly within the aforementioned
virtual environment. Executing

```bash
make marimo
```

will install and start marimo.
