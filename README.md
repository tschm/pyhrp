# [pyhrp](https://tschm.github.io/pyhrp)

[![PyPI version](https://badge.fury.io/py/pyhrp.svg)](https://badge.fury.io/py/pyhrp)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://pypi.org/project/pyhrp/)
[![Downloads](https://static.pepy.tech/personalized-badge/pyhrp?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/pyhrp)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/tschm/pyhrp/blob/main/LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/tschm/pyhrp/badge)](https://www.codefactor.io/repository/github/tschm/pyhrp)
[![Rhiza](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftschm%2Fpyhrp%2Fmain%2F.rhiza%2Ftemplate.yml&query=%24%5B'template-branch'%5D&label=rhiza)](https://github.com/jebel-quant/rhiza)

[![Coverage](https://tschm.github.io/pyhrp/coverage-badge.svg)](https://tschm.github.io/pyhrp/reports/html-coverage/)

A recursive implementation of the Hierarchical Risk Parity (hrp) approach
by Marcos Lopez de Prado.
We take advantage of the scipy.cluster.hierarchy package.

![Comparing 'ward' with 'single' and bisection](https://raw.githubusercontent.com/tschm/pyhrp/main/demo.png)

Here's a simple example

```python
import polars as pl
from pyhrp.hrp import build_tree, _compute_cov, _compute_corr
from pyhrp.algos import risk_parity

prices = pl.read_csv("tests/resources/stock_prices.csv", try_parse_dates=True).drop("date")

returns = prices.select(pl.all().pct_change()).drop_nulls().fill_null(0.0)
cov = _compute_cov(returns)
cor = _compute_corr(returns)

# Compute the dendrogram based on the correlation matrix and Ward's metric
dendrogram = build_tree(cor, method='ward')
dendrogram.plot()

# Compute the weights on the dendrogram
root = risk_parity(root=dendrogram.root, cov=cov)
root.portfolio.plot(names=dendrogram.names)

```

For your convenience you can bypass the construction of the covariance and
correlation matrix, and the construction of the dendrogram.

```python
from pyhrp.hrp import hrp
root = hrp(prices=prices, method="ward", bisection=False)

```

You may expect a weight series here but instead the `hrp` function returns a
`Node` object. The `node` simplifies all further post-analysis.

```python
weights = root.portfolio.weights
variance = root.portfolio.variance(cov)

# You can drill deeper into the tree
left = root.left
right = root.right

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
