# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.13.15",
#     "pandas==2.3.0",
#     "matplotlib==3.10.1",
#     "pyhrp==1.3.7",
# ]
# ///
import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from pyhrp.cluster import Asset
    from pyhrp.hrp import build_tree


@app.cell
def _():
    mo.md(
        r"""
    # 1 over N (the hierarchical version)

    Inspired by Thomas Raffinot

    - Compute a dendrogram using the 'ward' distance method.
      Do not compute a 2nd Dendrogram.
    - Apply the methods level by level. Go from level 0 (only the root),
      to level 1 (left and right of the root), to level 2 (...)
    - On level n evaluate the function f for all leaves for each node.
    """
    )
    return


@app.cell
def _():
    prices = pd.read_csv(str(mo.notebook_location / "public" / "stock_prices.csv"), index_col=0)
    returns = prices.pct_change().dropna(axis=0, how="all").fillna(0.0)
    returns.columns = [Asset(name=column) for column in returns.columns]
    return (returns,)


@app.cell
def _(returns):
    cor = returns.corr()
    cov = returns.cov()
    return cor, cov


@app.cell
def _(cor):
    # We first build the tree. This task is very separated from the computations of weights on such a tree.
    dendrogram = build_tree(cor, method="ward")
    dendrogram.plot()
    plt.show()
    return (dendrogram,)


@app.cell
def _(dendrogram):
    # We use the tree from the previous step and perform a 1/n
    # strategy in an iterative manner
    from pyhrp.algos import one_over_n

    # Drill deeper, level by level
    # Root is level 0 at Level 1 are two nodes...
    for level, portfolio in one_over_n(dendrogram):
        print(f"Level: {level}")
        portfolio.plot(names=dendrogram.names)
        plt.show()
    return


if __name__ == "__main__":
    app.run()
