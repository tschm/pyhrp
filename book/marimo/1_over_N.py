import marimo

__generated_with = "0.11.5"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # 1 over N (the hierarchical version)

        Inspired by Thomas Raffinot

        - Compute the 1st dendrogram using the 'single' distance method.
        - Compute the 2nd dendrogram by using the order of leaves
          of the 1st dendrogram (following an argument by Thomas Raffinot)
        - Apply Risk Parity in a recursive bottom-up traverse
        """
    )
    return


@app.cell
def _(__file__):
    from pathlib import Path

    data = Path(__file__).parent / "data"
    return Path, data


@app.cell
def _(data):
    import pandas as pd

    from pyhrp.cluster import Asset

    prices = pd.read_csv(data / "stock_prices.csv", index_col=0)
    returns = prices.pct_change().dropna(axis=0, how="all").fillna(0.0)
    returns.columns = [Asset(name=column) for column in returns.columns]
    return Asset, pd, prices, returns


@app.cell
def _(returns):
    cor = returns.corr()
    cov = returns.cov()
    return cor, cov


@app.cell
def _():
    import matplotlib.pyplot as plt

    from pyhrp.hrp import build_tree

    return build_tree, plt


@app.cell
def _(build_tree, cor, plt):
    # We first build the tree. This task is very separated from the computations of weights on such a tree.
    dendrogram = build_tree(cor, method="ward")
    dendrogram.plot()
    plt.show()
    return (dendrogram,)


@app.cell
def _(dendrogram, plt, portfolios):
    # We use the tree from the previous step and perform a 1/n
    # strategy in an iterative manner
    from pyhrp.algos import one_over_n

    # Drill deeper, level by level
    # Root is level 0 at Level 1 are two nodes...
    for level, portfolio in one_over_n(dendrogram):
        print(f"Level: {level}")
        portfolio.plot(names=dendrogram.names)
        plt.show()


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
