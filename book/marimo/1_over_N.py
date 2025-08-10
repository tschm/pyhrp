"""Marimo notebook demonstrating the hierarchical 1/N portfolio allocation strategy.

This notebook implements a hierarchical version of the 1/N portfolio allocation strategy,
inspired by Thomas Raffinot. It builds a dendrogram using the 'ward' distance method
and applies the 1/N strategy level by level, from the root to the leaves.
"""

import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

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
    # Read CSV with Polars
    prices_pl = pl.read_csv(str(mo.notebook_location() / "public" / "stock_prices.csv"))
    # Convert to pandas DataFrame with the first column as index
    index_col = prices_pl.columns[0]
    prices = prices_pl.to_pandas().set_index(index_col)
    returns = prices.pct_change().dropna(axis=0, how="all").fillna(0.0)
    # Store original column names for later use
    column_names = returns.columns.tolist()
    # Create a mapping from column names to Asset objects
    assets_map = {column: Asset(name=column) for column in column_names}
    return (returns, assets_map)


@app.cell
def _(returns, assets_map):
    cor = returns.corr()
    cov = returns.cov()
    return cor, cov, assets_map


@app.cell
def _(cor, assets_map):
    # We first build the tree. This task is very separated from the computations of weights on such a tree.
    # Convert column names to Asset objects for build_tree
    assets = [assets_map[col] for col in cor.columns]
    dendrogram = build_tree(cor, method="ward")
    # Replace the assets in the dendrogram with our Asset objects
    dendrogram.assets = assets
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
