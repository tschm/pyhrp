"""Marimo notebook demonstrating the Hierarchical Risk Parity (HRP) portfolio optimization method.

This notebook implements the Hierarchical Risk Parity algorithm as introduced by Marcos Lopez de Prado.
It compares different approaches to building the hierarchical tree, including the 'single' distance method
with and without bisection, and the 'ward' method. The notebook visualizes the resulting dendrograms
and portfolio weights for each approach.
"""

import marimo

__generated_with = "0.14.16"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import polars as pl

    from pyhrp.algos import risk_parity
    from pyhrp.cluster import Asset
    from pyhrp.hrp import build_tree


@app.cell
def _():
    mo.md(
        r"""
    # Hierarchical Risk Parity (HRP)

    We follow ideas by Marcos Lopez de Prado.

    - Compute the 1st dendrogram using the 'single' distance method.
    - Compute the 2nd dendrogram by using the order of leaves
      of the 1st dendrogram (following an argument by Thomas Raffinot)
    - Apply Risk Parity in a recursive bottom-up traverse
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
    return column_names, returns, assets_map


@app.cell
def _(column_names, returns):
    cor = returns.corr()
    cor.columns = [Asset(name=column) for column in column_names]
    cor.index = [Asset(name=column) for column in column_names]
    cov = returns.cov()
    cov.columns = [Asset(name=column) for column in column_names]
    cov.index = [Asset(name=column) for column in column_names]
    return cor, cov


@app.cell
def _(cor):
    # The first dendrogram is suffering. We observe the chaining effect
    # Convert column names to Asset objects for build_tree
    print(cor)

    dendrogram_before = build_tree(cor, method="single")
    dendrogram_before.plot()
    plt.show()
    return (dendrogram_before,)


@app.cell
def _(cov, dendrogram_before):
    # The weights are not well balanced
    # No surprise given exposure of nodes like 11, 12 or 15
    root_before = risk_parity(dendrogram_before.root, cov)
    root_before.portfolio.plot(names=dendrogram_before.names)
    plt.show()
    return


@app.cell
def _(cor):
    # The dendrogram suffers because of the 'chaining' effect. LdP is using
    # now only the order of the leaves (e.g. the assets) and
    # constructs a second Dendrogram.
    # Convert column names to Asset objects for build_tree
    dendrogram_bisection = build_tree(cor, method="single", bisection=True)
    dendrogram_bisection.plot()
    plt.show()
    return (dendrogram_bisection,)


@app.cell
def _(cov, dendrogram_bisection):
    root_bisection = risk_parity(dendrogram_bisection.root, cov)
    root_bisection.portfolio.plot(names=dendrogram_bisection.names)
    plt.show()
    return (root_bisection,)


@app.cell
def _(cor):
    dendrogram_ward = build_tree(cor, method="ward")
    dendrogram_ward.plot()
    plt.show()
    return (dendrogram_ward,)


@app.cell
def _(cov, dendrogram_ward):
    root_ward = risk_parity(dendrogram_ward.root, cov)
    root_ward.portfolio.plot(names=dendrogram_ward.names)
    plt.show()
    return (root_ward,)


@app.cell
def _(root_bisection, root_ward):
    # Assuming root_before.portfolio.weights1 and root_before.portfolio.weights2 are two weight series
    _weights_ward = root_ward.portfolio.weights
    _weights_bisection = root_bisection.portfolio.weights

    # Create a DataFrame from both weight series to align them on the same index (assuming they have the same index)
    weights_df = pd.DataFrame({"ward": _weights_ward, "single/bisection": _weights_bisection})

    # Plot both weight series
    weights_df.plot(kind="bar", width=0.8)

    # Ensure all possible x-axis labels are shown
    plt.xticks(ticks=range(len(weights_df)), labels=[asset.name for asset in weights_df.index], rotation=90)

    # Optionally, adjust the layout to avoid label clipping
    plt.tight_layout()

    # Show the plot
    plt.show()
    return


if __name__ == "__main__":
    app.run()
