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

__generated_with = "0.11.4"
app = marimo.App()

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

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
def _(__file__):
    from pathlib import Path

    data = Path(__file__).parent / "data"
    return Path, data


@app.cell
def _(data):
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
def _(cor):
    # The first dendrogram is suffering. We observe the chaining effect
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
    return (root_before,)


@app.cell
def _(cor):
    # The dendrogram suffers because of the 'chaining' effect. LdP is using
    # now only the order of the leaves (e.g. the assets) and
    # constructs a second Dendrogram.
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
    return (weights_df,)


if __name__ == "__main__":
    app.run()
