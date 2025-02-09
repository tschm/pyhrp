import marimo

__generated_with = "0.11.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Demo pyhrp""")
    return


@app.cell
def _(__file__):
    from pathlib import Path

    import pandas as pd

    path = Path(__file__).parent
    prices = pd.read_csv(path / "data" / "stock_prices.csv", index_col=0)

    # compute returns
    returns = prices.pct_change().dropna(axis=0, how="all").fillna(0.0)
    return Path, path, pd, prices, returns


@app.cell
def _(returns):
    # compute the dendrogram
    from pyhrp.hrp import Dendrogram

    cor = returns.corr().values
    dendrogram = Dendrogram.build(cor, method="ward")
    dendrogram.plot(labels=returns.columns)
    return Dendrogram, cor, dendrogram


@app.cell
def _(dendrogram):
    dendrogram.root.pre_order()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
