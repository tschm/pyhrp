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
    from pyhrp.graph import dendrogram
    from pyhrp.hrp import _dist, _linkage

    cor = returns.corr().values
    links = _linkage(_dist(cor), method="ward")
    dendrogram(links=links, labels=returns.columns)
    return cor, dendrogram, _dist, _linkage, links


@app.cell
def _(links):
    from pyhrp.hrp import _tree

    rootnode = _tree(links)
    # moving from the rootnode to the left we end up on a node
    print(rootnode.get_left())
    # moving from the rootnode to the right we end up on a node
    print(rootnode.get_right())
    print(rootnode.get_count())
    print(rootnode.get_left().get_count())
    print(rootnode.get_right().get_count())
    return rootnode, _tree


@app.cell
def _(rootnode):
    rootnode.pre_order()
    return


app._unparsable_cell(
    r"""
    rootnode.get_id()
    rootnode.get_left().get_id()
    rootnode.get_right().get_id()
    rootnode.
    """,
    name="_",
)


@app.cell
def _(links):
    links
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
