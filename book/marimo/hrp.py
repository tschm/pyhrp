"""Marimo notebook demonstrating the Hierarchical Risk Parity (HRP) portfolio optimization method.

This notebook implements the Hierarchical Risk Parity algorithm as introduced by Marcos Lopez de Prado.
It compares different approaches to building the hierarchical tree, including the 'single' distance method
with and without bisection, and the 'ward' method. The notebook visualizes the resulting dendrograms
and portfolio weights for each approach.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.18.4",
#     "plotly",
#     "polars",
#     "pyhrp",
# ]
#
# [tool.uv.sources]
# pyhrp = { path = "../..", editable=true }
#
# ///

import marimo

__generated_with = "0.14.16"
app = marimo.App()

with app.setup:
    import marimo as mo
    import polars as pl

    from pyhrp.algos import risk_parity
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
    _prices = pl.read_csv(str(mo.notebook_location() / "public" / "stock_prices.csv"))
    returns = _prices.drop(_prices.columns[0]).select(pl.all().pct_change()).drop_nulls().fill_null(0.0)
    column_names = returns.columns
    return column_names, returns


@app.cell
def _(returns):
    from pyhrp.hrp import _compute_corr, _compute_cov

    cor = _compute_corr(returns)
    cov = _compute_cov(returns)
    return cor, cov


@app.cell
def _(cor):
    dendrogram_before = build_tree(cor, method="single")
    dendrogram_before.plot().show()
    return (dendrogram_before,)


@app.cell
def _(cov, dendrogram_before):
    root_before = risk_parity(dendrogram_before.root, cov)
    root_before.portfolio.plot(names=dendrogram_before.names).show()
    return


@app.cell
def _(cor):
    dendrogram_bisection = build_tree(cor, method="single", bisection=True)
    dendrogram_bisection.plot().show()
    return (dendrogram_bisection,)


@app.cell
def _(cov, dendrogram_bisection):
    root_bisection = risk_parity(dendrogram_bisection.root, cov)
    root_bisection.portfolio.plot(names=dendrogram_bisection.names).show()
    return (root_bisection,)


@app.cell
def _(cor):
    dendrogram_ward = build_tree(cor, method="ward")
    dendrogram_ward.plot().show()
    return (dendrogram_ward,)


@app.cell
def _(cov, dendrogram_ward):
    root_ward = risk_parity(dendrogram_ward.root, cov)
    root_ward.portfolio.plot(names=dendrogram_ward.names).show()
    return (root_ward,)


@app.cell
def _(root_bisection, root_ward):
    import plotly.graph_objects as go

    _weights_ward = root_ward.portfolio.weights
    _weights_bisection = root_bisection.portfolio.weights
    _assets = list(_weights_ward.keys())
    _fig = go.Figure(
        [
            go.Bar(name="ward", x=_assets, y=[_weights_ward[a] for a in _assets]),
            go.Bar(name="single/bisection", x=_assets, y=[_weights_bisection[a] for a in _assets]),
        ]
    )
    _fig.update_layout(barmode="group", xaxis={"tickangle": -90})
    _fig.show()
    return


if __name__ == "__main__":
    app.run()
