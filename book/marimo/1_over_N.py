"""Marimo notebook demonstrating the hierarchical 1/N portfolio allocation strategy.

This notebook implements a hierarchical version of the 1/N portfolio allocation strategy,
inspired by Thomas Raffinot. It builds a dendrogram using the 'ward' distance method
and applies the 1/N strategy level by level, from the root to the leaves.
"""

# /// script
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

__generated_with = "0.16.5"
app = marimo.App()

with app.setup:
    import marimo as mo
    import polars as pl

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
    from pyhrp.hrp import compute_corr, compute_cov

    _prices = pl.read_csv(str(mo.notebook_location() / "public" / "stock_prices.csv"))
    returns = _prices.drop(_prices.columns[0]).select(pl.all().pct_change()).drop_nulls().fill_null(0.0)
    cor = compute_corr(returns)
    cov = compute_cov(returns)
    return (cor, cov, returns)


@app.cell
def _(cor):
    dendrogram = build_tree(cor, method="ward")
    dendrogram.plot().show()
    return (dendrogram,)


@app.cell
def _(dendrogram):
    from pyhrp.algos import one_over_n

    for level, portfolio in one_over_n(dendrogram):
        print(f"Level: {level}")
        portfolio.plot(names=dendrogram.names).show()
    return


if __name__ == "__main__":
    app.run()
