"""Plotly visualization for hierarchical clustering dendrograms.

Kept separate from the allocation core (``hrp.py``) so that the optional
plotting dependency (plotly) and the visualization logic do not couple the
algorithm modules. ``Dendrogram.plot`` delegates here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch

if TYPE_CHECKING:
    from .hrp import Dendrogram

__all__ = ["plot_dendrogram"]


def plot_dendrogram(dendrogram: Dendrogram, **kwargs: object) -> go.Figure:
    """Build and return a plotly dendrogram figure for a Dendrogram.

    Args:
        dendrogram (Dendrogram): Clustering result carrying the linkage matrix
            and asset labels to render.
        **kwargs (object): Extra keyword arguments forwarded to
            ``scipy.cluster.hierarchy.dendrogram`` (e.g. ``color_threshold``).

    Returns:
        go.Figure: A plotly figure drawing the dendrogram as line traces.

    Raises:
        ValueError: If the dendrogram has no linkage matrix to plot.

    Examples:
        >>> import polars as pl
        >>> from pyhrp.hrp import build_tree
        >>> from pyhrp.plot import plot_dendrogram
        >>> cor = pl.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]})
        >>> fig = plot_dendrogram(build_tree(cor, method="ward"))
        >>> type(fig).__name__
        'Figure'
    """
    if dendrogram.linkage is None:
        msg = "Dendrogram has no linkage matrix to plot."
        raise ValueError(msg)
    ddata = sch.dendrogram(dendrogram.linkage, labels=dendrogram.assets, no_plot=True, **kwargs)
    fig = go.Figure()
    for xs, ys in zip(ddata["icoord"], ddata["dcoord"], strict=False):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "steelblue"}, showlegend=False))
    n = len(dendrogram.assets)
    fig.update_layout(
        xaxis={
            "tickmode": "array",
            "tickvals": [5 + 10 * i for i in range(n)],
            "ticktext": ddata["ivl"],
            "tickangle": -90,
        },
    )
    return fig
