"""Tests for the plotly dendrogram visualization."""

from __future__ import annotations

import polars as pl
import pytest

from pyhrp.cluster import Cluster
from pyhrp.covariance import compute_corr
from pyhrp.dendrogram import Dendrogram, build_tree
from pyhrp.plot import plot_dendrogram


def test_plot_dendrogram_returns_figure() -> None:
    """plot_dendrogram builds a plotly figure from a small dendrogram."""
    cor = pl.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]})
    fig = plot_dendrogram(build_tree(cor, method="ward"))
    assert type(fig).__name__ == "Figure"


def test_dendrogram_plot_executes(returns: pl.DataFrame) -> None:
    """Ensure Dendrogram.plot executes without error.

    We build a dendrogram from a correlation matrix and call plot.
    No figure is shown; we just ensure the function runs.
    """
    cor = compute_corr(returns)
    dendrogram = build_tree(cor=cor, method="single", bisection=False)
    dendrogram.plot()


def test_dendrogram_plot_with_kwargs(returns: pl.DataFrame) -> None:
    """Test Dendrogram.plot with custom kwargs."""
    cor = compute_corr(returns)
    dendrogram = build_tree(cor=cor, method="single", bisection=False)
    dendrogram.plot(color_threshold=0.5, above_threshold_color="red")


def test_dendrogram_plot_without_linkage_raises() -> None:
    """plot() needs a linkage matrix and reports a clear error when it is missing."""
    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    dendrogram = Dendrogram(root=root, assets=["A", "B"], linkage=None)
    with pytest.raises(ValueError, match="no linkage matrix to plot"):
        dendrogram.plot()
