"""Additional tests to reach 100% coverage for hrp.py.

Covers Dendrogram.plot and validation branches in Dendrogram.__post_init__.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pyhrp.cluster import Cluster
from pyhrp.hrp import Dendrogram, build_tree


def test_dendrogram_plot_executes(returns: pd.DataFrame) -> None:
    """Ensure Dendrogram.plot executes without error.

    We build a dendrogram from a correlation matrix and call plot.
    No figure is shown; we just ensure the function runs.
    """
    cor = returns.corr()
    dendrogram = build_tree(cor=cor, method="single", bisection=False)
    # Should not raise
    dendrogram.plot()


def test_post_init_raises_on_non_dataframe_distance() -> None:
    """__post_init__ should raise TypeError when distance is not a DataFrame."""
    root = Cluster(1)  # single leaf
    assets = ["A"]
    with pytest.raises(TypeError):
        Dendrogram(root=root, assets=assets, distance=[[0.0]], linkage=None, method="single")


def test_post_init_raises_on_distance_misalignment() -> None:
    """__post_init__ should raise ValueError when distance index/columns don't match assets order."""
    # Tree with two leaves (IDs don't matter for this validation)
    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    assets = ["A", "B"]
    # Distance with reversed order -> misaligned
    dist = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=["B", "A"], columns=["B", "A"])
    with pytest.raises(ValueError):
        Dendrogram(root=root, assets=assets, distance=dist, linkage=None, method="single")


def test_post_init_raises_on_leaf_asset_count_mismatch() -> None:
    """__post_init__ should raise ValueError when leaf count != number of assets."""
    # Two-leaf tree but only one asset
    root = Cluster(99, left=Cluster(1), right=Cluster(2))
    assets = ["A"]
    with pytest.raises(ValueError):
        Dendrogram(root=root, assets=assets, distance=None, linkage=None, method="single")
