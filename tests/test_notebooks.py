"""Tests for the Marimo notebooks."""

import runpy
from pathlib import Path

import polars as pl

from pyhrp.cluster import Cluster


def test_notebooks() -> None:
    """Test notebooks execute and expose expected typed outputs."""
    repo_root = Path(__file__).resolve().parents[1]
    notebooks_dir = repo_root / "book" / "marimo"
    prices_path = repo_root / "tests" / "resources" / "stock_prices.csv"

    for py_file in notebooks_dir.glob("*.py"):
        namespace = runpy.run_path(str(py_file))

        if py_file.name == "demo.py":
            prices = namespace["_load_prices"](prices_path)
            returns = prices.select(pl.all().pct_change()).drop_nulls()
            cov, cor = namespace["_compute_cov_and_corr"](returns)
            root = namespace["risk_parity"](namespace["build_tree"](cor, method="ward").root, cov)

            assert isinstance(root, Cluster)
            assert isinstance(root.portfolio.weights, dict)
            assert root.portfolio.weights
            weights = list(root.portfolio.weights.values())
            assert abs(sum(weights) - 1.0) < 1e-9, "HRP weights must sum to 1.0"
            assert all(0.0 <= w <= 1.0 for w in weights), "All HRP weights must be in [0, 1]"
        else:
            assert "app" in namespace
