"""Generate the README demo image comparing HRP methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from pyhrp.algos import risk_parity
from pyhrp.hrp import build_tree


def _load_prices(path: Path) -> pl.DataFrame:
    prices = pl.read_csv(path)
    if "date" in prices.columns:
        return prices.drop("date")
    return prices


def _compute_cov_and_corr(returns: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    matrix = returns.to_numpy().T
    cols = returns.columns
    cov = pl.DataFrame(dict(zip(cols, np.cov(matrix), strict=False)))
    cor = pl.DataFrame(dict(zip(cols, np.corrcoef(matrix), strict=False)))
    return cov, cor


def generate_demo_image(output: Path) -> Path:
    """Generate a bar chart comparing HRP weight allocations and save it to output."""
    repo_root = Path(__file__).resolve().parents[2]
    prices_path = repo_root / "tests" / "resources" / "stock_prices.csv"

    prices = _load_prices(prices_path)
    returns = prices.select(pl.all().pct_change()).drop_nulls()
    cov, cor = _compute_cov_and_corr(returns)

    dendrogram_ward = build_tree(cor, method="ward")
    root_ward = risk_parity(dendrogram_ward.root, cov)

    dendrogram_bisection = build_tree(cor, method="single", bisection=True)
    root_bisection = risk_parity(dendrogram_bisection.root, cov)

    weights_ward = root_ward.portfolio.weights
    weights_bisection = root_bisection.portfolio.weights
    assets = list(weights_ward.keys())

    fig = go.Figure(
        [
            go.Bar(name="ward", x=assets, y=[weights_ward[a] for a in assets]),
            go.Bar(name="single/bisection", x=assets, y=[weights_bisection[a] for a in assets]),
        ]
    )
    fig.update_layout(barmode="group", xaxis={"tickangle": -90})
    fig.write_image(str(output), width=1200, height=600, scale=2)
    return output


def main() -> None:
    """Parse CLI arguments and generate the demo image."""
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "demo.png",
        help="Output file path (default: %(default)s)",
    )
    args = parser.parse_args()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = generate_demo_image(output_path)
    print(f"Generated {generated}")


if __name__ == "__main__":
    main()
