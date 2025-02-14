from pathlib import Path

import pandas as pd
import pytest

from pyhrp.cluster import Asset


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def prices(resource_dir):
    _prices = pd.read_csv(resource_dir / "stock_prices.csv", parse_dates=True, index_col="date").truncate(
        before="2017-01-01"
    )

    _prices.columns = [Asset(name=column) for column in _prices.columns]
    return _prices


@pytest.fixture(scope="session")
def returns(prices):
    return prices.pct_change().dropna(axis=0, how="all").fillna(0.0)
