from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def prices(resource_dir):
    return pd.read_csv(resource_dir / "stock_prices.csv", parse_dates=True, index_col="date").truncate(
        before="2017-01-01"
    )


@pytest.fixture(scope="session")
def returns(prices):
    return prices.pct_change().dropna(axis=0, how="all").fillna(0.0)


@pytest.fixture(scope="session")
def distance(returns):
    cor = returns.corr().values
    distance = np.sqrt(np.clip((1.0 - cor) / 2.0, a_min=0.0, a_max=1.0))
    np.fill_diagonal(distance, val=0.0)
    return distance
