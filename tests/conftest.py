"""Pytest configuration and fixtures for the tests."""

import json
from pathlib import Path

import pandas as pd
import pytest
from pandas import DataFrame


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture() -> Path:
    """Fixture that provides the path to the test resources directory.

    This fixture is session-scoped, meaning it's created once per test session.

    Returns:
        Path: The path to the test resources directory.
    """
    return Path(__file__).parent / "resources"


@pytest.fixture(name="root_dir")
def root_fixture(resource_dir: Path) -> Path:
    """Fixture that provides the path to the project root directory.

    Args:
        resource_dir: Path to the test resources directory.

    Returns:
        Path: The path to the project root directory.
    """
    return resource_dir.parent.parent.parent


@pytest.fixture(scope="session")
def prices(resource_dir: Path) -> DataFrame:
    """Fixture that provides a DataFrame of stock prices.

    This fixture is session-scoped, meaning it's created once per test session.
    It loads stock price data from a CSV file and converts column names to Asset objects.

    Args:
        resource_dir: Path to the test resources directory.

    Returns:
        DataFrame: A DataFrame containing stock prices with Asset objects as column names.
    """
    # Load price data from CSV file
    _prices = pd.read_csv(resource_dir / "stock_prices.csv", parse_dates=True, index_col="date").truncate(
        before="2017-01-01"
    )

    return _prices


@pytest.fixture(scope="session")
def returns(prices: DataFrame) -> DataFrame:
    """Fixture that provides a DataFrame of stock returns.

    This fixture is session-scoped, meaning it's created once per test session.
    It calculates returns from the prices DataFrame and handles missing values.

    Args:
        prices: DataFrame of stock prices.

    Returns:
        DataFrame: A DataFrame containing stock returns.
    """
    # Calculate returns from prices and handle missing values
    return prices.pct_change().dropna(axis=0, how="all").fillna(0.0)


@pytest.fixture(scope="session")
def market_data(resource_dir: Path) -> dict:
    """Fixture that provides market data in JSON format.

    This fixture is session-scoped, meaning it's created once per test session.
    It loads market data from a JSON file in the test resources directory.

    Args:
        resource_dir: Path to the test resources directory.

    Returns:
        dict: A dictionary containing market data loaded from JSON.
    """
    with open(resource_dir / "market_data.json") as f:
        return json.load(f)
