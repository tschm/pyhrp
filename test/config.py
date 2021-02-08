import os
import pandas as pd


def resource(name):
    return os.path.join(os.path.dirname(__file__), "resources", name)


def get_data():
    # https://github.com/robertmartin8/PyPortfolioOpt
    return pd.read_csv(resource("stock_prices.csv"), parse_dates=True, index_col="date").truncate(before="2017-01-01")


def get_benchmark_data():
    # https://github.com/robertmartin8/PyPortfolioOpt
    return pd.read_csv(resource("spy_prices.csv"), parse_dates=True, index_col="date")
