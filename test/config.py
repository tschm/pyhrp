import os
import pandas as pd


def resource(name):
    return os.path.join(os.path.dirname(__file__), "resources", name)


def read_pd(name, **kwargs):
    return pd.read_csv(resource(name), **kwargs)


def get_data():
    # https://github.com/robertmartin8/PyPortfolioOpt
    return read_pd("stock_prices.csv", parse_dates=True, index_col="date")


def get_benchmark_data():
    # https://github.com/robertmartin8/PyPortfolioOpt
    return read_pd("spy_prices.csv", parse_dates=True, index_col="date")
