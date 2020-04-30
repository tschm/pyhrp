import pandas as pd

from test.config import get_data, resource
from pyhrp.obsolete import hrp


def test_hrp():
    prices = get_data()

    variance, weights = hrp(prices=prices)

    w = pd.Series(index=prices.keys(), data=weights, name="Weights").sort_index()

    x = pd.read_csv(resource("weights_hrp.csv"), squeeze=True, index_col=0, header=None)
    x.name = "Weights"
    x.index.name = None

    pd.testing.assert_series_equal(x, w, check_exact=False)