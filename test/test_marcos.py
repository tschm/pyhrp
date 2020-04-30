import pandas as pd

from pyhrp.marcos import marcos
from test.config import resource, get_data


def test_marcos():
    prices = get_data()

    root = marcos(prices=prices)

    # uncomment this line if you want generating a new file
    # root.weights_series(index=list(prices.keys())).to_csv(resource("weights_marcos.csv"), header=False)

    x = pd.read_csv(resource("weights_marcos.csv"), squeeze=True, index_col=0, header=None)
    x.name = "Weights"
    x.index.name = None

    pd.testing.assert_series_equal(x, root.weights, check_exact=False)