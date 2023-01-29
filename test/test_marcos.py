from test.config import get_data, resource

import pandas as pd

from pyhrp.marcos import marcos


def test_marcos():
    prices = get_data()

    root = marcos(prices=prices)

    # uncomment this line if you want generating a new file
    # root.weights.to_csv(resource("weights_marcos.csv"), header=False)

    x = pd.read_csv(resource("weights_marcos.csv"), squeeze=True, index_col=0, header=0)
    x.index.name = None

    pd.testing.assert_series_equal(x, root.weights, check_exact=False)
