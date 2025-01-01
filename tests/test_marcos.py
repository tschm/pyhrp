from __future__ import annotations

import pandas as pd

from pyhrp.marcos import marcos


def test_marcos(resource_dir):
    prices = pd.read_csv(resource_dir / "stock_prices.csv", parse_dates=True, index_col="date").truncate(
        before="2017-01-01"
    )

    root = marcos(prices=prices)

    # uncomment this line if you want generating a new file
    # root.weights.to_csv(resource("weights_marcos.csv"), header=False)

    x = pd.read_csv(resource_dir / "weights_marcos.csv", index_col=0, header=0).squeeze()

    x.index.name = None

    pd.testing.assert_series_equal(x, root.weights, check_exact=False)
