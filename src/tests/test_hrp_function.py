from __future__ import annotations

import pandas as pd

from pyhrp.hrp import hrp


def test_hrp(prices, resource_dir):
    cluster = hrp(prices=prices, method="ward", bisection=False)

    # uncomment this line if you want generating a new file
    # root.weights.to_csv(resource("weights_hrp.csv"), header=False)

    x = pd.read_csv(resource_dir / "weights_hrp.csv", index_col=0, header=0).squeeze()

    x.index.name = None

    pd.testing.assert_series_equal(x, cluster.portfolio.weights, check_exact=False)


def test_marcos(resource_dir, prices):
    cluster = hrp(prices=prices, method="ward", bisection=True)

    # uncomment this line if you want generating a new file
    # root.weights.to_csv(resource("weights_marcos.csv"), header=False)

    x = pd.read_csv(resource_dir / "weights_marcos.csv", index_col=0, header=0).squeeze()

    x.index.name = None

    pd.testing.assert_series_equal(x, cluster.portfolio.weights, check_exact=False)
