import numpy as np
from pyhrp.hierarchical_portfolio import HRPOpt
from test.config import get_data


def test_hrp_portfolio():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    hrp = HRPOpt(returns)
    w = hrp.optimize()
    assert isinstance(w, dict)
    assert set(w.keys()) == set(df.columns)
    np.testing.assert_almost_equal(sum(w.values()), 1)
    assert all([i >= 0 for i in w.values()])


def test_cluster_var():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    cov = returns.cov()
    tickers = ["SHLD", "AMD", "BBY", "RRC", "FB", "WMT", "T", "BABA", "PFE", "UAA"]
    var = HRPOpt._get_cluster_var(cov, tickers)
    np.testing.assert_almost_equal(var, 0.00012842967106653283)


def test_quasi_diag():
    df = get_data()
    returns = df.pct_change().dropna(how="all")
    hrp = HRPOpt(returns)
    hrp.optimize()
    clusters = hrp.clusters
    assert HRPOpt._get_quasi_diag(clusters)[:5] == [12, 6, 15, 14, 2]
