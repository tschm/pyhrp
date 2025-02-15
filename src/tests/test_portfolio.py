import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from pyhrp.cluster import Asset, Portfolio


def test_portfolio():
    p = Portfolio()
    a = Asset(name="A")
    b = Asset(name="B")
    c = Asset(name="C")

    p[a] = 0.4
    p[b] = 0.3
    p[c] = 0.3

    cov = pd.DataFrame(index=[a, b, c], columns=[a, b, c], data=np.diag([2, 3, 4]))

    ax = p.plot(names=["A", "B", "C"])
    ax.set_facecolor("lightyellow")  # Change the background color
    ax.grid(True)  # Enable gridlines

    plt.show()

    assert p.variance(cov) == pytest.approx(0.95)


def test_getset_item():
    p = Portfolio()
    a = Asset(name="A")
    p[a] = 0.4
    assert p[a] == 0.4


def test_asset_equality():
    a = Asset(name="A")
    b = Asset(name="A")
    assert a == b
    assert not a < 2
