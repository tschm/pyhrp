import pytest
from matplotlib import pyplot as plt

from pyhrp.cluster import Portfolio


def test_portfolio():
    p = Portfolio()
    p.variance = 5
    p["A"] = 0.4
    p["B"] = 0.3
    p["C"] = 0.3

    assert str(p) == "Portfolio(_variance=5, _weights={'A': 0.4, 'B': 0.3, 'C': 0.3})"
    ax = p.plot()
    ax.set_facecolor("lightyellow")  # Change the background color
    ax.grid(True)  # Enable gridlines

    plt.show()


def test_negative_variance():
    with pytest.raises(ValueError):
        p = Portfolio()
        p.variance = -1


def test_portfolio_variance():
    p = Portfolio()
    p.variance = 5
    assert p.variance == 5


def test_getset_item():
    p = Portfolio()
    p["A"] = 0.4
    assert p["A"] == 0.4


def test_copy():
    p = Portfolio()
    p["A"] = 0.4
    p.variance = 5

    p2 = p.copy()
    p2["A"] = 0.5
    p2.variance = 10

    # no change here...
    assert p.variance == 5
    assert p["A"] == 0.4
