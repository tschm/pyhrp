import pytest

from pyhrp.algos import one_over_n
from pyhrp.cluster import Asset
from pyhrp.cluster import Cluster as Node
from pyhrp.hrp import Dendrogram


def test_one_over_n():
    """Test the one_over_n algorithm with a simple tree structure."""
    # Create a simple tree structure
    root = Node(10)
    root.left = Node(11)
    root.right = Node(0)

    root.left.left = Node(1)
    root.left.right = Node(2)

    # Create assets
    a = Asset(name="A")
    b = Asset(name="B")
    c = Asset(name="C")

    # Create dendrogram
    dendrogram = Dendrogram(root=root, assets=[a, b, c])

    # Collect portfolios from one_over_n algorithm
    portfolios = list(one_over_n(dendrogram))

    # Check that we get the expected number of levels
    assert len(portfolios) == len(root.levels)

    # Check the first level portfolio
    level0, portfolio0 = portfolios[0]
    assert level0 == 0

    # The first level should have weights that sum to 1
    assert sum(portfolio0.weights.values) == pytest.approx(1.0)

    # Check that all assets are in the portfolio
    assert set(portfolio0.assets) == {a, b, c}

    # Check that weights decrease with each level
    if len(portfolios) > 1:
        _, portfolio1 = portfolios[1]
        # The sum of weights should still be 1 at each level
        assert sum(portfolio1.weights.values) == pytest.approx(1.0)


def test_wrong_number_of_nodes():
    root = Node(10)
    root.left = Node(11)
    root.right = Node(0)
    root.left.left = Node(1)
    root.left.right = Node(2)

    a = Asset(name="A")
    b = Asset(name="B")
    c = Asset(name="C")
    d = Asset(name="D")

    with pytest.raises(ValueError):
        Dendrogram(root=root, assets=[a, b, c, d])
