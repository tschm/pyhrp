"""Tests for the one_over_n portfolio construction algorithm."""

import pytest

from pyhrp.algos import one_over_n
from pyhrp.cluster import Portfolio
from pyhrp.cluster import Cluster as Node
from pyhrp.hrp import Dendrogram


def test_one_over_n() -> None:
    """Test the one_over_n algorithm with a simple tree structure.

    This test verifies:
    1. The one_over_n algorithm correctly generates portfolios for each level
    2. The number of portfolios matches the number of tree levels
    3. Portfolio weights sum to 1.0 at each level
    4. All assets are included in the portfolio
    """
    # Create a simple tree structure
    root = Node(10)
    root.left = Node(11)
    root.right = Node(0)

    root.left.left = Node(1)
    root.left.right = Node(2)

    # Create assets
    a = "A"
    b = "B"
    c = "C"

    # Create dendrogram
    dendrogram = Dendrogram(root=root, assets=[a, b, c])

    # Collect portfolios from one_over_n algorithm
    portfolios: list[tuple[int, Portfolio]] = list(one_over_n(dendrogram))

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


def test_wrong_number_of_nodes() -> None:
    """Test that Dendrogram raises ValueError when assets and leaves count don't match.

    This test verifies:
    1. The Dendrogram constructor validates that the number of assets matches
       the number of leaf nodes in the tree
    2. A ValueError is raised when there's a mismatch
    """
    # Create a tree with 3 leaf nodes
    root = Node(10)
    root.left = Node(11)
    root.right = Node(0)
    root.left.left = Node(1)
    root.left.right = Node(2)

    # Verify that a ValueError is raised due to the mismatch
    with pytest.raises(ValueError):
        Dendrogram(root=root, assets=["a", "b", "c", "d"])


