"""Additional tests to reach 100% coverage for hrp.py.

Covers Dendrogram.plot and validation branches in Dendrogram.__post_init__.
Also includes comprehensive tests for build_tree, hrp, and edge cases.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import polars as pl
import pytest

from pyhrp.cluster import Cluster
from pyhrp.hrp import Dendrogram, _compute_corr, _compute_distance_matrix, build_tree, hrp


def test_dendrogram_plot_executes(returns: pl.DataFrame) -> None:
    """Ensure Dendrogram.plot executes without error.

    We build a dendrogram from a correlation matrix and call plot.
    No figure is shown; we just ensure the function runs.
    """
    cor = _compute_corr(returns)
    dendrogram = build_tree(cor=cor, method="single", bisection=False)
    dendrogram.plot()


def test_dendrogram_plot_with_kwargs(returns: pl.DataFrame) -> None:
    """Test Dendrogram.plot with custom kwargs."""
    cor = _compute_corr(returns)
    dendrogram = build_tree(cor=cor, method="single", bisection=False)
    dendrogram.plot(color_threshold=0.5, above_threshold_color="red")


def test_post_init_raises_on_non_dataframe_distance() -> None:
    """__post_init__ should raise TypeError when distance is not a DataFrame."""
    root = Cluster(1)  # single leaf
    assets = ["A"]
    with pytest.raises(TypeError):
        Dendrogram(root=root, assets=assets, distance=[[0.0]], linkage=None, method="single")


def test_post_init_raises_on_distance_misalignment() -> None:
    """__post_init__ should raise ValueError when distance columns don't match assets order."""
    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    assets = ["A", "B"]
    # Distance with reversed column order -> misaligned
    dist = pl.DataFrame({"B": [0.0, 1.0], "A": [1.0, 0.0]})
    with pytest.raises(ValueError, match="must align with assets"):
        Dendrogram(root=root, assets=assets, distance=dist, linkage=None, method="single")


def test_post_init_raises_on_leaf_asset_count_mismatch() -> None:
    """__post_init__ should raise ValueError when leaf count != number of assets."""
    root = Cluster(99, left=Cluster(1), right=Cluster(2))
    assets = ["A"]
    with pytest.raises(ValueError, match="does not match number of assets"):
        Dendrogram(root=root, assets=assets, distance=None, linkage=None, method="single")


def test_post_init_with_valid_distance_matrix() -> None:
    """Test successful initialization with a valid distance matrix."""
    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    assets = ["A", "B"]
    dist = pl.DataFrame({"A": [0.0, 0.5], "B": [0.5, 0.0]})
    linkage = np.array([[0, 1, 0.5, 2]])

    dendrogram = Dendrogram(root=root, assets=assets, distance=dist, linkage=linkage, method="single")

    assert dendrogram.distance is not None
    assert dendrogram.method == "single"
    assert len(dendrogram.assets) == 2


def test_post_init_without_distance_matrix() -> None:
    """Test successful initialization without distance matrix."""
    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    assets = ["A", "B"]
    linkage = np.array([[0, 1, 0.5, 2]])

    dendrogram = Dendrogram(root=root, assets=assets, distance=None, linkage=linkage, method="single")

    assert dendrogram.distance is None


def test_dendrogram_ids_property() -> None:
    """Test the ids property returns correct leaf node values in order."""
    leaf0 = Cluster(0)
    leaf1 = Cluster(1)
    node2 = Cluster(2, left=leaf0, right=leaf1)
    leaf3 = Cluster(3)
    root = Cluster(4, left=node2, right=leaf3)

    assets = ["A", "B", "C"]
    dendrogram = Dendrogram(root=root, assets=assets, distance=None, linkage=None, method="single")

    assert dendrogram.ids == [0, 1, 3]


def test_dendrogram_names_property() -> None:
    """Test the names property returns asset names in dendrogram order."""
    leaf0 = Cluster(0)
    leaf1 = Cluster(1)
    leaf2 = Cluster(2)
    node3 = Cluster(3, left=leaf0, right=leaf1)
    root = Cluster(4, left=node3, right=leaf2)

    assets = ["X", "Y", "Z"]
    dendrogram = Dendrogram(root=root, assets=assets, distance=None, linkage=None, method="single")

    assert dendrogram.names == ["X", "Y", "Z"]


def test_build_tree_with_small_correlation_matrix() -> None:
    """Test build_tree with a small 2x2 correlation matrix."""
    cor = pl.DataFrame({"A": [1.0, 0.8], "B": [0.8, 1.0]})

    dendrogram = build_tree(cor=cor, method="single", bisection=False)

    assert dendrogram.root is not None
    assert len(dendrogram.assets) == 2
    assert dendrogram.assets == ["A", "B"]
    assert dendrogram.linkage is not None
    assert dendrogram.linkage.shape == (1, 4)


def test_build_tree_with_three_assets() -> None:
    """Test build_tree with a 3x3 correlation matrix."""
    cor = pl.DataFrame({"A": [1.0, 0.9, 0.3], "B": [0.9, 1.0, 0.4], "C": [0.3, 0.4, 1.0]})

    dendrogram = build_tree(cor=cor, method="ward", bisection=False)

    assert dendrogram.root is not None
    assert len(dendrogram.assets) == 3
    assert dendrogram.linkage.shape == (2, 4)


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_build_tree_with_different_methods(returns: pl.DataFrame, method: str) -> None:
    """Test build_tree with all supported linkage methods."""
    cor = _compute_corr(returns)
    dendrogram = build_tree(cor=cor, method=method, bisection=False)

    assert dendrogram.root is not None
    assert dendrogram.method == method
    assert len(dendrogram.assets) == len(cor.columns)
    assert dendrogram.linkage is not None


def test_build_tree_bisection_true(returns: pl.DataFrame) -> None:
    """Test build_tree with bisection enabled."""
    cor = _compute_corr(returns)
    dendrogram = build_tree(cor=cor, method="single", bisection=True)

    assert dendrogram.root is not None
    assert dendrogram.linkage is not None
    assert len(dendrogram.assets) == len(cor.columns)


def test_build_tree_raises_on_non_dataframe() -> None:
    """Test that build_tree raises TypeError when cor is not a DataFrame."""
    cor = [[1.0, 0.5], [0.5, 1.0]]

    with pytest.raises(TypeError):
        build_tree(cor=cor, method="single", bisection=False)


def test_build_tree_distance_matrix_properties() -> None:
    """Test that the generated distance matrix has correct properties."""
    cor = pl.DataFrame({"A": [1.0, 0.8], "B": [0.8, 1.0]})

    dendrogram = build_tree(cor=cor, method="single", bisection=False)
    dist_np = dendrogram.distance.to_numpy()

    # Symmetric
    np.testing.assert_array_almost_equal(dist_np, dist_np.T)
    # Diagonal should be zero
    assert (np.diag(dist_np) == 0).all()
    # All distances non-negative
    assert (dist_np >= 0).all()


def test_compute_distance_matrix_identity() -> None:
    """Test distance computation with identity correlation matrix."""
    cor = pl.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})

    dist = _compute_distance_matrix(cor)

    expected_dist = np.sqrt(0.5)
    assert dist["B"][0] == pytest.approx(expected_dist)
    assert dist["A"][1] == pytest.approx(expected_dist)
    assert dist["A"][0] == 0.0
    assert dist["B"][1] == 0.0


def test_compute_distance_matrix_perfect_correlation() -> None:
    """Test distance computation with perfect correlation."""
    cor = pl.DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0]})

    dist = _compute_distance_matrix(cor)

    assert dist["B"][0] == pytest.approx(0.0)
    assert dist["A"][1] == pytest.approx(0.0)


def test_compute_distance_matrix_negative_correlation() -> None:
    """Test distance computation with negative correlation."""
    cor = pl.DataFrame({"A": [1.0, -0.5], "B": [-0.5, 1.0]})

    dist = _compute_distance_matrix(cor)

    expected_dist = np.sqrt((1.0 - (-0.5)) / 2.0)
    assert dist["B"][0] == pytest.approx(expected_dist)
    assert dist["A"][1] == pytest.approx(expected_dist)


def test_compute_distance_matrix_symmetry() -> None:
    """Test that computed distance matrix is symmetric."""
    cor = pl.DataFrame({"A": [1.0, 0.7, 0.3], "B": [0.7, 1.0, 0.5], "C": [0.3, 0.5, 1.0]})

    dist = _compute_distance_matrix(cor)
    dist_np = dist.to_numpy()

    np.testing.assert_array_almost_equal(dist_np, dist_np.T)


def test_hrp_without_node(prices: pl.DataFrame) -> None:
    """Test hrp function without providing a pre-built node."""
    result = hrp(prices=prices, node=None, method="single", bisection=False)

    assert result is not None
    assert isinstance(result, Cluster)
    assert len(result.portfolio.assets) > 0


def test_hrp_with_node(prices: pl.DataFrame, returns: pl.DataFrame) -> None:
    """Test hrp function with a pre-built node."""
    cor = _compute_corr(returns)
    dendrogram = build_tree(cor=cor, method="ward", bisection=False)

    result = hrp(prices=prices, node=dendrogram.root, method="ward", bisection=False)

    assert result is not None
    assert isinstance(result, Cluster)


def test_hrp_with_bisection(prices: pl.DataFrame) -> None:
    """Test hrp function with bisection enabled."""
    result = hrp(prices=prices, node=None, method="single", bisection=True)

    assert result is not None
    assert isinstance(result, Cluster)


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_hrp_with_different_methods(prices: pl.DataFrame, method: str) -> None:
    """Test hrp with all supported linkage methods."""
    result = hrp(prices=prices, node=None, method=method, bisection=False)

    assert result is not None
    assert isinstance(result, Cluster)


def test_hrp_weights_sum_to_one(prices: pl.DataFrame) -> None:
    """Test that HRP weights sum to approximately 1."""
    result = hrp(prices=prices, node=None, method="ward", bisection=False)

    weights_sum = sum(result.portfolio.weights_dict.values())
    assert weights_sum == pytest.approx(1.0, rel=1e-6)


def test_hrp_with_small_dataset() -> None:
    """Test hrp with a minimal dataset (2 assets)."""
    prices = pl.DataFrame(
        {"A": [100, 101, 102, 101, 103, 104, 103, 105, 106, 107], "B": [50, 51, 50, 52, 51, 53, 54, 53, 55, 56]}
    )

    result = hrp(prices=prices, node=None, method="single", bisection=False)

    assert result is not None
    assert len(result.portfolio.assets) == 2
    weights_sum = sum(result.portfolio.weights_dict.values())
    assert weights_sum == pytest.approx(1.0, rel=1e-6)


def test_build_tree_linkage_shape_correct() -> None:
    """Test that linkage matrix has correct shape (n-1, 4) for n assets."""
    cor = pl.DataFrame({"A": [1.0, 0.8, 0.6], "B": [0.8, 1.0, 0.7], "C": [0.6, 0.7, 1.0]})

    dendrogram = build_tree(cor=cor, method="single", bisection=False)
    n_assets = len(cor.columns)

    assert dendrogram.linkage.shape == (n_assets - 1, 4)


def test_build_tree_root_has_all_leaves() -> None:
    """Test that the root cluster contains all assets as leaves."""
    cor = pl.DataFrame(
        {
            "A": [1.0, 0.5, 0.3, 0.2],
            "B": [0.5, 1.0, 0.4, 0.3],
            "C": [0.3, 0.4, 1.0, 0.6],
            "D": [0.2, 0.3, 0.6, 1.0],
        }
    )

    dendrogram = build_tree(cor=cor, method="ward", bisection=False)

    assert len(dendrogram.root.leaves) == 4


def test_dendrogram_frozen_dataclass() -> None:
    """Test that Dendrogram is frozen (immutable)."""
    root = Cluster(1)
    assets = ["A"]
    dendrogram = Dendrogram(root=root, assets=assets, distance=None, linkage=None, method="single")

    with pytest.raises(FrozenInstanceError):
        dendrogram.method = "complete"


def test_build_tree_preserves_asset_order() -> None:
    """Test that build_tree preserves the order of assets from correlation matrix."""
    cor = pl.DataFrame({"X": [1.0, 0.5, 0.3], "Y": [0.5, 1.0, 0.4], "Z": [0.3, 0.4, 1.0]})

    dendrogram = build_tree(cor=cor, method="single", bisection=False)

    assert dendrogram.assets == cor.columns


def test_hrp_handles_missing_data_in_prices() -> None:
    """Test that hrp correctly handles price data with missing values."""
    col_a = [float(100 + i + (i % 3)) for i in range(20)]
    col_a[5] = None  # type: ignore[call-overload]
    col_b = [float(50 + i * 0.5 - (i % 2)) for i in range(20)]
    col_b[10] = None  # type: ignore[call-overload]
    col_c = [float(75 + i * 0.3) for i in range(20)]

    prices = pl.DataFrame({"A": col_a, "B": col_b, "C": col_c})

    result = hrp(prices=prices, node=None, method="ward", bisection=False)

    assert result is not None
    weights_sum = sum(result.portfolio.weights_dict.values())
    assert weights_sum == pytest.approx(1.0, rel=1e-6)


def test_compute_distance_matrix_clipping() -> None:
    """Test that distance computation correctly clips negative values."""
    cor = pl.DataFrame({"A": [1.0, 0.999999], "B": [0.999999, 1.0]})

    dist = _compute_distance_matrix(cor)

    assert (dist.to_numpy() >= 0).all()


def test_dendrogram_distance_index_not_equals_columns() -> None:
    """Test validation when distance columns don't match assets."""
    root = Cluster(2, left=Cluster(0), right=Cluster(1))
    assets = ["A", "B"]
    dist = pl.DataFrame({"C": [0.0, 1.0], "D": [1.0, 0.0]})

    with pytest.raises(ValueError, match="must align with assets"):
        Dendrogram(root=root, assets=assets, distance=dist, linkage=None, method="single")
