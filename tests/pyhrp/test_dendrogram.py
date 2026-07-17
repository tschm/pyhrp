"""Tests for hierarchical tree construction and the Dendrogram container.

Covers build_tree, the private distance/linkage/bisection helpers, and the
Dendrogram dataclass (validation, ordering properties, immutability).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from polars import DataFrame

from pyhrp.algos import risk_parity
from pyhrp.cluster import Cluster
from pyhrp.covariance import compute_corr
from pyhrp.dendrogram import (
    Dendrogram,
    _bisect_tree,
    _compute_distance_matrix,
    _get_linkage,
    build_tree,
)


@st.composite
def covariance_matrices(draw: st.DrawFn) -> pl.DataFrame:
    """Generate symmetric positive-definite covariance matrices."""
    n_assets = draw(st.integers(min_value=2, max_value=8))
    base = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_assets, n_assets),
            elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        )
    )
    cov = base @ base.T + np.eye(n_assets) * 1e-6
    assets = [f"A{i}" for i in range(n_assets)]
    return pl.DataFrame(dict(zip(assets, cov, strict=True)))


@st.composite
def correlation_matrices(draw: st.DrawFn) -> pl.DataFrame:
    """Generate valid correlation matrices from covariance matrices."""
    cov = draw(covariance_matrices())
    cov_np = cov.to_numpy()
    std = np.sqrt(np.diag(cov_np))
    corr = cov_np / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    assets = cov.columns
    return pl.DataFrame(dict(zip(assets, corr, strict=True)))


class TestDendrogram:
    """Tests for the Dendrogram result container."""

    def test_post_init_raises_on_non_dataframe_distance(self) -> None:
        """__post_init__ should raise TypeError when distance is not a DataFrame."""
        root = Cluster(1)  # single leaf
        assets = ["A"]
        with pytest.raises(TypeError):
            Dendrogram(root=root, assets=assets, distance=[[0.0]], linkage=None, method="single")

    def test_post_init_raises_on_distance_misalignment(self) -> None:
        """__post_init__ should raise ValueError when distance columns don't match assets order."""
        root = Cluster(2, left=Cluster(0), right=Cluster(1))
        assets = ["A", "B"]
        # Distance with reversed column order -> misaligned
        dist = pl.DataFrame({"B": [0.0, 1.0], "A": [1.0, 0.0]})
        with pytest.raises(ValueError, match="must align with assets"):
            Dendrogram(root=root, assets=assets, distance=dist, linkage=None, method="single")

    def test_post_init_raises_on_leaf_asset_count_mismatch(self) -> None:
        """__post_init__ should raise ValueError when leaf count != number of assets."""
        root = Cluster(99, left=Cluster(1), right=Cluster(2))
        assets = ["A"]
        with pytest.raises(ValueError, match="does not match number of assets"):
            Dendrogram(root=root, assets=assets, distance=None, linkage=None, method="single")

    def test_post_init_with_valid_distance_matrix(self) -> None:
        """Test successful initialization with a valid distance matrix."""
        root = Cluster(2, left=Cluster(0), right=Cluster(1))
        assets = ["A", "B"]
        dist = pl.DataFrame({"A": [0.0, 0.5], "B": [0.5, 0.0]})
        linkage = np.array([[0, 1, 0.5, 2]])

        dendrogram = Dendrogram(root=root, assets=assets, distance=dist, linkage=linkage, method="single")

        assert dendrogram.distance is not None
        assert dendrogram.method == "single"
        assert len(dendrogram.assets) == 2

    def test_post_init_without_distance_matrix(self) -> None:
        """Test successful initialization without distance matrix."""
        root = Cluster(2, left=Cluster(0), right=Cluster(1))
        assets = ["A", "B"]
        linkage = np.array([[0, 1, 0.5, 2]])

        dendrogram = Dendrogram(root=root, assets=assets, distance=None, linkage=linkage, method="single")

        assert dendrogram.distance is None

    def test_ids_property(self) -> None:
        """Test the ids property returns correct leaf node values in order."""
        leaf0 = Cluster(0)
        leaf1 = Cluster(1)
        node2 = Cluster(2, left=leaf0, right=leaf1)
        leaf3 = Cluster(3)
        root = Cluster(4, left=node2, right=leaf3)

        assets = ["A", "B", "C"]
        dendrogram = Dendrogram(root=root, assets=assets, distance=None, linkage=None, method="single")

        assert dendrogram.ids == [0, 1, 3]

    def test_names_property(self) -> None:
        """Test the names property returns asset names in dendrogram order."""
        leaf0 = Cluster(0)
        leaf1 = Cluster(1)
        leaf2 = Cluster(2)
        node3 = Cluster(3, left=leaf0, right=leaf1)
        root = Cluster(4, left=node3, right=leaf2)

        assets = ["X", "Y", "Z"]
        dendrogram = Dendrogram(root=root, assets=assets, distance=None, linkage=None, method="single")

        assert dendrogram.names == ["X", "Y", "Z"]

    def test_frozen_dataclass(self) -> None:
        """Test that Dendrogram is frozen (immutable)."""
        root = Cluster(1)
        assets = ["A"]
        dendrogram = Dendrogram(root=root, assets=assets, distance=None, linkage=None, method="single")

        with pytest.raises(FrozenInstanceError):
            dendrogram.method = "complete"

    def test_distance_index_not_equals_columns(self) -> None:
        """Test validation when distance columns don't match assets."""
        root = Cluster(2, left=Cluster(0), right=Cluster(1))
        assets = ["A", "B"]
        dist = pl.DataFrame({"C": [0.0, 1.0], "D": [1.0, 0.0]})

        with pytest.raises(ValueError, match="must align with assets"):
            Dendrogram(root=root, assets=assets, distance=dist, linkage=None, method="single")

    def test_wrong_number_of_nodes(self) -> None:
        """Test that Dendrogram raises ValueError when assets and leaves count don't match.

        This test verifies:
        1. The Dendrogram constructor validates that the number of assets matches
           the number of leaf nodes in the tree
        2. A ValueError is raised when there's a mismatch
        """
        # Create a tree with 3 leaf nodes
        root = Cluster(10)
        root.left = Cluster(11)
        root.right = Cluster(0)
        root.left.left = Cluster(1)
        root.left.right = Cluster(2)

        # Verify that a ValueError is raised due to the mismatch
        with pytest.raises(ValueError, match="does not match number of assets"):
            Dendrogram(root=root, assets=["a", "b", "c", "d"])


def test_linkage(returns: DataFrame, resource_dir: Path) -> None:
    """Test the linkage matrix generation in the build_tree function.

    This test verifies:
    1. The correct order of node IDs in the dendrogram
    2. The linkage matrix matches the expected values from a reference file

    Args:
        returns: DataFrame of asset returns
        resource_dir: Path to test resources directory
    """
    dendrogram = build_tree(cor=compute_corr(returns), method="single", bisection=False)

    assert dendrogram.ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]

    np.testing.assert_array_almost_equal(dendrogram.linkage, np.loadtxt(resource_dir / "links.csv", delimiter=","))


def test_bisection(returns: DataFrame, resource_dir: Path) -> None:
    """Test the bisection method in the build_tree function.

    This test verifies that the order of node IDs remains consistent
    when using the bisection method.

    Args:
        returns: DataFrame of asset returns
        resource_dir: Path to test resources directory
    """
    dendrogram = build_tree(cor=compute_corr(returns), method="single", bisection=True)

    assert dendrogram.ids == [11, 7, 19, 6, 14, 5, 10, 13, 3, 1, 4, 16, 0, 2, 17, 9, 8, 18, 12, 15]


def test_plot_bisection(returns: DataFrame) -> None:
    """Test building a dendrogram with bisection and verify the node order.

    This test verifies:
    1. The dendrogram is properly constructed with bisection
    2. The dendrogram has the expected structure (root and linkage)
    3. The order of asset names in the dendrogram matches the expected order

    Args:
        returns: DataFrame of asset returns
    """
    cor = compute_corr(returns)
    dendrogram = build_tree(cor=cor, method="single", bisection=True)

    assert dendrogram.root is not None
    assert dendrogram.linkage is not None

    assert dendrogram.names == [
        "UAA",
        "WMT",
        "SBUX",
        "AMD",
        "RRC",
        "GE",
        "T",
        "XOM",
        "BABA",
        "AAPL",
        "AMZN",
        "MA",
        "GOOG",
        "FB",
        "PFE",
        "GM",
        "BAC",
        "JPM",
        "SHLD",
        "BBY",
    ]


@pytest.mark.parametrize("method", ["single", "ward", "average", "complete"])
def test_invariant_order(returns: DataFrame, method: str) -> None:
    """Test that the order of nodes is invariant to the bisection parameter.

    This test verifies that regardless of whether bisection is used or not,
    the resulting dendrogram maintains the same order of assets, IDs, and names
    for different clustering methods.

    Args:
        returns: DataFrame of asset returns
        method: Clustering method to use (single, ward, average, or complete)
    """
    cor = compute_corr(returns)
    dendrogram1 = build_tree(cor=cor, method=method, bisection=True)
    dendrogram2 = build_tree(cor=cor, method=method, bisection=False)

    assert dendrogram1.assets == dendrogram2.assets
    assert dendrogram1.ids == dendrogram2.ids
    assert dendrogram1.names == dendrogram2.names


def test_bisect_tree_helper() -> None:
    """Test the module-level _bisect_tree helper."""
    root, next_id = _bisect_tree(ids=[0, 1, 2, 3], next_id=3)

    assert next_id == 6
    assert root.value == 6
    assert [leaf.value for leaf in root.leaves] == [0, 1, 2, 3]


def test_bisect_tree_helper_empty_ids() -> None:
    """Test _bisect_tree rejects empty ids input."""
    with pytest.raises(ValueError, match="at least one node id"):
        _bisect_tree(ids=[], next_id=0)


def test_get_linkage_helper() -> None:
    """Test the module-level _get_linkage helper."""
    root, _ = _bisect_tree(ids=[0, 1, 2, 3], next_id=3)

    assert _get_linkage(root) == [
        [0.0, 1.0, 3.0, 2.0],
        [2.0, 3.0, 3.0, 2.0],
        [4.0, 5.0, 7.0, 4.0],
    ]


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
    cor = compute_corr(returns)
    dendrogram = build_tree(cor=cor, method=method, bisection=False)

    assert dendrogram.root is not None
    assert dendrogram.method == method
    assert len(dendrogram.assets) == len(cor.columns)
    assert dendrogram.linkage is not None


def test_build_tree_bisection_true(returns: pl.DataFrame) -> None:
    """Test build_tree with bisection enabled."""
    cor = compute_corr(returns)
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


def test_build_tree_preserves_asset_order() -> None:
    """Test that build_tree preserves the order of assets from correlation matrix."""
    cor = pl.DataFrame({"X": [1.0, 0.5, 0.3], "Y": [0.5, 1.0, 0.4], "Z": [0.3, 0.4, 1.0]})

    dendrogram = build_tree(cor=cor, method="single", bisection=False)

    assert dendrogram.assets == cor.columns


def test_build_tree_requires_two_assets() -> None:
    """build_tree rejects correlation matrices with fewer than two assets."""
    cor = pl.DataFrame({"A": [1.0]})
    with pytest.raises(ValueError, match="at least two assets"):
        build_tree(cor)


def test_build_tree_non_finite_off_diagonal_raises() -> None:
    """build_tree rejects correlation matrices with non-finite off-diagonal entries."""
    cor = pl.DataFrame({"A": [1.0, np.inf], "B": [np.inf, 1.0]})
    with pytest.raises(ValueError, match="non-finite values"):
        build_tree(cor)


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


def test_compute_distance_matrix_clipping() -> None:
    """Test that distance computation correctly clips negative values."""
    cor = pl.DataFrame({"A": [1.0, 0.999999], "B": [0.999999, 1.0]})

    dist = _compute_distance_matrix(cor)

    assert (dist.to_numpy() >= 0).all()


@pytest.mark.property
@settings(deadline=None, max_examples=200)
@given(cor=correlation_matrices())
def test_build_tree_property_valid_corr_matrix(cor: pl.DataFrame) -> None:
    """build_tree should accept valid random correlation matrices."""
    dendrogram = build_tree(cor=cor, method="single", bisection=False)

    assert dendrogram.linkage is not None
    assert dendrogram.assets == cor.columns
    assert len(dendrogram.root.leaves) == len(cor.columns)


def test_build_tree_depth_one_bisection_two_assets() -> None:
    """Two-asset bisection tree should have depth 1 and valid weights."""
    cor = pl.DataFrame({"A": [1.0, 0.25], "B": [0.25, 1.0]})
    cov = pl.DataFrame({"A": [0.04, 0.01], "B": [0.01, 0.09]})

    dendrogram = build_tree(cor=cor, method="single", bisection=True)

    assert len(dendrogram.root.levels) == 2
    assert dendrogram.root.left is not None
    assert dendrogram.root.right is not None
    assert dendrogram.root.left.is_leaf
    assert dendrogram.root.right.is_leaf

    cluster = risk_parity(root=dendrogram.root, cov=cov)
    assert sum(cluster.portfolio.weights.values()) == pytest.approx(1.0)


def test_dendrogram_one_over_n_wrapper_matches_function() -> None:
    """Dendrogram.one_over_n delegates to algos.one_over_n on its own tree/assets."""
    from pyhrp.algos import one_over_n

    root = Cluster(10, left=Cluster(11, left=Cluster(1), right=Cluster(2)), right=Cluster(0))
    dendrogram = Dendrogram(root=root, assets=["A", "B", "C"])

    via_wrapper = [(lvl, dict(p.weights)) for lvl, p in dendrogram.one_over_n()]
    via_function = [(lvl, dict(p.weights)) for lvl, p in one_over_n(dendrogram.root, dendrogram.assets)]

    assert via_wrapper == via_function
    assert sum(via_wrapper[0][1].values()) == pytest.approx(1.0)
