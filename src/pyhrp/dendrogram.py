"""Hierarchical clustering tree construction and the Dendrogram container.

This module builds the hierarchical clustering tree consumed by the HRP
allocation entry points and stores it in a :class:`Dendrogram`:
- build_tree: Build a hierarchical cluster tree from a correlation matrix
- build_tree_from_operator: Build the tree from a covariance operator, matrix-free for single linkage
- Dendrogram: Container for the clustering result and its visualization
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from .algos import one_over_n as _one_over_n
from .cluster import Cluster, Portfolio
from .operators import CovarianceOperator, as_operator, block

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["Dendrogram", "build_tree", "build_tree_from_operator"]


@dataclass(frozen=True)
class Dendrogram:
    """Container for hierarchical clustering dendrogram data and visualization.

    This class stores the results of hierarchical clustering and provides methods
    for accessing and visualizing the dendrogram structure.

    Attributes:
        root (Cluster): The root node of the hierarchical clustering tree
        assets (list[str]): Names of assets included in the clustering
        linkage (np.ndarray | None): Linkage matrix in scipy format for plotting
        distance (pl.DataFrame | None): Distance matrix used for clustering
        method (str | None): Linkage method used for clustering
    """

    root: Cluster
    assets: list[str]
    distance: pl.DataFrame | None = None
    linkage: np.ndarray | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        """Validate dataclass fields after initialization.

        Ensures that the optional distance matrix, when provided, is a polars
        DataFrame with columns aligned to the asset list, and verifies that the
        number of leaves in the cluster tree matches the number of assets.
        """
        if self.distance is not None:
            if not isinstance(self.distance, pl.DataFrame):
                msg = "distance must be a polars DataFrame."
                raise TypeError(msg)

            if self.distance.columns != list(self.assets):
                msg = "Distance matrix index/columns must align with assets."
                raise ValueError(msg)

        if len(self.root.leaves) != len(self.assets):
            msg = "Number of leaves does not match number of assets."
            raise ValueError(msg)

    def plot(self, **kwargs: object) -> go.Figure:
        """Build and return a plotly dendrogram figure.

        Delegates to :func:`pyhrp.plot.plot_dendrogram`; the plotly dependency
        is imported lazily so importing the allocation core stays plotly-free.
        """
        from .plot import plot_dendrogram

        return plot_dendrogram(self, **kwargs)

    def one_over_n(self) -> Generator[tuple[int, Portfolio]]:
        """Yield the hierarchical 1/N portfolios level by level for this tree.

        Container-level convenience wrapper around :func:`pyhrp.algos.one_over_n`.

        Yields:
            tuple[int, Portfolio]: The level number and the equal-weight portfolio
            at that level.
        """
        yield from _one_over_n(self.root, self.assets)

    @property
    def ids(self) -> list[int]:
        """Node values in the order left -> right as they appear in the dendrogram."""
        return [node.value for node in self.root.leaves]

    @property
    def names(self) -> list[str]:
        """The asset names as induced by the order of ids."""
        return [self.assets[i] for i in self.ids]


def _compute_distance_matrix(corr: pl.DataFrame) -> pl.DataFrame:
    """Convert correlation matrix to distance matrix."""
    c = corr.to_numpy()
    dist = np.sqrt(np.clip((1.0 - c) / 2.0, a_min=0.0, a_max=1.0))
    np.fill_diagonal(dist, 0.0)
    cols = corr.columns
    return pl.DataFrame(dict(zip(cols, dist, strict=True)))


def _bisect_tree(ids: list[int], next_id: int) -> tuple[Cluster, int]:
    """Build tree by recursive bisection."""
    if not ids:
        msg = "ids must contain at least one node id."
        raise ValueError(msg)
    if len(ids) == 1:
        return Cluster(value=ids[0]), next_id

    mid = len(ids) // 2
    left_ids, right_ids = ids[:mid], ids[mid:]
    left, next_id = _bisect_tree(left_ids, next_id)
    right, next_id = _bisect_tree(right_ids, next_id)
    next_id += 1
    return Cluster(value=next_id, left=left, right=right), next_id


def _cluster_diameter(node: Cluster, dist: np.ndarray) -> float:
    """Largest pairwise distance between any two leaves under ``node``.

    This is the merge height used for the bisection tree's linkage rows. The
    diameter is monotone along the tree — a parent's leaf set is a superset of
    each child's, so its maximum can only grow — which is what scipy's linkage
    format requires of its distance column.

    Args:
        node (Cluster): The subtree whose leaves span the cluster.
        dist (np.ndarray): Square distance matrix indexed by leaf value.

    Returns:
        float: The cluster diameter; 0.0 for a single leaf.
    """
    idx = [int(leaf.value) for leaf in node.leaves]
    if len(idx) < 2:
        return 0.0
    return float(dist[np.ix_(idx, idx)].max())


def _get_linkage(node: Cluster, dist: np.ndarray) -> list[list[float]]:
    """Convert tree structure back to linkage matrix format.

    Rows are emitted in post-order, matching the id assignment in
    :func:`_bisect_tree`, so row ``i`` defines the cluster with id ``n + i``.
    Column 2 carries the cluster diameter (see :func:`_cluster_diameter`),
    because scipy reads that column as a cophenetic merge distance.

    Args:
        node (Cluster): The subtree to linearise.
        dist (np.ndarray): Square distance matrix indexed by leaf value.

    Returns:
        list[list[float]]: Rows of ``[left id, right id, height, leaf count]``.
    """
    links_list: list[list[float]] = []
    if node.left is not None and node.right is not None:
        left, right = node._child_clusters()
        links_list.extend(_get_linkage(left, dist))
        links_list.extend(_get_linkage(right, dist))
        links_list.append(
            [
                float(left.value),
                float(right.value),
                _cluster_diameter(node, dist),
                float(len(left.leaves) + len(right.leaves)),
            ]
        )
    return links_list


def _check_finite_correlations(cor: pl.DataFrame, c: np.ndarray) -> None:
    """Raise if the correlation matrix contains non-finite values.

    Names the offending assets when the non-finite values sit on the diagonal,
    since a constant (zero-variance) price series is the usual cause.
    """
    bad = [col for col, diag in zip(cor.columns, np.diagonal(c), strict=True) if not np.isfinite(diag)]
    if bad:
        msg = (
            f"Correlation matrix contains non-finite values for assets {bad}; "
            "constant (zero-variance) price series produce NaN correlations."
        )
        raise ValueError(msg)
    if not np.isfinite(c).all():
        msg = "Correlation matrix contains non-finite values."
        raise ValueError(msg)


def _validate_correlation_matrix(cor: pl.DataFrame) -> None:
    """Validate the correlation matrix accepted by :func:`build_tree`.

    Raises:
        TypeError: If ``cor`` is not a polars DataFrame.
        ValueError: If it has fewer than two assets or contains non-finite values.
    """
    if not isinstance(cor, pl.DataFrame):
        msg = "Correlation matrix must be a polars DataFrame."
        raise TypeError(msg)
    if len(cor.columns) < 2:
        msg = "Correlation matrix must contain at least two assets."
        raise ValueError(msg)
    _check_finite_correlations(cor, cor.to_numpy())


def _to_cluster(node: sch.ClusterNode) -> Cluster:
    """Convert a scipy ClusterNode tree into our Cluster format.

    Args:
        node (sch.ClusterNode): A node from scipy's hierarchical clustering.

    Returns:
        Cluster: Equivalent node in our Cluster format.
    """
    # Two passes instead of recursion. scipy's tree is as deep as the linkage is
    # unbalanced -- single linkage chains, so depth grows with the asset count --
    # and recursing here raised RecursionError before allocation was ever reached.
    #
    # Pass one collects nodes parent-before-child; reversing that order visits every
    # child before its parent, which is what building bottom-up requires.
    order: list[sch.ClusterNode] = []
    stack: list[sch.ClusterNode] = [node]
    while stack:
        current = stack.pop()
        order.append(current)
        if current.left is not None and current.right is not None:
            stack.append(current.left)
            stack.append(current.right)

    # Pass two builds each Cluster once its children exist. scipy assigns every
    # node a unique id, so the id is a safe key for the partially built tree.
    converted: dict[int, Cluster] = {}
    for current in reversed(order):
        if current.left is not None and current.right is not None:
            converted[current.id] = Cluster(
                value=current.id,
                left=converted[current.left.id],
                right=converted[current.right.id],
            )
        else:
            converted[current.id] = Cluster(value=current.id)

    return converted[node.id]


def build_tree(
    cor: pl.DataFrame, method: Literal["single", "complete", "average", "ward"] = "ward", bisection: bool = False
) -> Dendrogram:
    """Build hierarchical cluster tree from correlation matrix.

    This function converts a correlation matrix to a distance matrix, performs
    hierarchical clustering, and returns a Dendrogram object containing the
    resulting tree structure.

    Note:
        There is no limit on the depth of the resulting tree. That matters because
        ``method="single"`` chains: on smoothly decaying correlations it produces a
        tree of depth *n* rather than *O(log n)*, so an ordinary equity universe can
        exceed Python's recursion limit. Tree construction and every traversal that
        walks the result are iterative for exactly that reason.

    Args:
        cor (pl.DataFrame): Correlation matrix of asset returns (columns are assets)
        method (Literal["single", "complete", "average", "ward"]): Linkage method for hierarchical clustering
            - "single": minimum distance between points (nearest neighbor)
            - "complete": maximum distance between points (furthest neighbor)
            - "average": average distance between all points
            - "ward": Ward variance minimization
        bisection (bool): Whether to use bisection method for tree construction

    Returns:
        Dendrogram: Object containing the hierarchical clustering tree, with:
            - root: Root cluster node
            - linkage: Linkage matrix for plotting
            - assets: List of assets
            - method: Clustering method used
            - distance: Distance matrix

    Examples:
        >>> import polars as pl
        >>> from pyhrp.dendrogram import build_tree
        >>> cor = pl.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]})
        >>> dg = build_tree(cor, method="ward")
        >>> dg.root.leaf_count
        2
    """
    _validate_correlation_matrix(cor)
    dist = _compute_distance_matrix(cor)
    links = sch.linkage(ssd.squareform(dist.to_numpy(), checks=False), method=method)

    root = _to_cluster(sch.to_tree(links, rd=False))

    # Apply bisection if requested
    if bisection:
        # Rebuild tree using bisection
        leaf_ids: list[int] = [int(node.value) for node in root.leaves]
        root, _ = _bisect_tree(ids=leaf_ids, next_id=max(leaf_ids))
        links = np.array(_get_linkage(root, dist.to_numpy()))

    return Dendrogram(root=root, linkage=links, method=method, distance=dist, assets=cor.columns)


def _volatilities(op: CovarianceOperator) -> np.ndarray:
    """Return ``sqrt(diag(Sigma))``, refusing an asset without positive variance."""
    var = np.asarray(op.diag, dtype=np.float64)
    if not np.all(var > 0.0):
        bad = np.flatnonzero(~(var > 0.0)).tolist()
        msg = f"Covariance has non-positive or non-finite variance at positions {bad}; correlations are undefined."
        raise ValueError(msg)
    return np.sqrt(var)


def _unit(n: int, i: int) -> np.ndarray:
    """The ``i``-th unit vector of length ``n``."""
    e = np.zeros(n)
    e[i] = 1.0
    return e


def _linkage_from_merges(merges: list[tuple[float, int, int]], n: int) -> np.ndarray:
    """Turn merges into SciPy's linkage format.

    Each merge ``(height, i, j)`` joins the clusters holding assets ``i`` and
    ``j``. Merges are sorted by height and replayed by union-find, and each row
    lists its two cluster ids smaller first, as SciPy labels them.

    Args:
        merges (list[tuple[float, int, int]]): The ``n - 1`` merges, in any order.
        n (int): The number of assets.

    Returns:
        np.ndarray: The ``(n - 1) x 4`` linkage matrix.
    """
    # Union-find over the assets; label[r] is the cluster id of the set rooted at r.
    root = np.arange(n)
    label = np.arange(n)
    size = np.ones(n, dtype=np.intp)

    def find(i: int) -> int:
        """Return the representative of ``i``, halving the path on the way."""
        while root[i] != i:
            root[i] = root[root[i]]
            i = int(root[i])
        return i

    links = np.empty((n - 1, 4))
    for k, (height, i, j) in enumerate(sorted(merges, key=lambda merge: merge[0])):
        a, b = find(i), find(j)
        links[k] = (min(label[a], label[b]), max(label[a], label[b]), height, size[a] + size[b])
        root[b] = a
        label[a] = n + k
        size[a] += size[b]
    return links


def _single_linkage(op: CovarianceOperator, vol: np.ndarray) -> np.ndarray:
    """Single linkage on ``d_ij = sqrt((1 - rho_ij) / 2)`` without the distance matrix.

    Single linkage is the minimum spanning tree of the distance graph. Prim's
    algorithm grows it from one column of ``Sigma`` per step, so memory is
    ``O(n)`` beyond the operator and the time is ``n`` column products
    (``O(n k)`` each for a rank-``k`` factor or Gram operator).

    Args:
        op (CovarianceOperator): The covariance.
        vol (np.ndarray): The volatilities ``sqrt(diag(Sigma))``.

    Returns:
        np.ndarray: The ``(n - 1) x 4`` linkage matrix in SciPy's format.
    """
    n = op.n
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    parent = np.zeros(n, dtype=np.intp)
    edges: list[tuple[float, int, int]] = []
    current = 0
    for _ in range(n - 1):
        in_tree[current] = True
        rest = np.flatnonzero(~in_tree)
        column = np.asarray(op.matvec(_unit(n, current)))[rest]
        rho = column / (vol[rest] * vol[current])
        dist = np.sqrt(np.clip((1.0 - rho) / 2.0, 0.0, 1.0))
        closer = dist < best[rest]
        best[rest[closer]] = dist[closer]
        parent[rest[closer]] = current
        current = int(rest[np.argmin(best[rest])])
        edges.append((float(best[current]), int(parent[current]), current))
    return _linkage_from_merges(edges, n)


def _ward_linkage(op: CovarianceOperator, vol: np.ndarray) -> np.ndarray:
    """Ward linkage on ``d_ij = sqrt((1 - rho_ij) / 2)`` without the distance matrix.

    These distances are Euclidean: they are ``|p_i - p_j|`` for points with inner
    products ``p_i . p_j = rho_ij / 4``. The centroid of a cluster ``A`` is then
    reached through the vector ``v_A`` holding ``1 / (|A| sigma_i)`` on its assets,
    with ``c_A . c_B = v_A' Sigma v_B / 4``. One product ``Sigma v_A`` therefore gives
    the centroid inner products of ``A`` with every cluster at once (a sum over each
    cluster's assets), and with them the Ward distance

        ``D(A, B)**2 = 2 |A| |B| / (|A| + |B|) * |c_A - c_B|**2``,

    which is the distance SciPy's Lance-Williams update reports. The nearest-
    neighbour chain algorithm needs ``O(n)`` such queries, so the time is ``O(n)``
    operator products (``O(n k)`` each for a rank-``k`` factor or Gram operator)
    and the memory is ``O(n)``.

    Args:
        op (CovarianceOperator): The covariance.
        vol (np.ndarray): The volatilities ``sqrt(diag(Sigma))``.

    Returns:
        np.ndarray: The ``(n - 1) x 4`` linkage matrix in SciPy's format.
    """
    n = op.n
    # Every cluster is named by one of its assets, its representative; owner maps an
    # asset to the representative of its cluster.
    owner = np.arange(n)
    size = np.ones(n)
    norm2 = np.full(n, 0.25)  # |c_A|**2; a single asset has rho_ii / 4
    active = np.ones(n, dtype=bool)
    merges: list[tuple[float, int, int]] = []
    chain: list[int] = []
    while len(merges) < n - 1:
        if not chain:
            chain.append(int(np.flatnonzero(active)[0]))
        a = chain[-1]
        members = np.flatnonzero(owner == a)
        v = np.zeros(n)
        v[members] = 1.0 / (size[a] * vol[members])
        g = np.asarray(op.matvec(v))
        dot = np.bincount(owner, weights=g / vol, minlength=n) / (4.0 * size)
        d2 = 2.0 * size[a] * size / (size[a] + size) * np.maximum(norm2[a] + norm2 - 2.0 * dot, 0.0)
        d2[~active] = np.inf
        d2[a] = np.inf
        b = int(np.argmin(d2))
        # Reciprocal nearest neighbours: a's nearest is the cluster below it on the chain.
        if len(chain) > 1 and d2[chain[-2]] <= d2[b]:
            b = chain[-2]
            del chain[-2:]
            sa, sb = size[a], size[b]
            norm2[a] = (sa * sa * norm2[a] + 2.0 * sa * sb * dot[b] + sb * sb * norm2[b]) / (sa + sb) ** 2
            size[a] = sa + sb
            owner[owner == b] = a
            active[b] = False
            merges.append((float(np.sqrt(d2[b])), a, b))
        else:
            chain.append(b)
    return _linkage_from_merges(merges, n)


def _bisect_linkage(root: Cluster, links: np.ndarray) -> tuple[Cluster, np.ndarray]:
    """Rebuild a tree by bisection of its leaf order, without distances.

    The height of a bisection node is the largest merge height between
    neighbouring leaves in its range, which for a monotone tree is the range's
    cophenetic diameter. Neighbouring leaves ``i`` and ``i + 1`` of a dendrogram
    meet at the node that splits between them, so these gaps come from the
    original tree in one pass. A parent's range contains its children's, so no
    node sits below its children.

    Args:
        root (Cluster): The clustering tree (single or Ward linkage).
        links (np.ndarray): Its ``(n - 1) x 4`` linkage matrix.

    Returns:
        tuple[Cluster, np.ndarray]: The bisection tree and its linkage matrix.
    """
    n = links.shape[0] + 1
    leaves, spans = root._layout()
    gaps = np.empty(n - 1)
    for node, _, mid, _ in spans:
        gaps[mid - 1] = links[int(node.value) - n, 2]

    ids = [int(leaf.value) for leaf in leaves]
    bisected, _ = _bisect_tree(ids=ids, next_id=n - 1)
    rows = np.empty((n - 1, 4))
    for node, lo, _, hi in bisected._layout()[1]:
        left, right = node._child_clusters()
        rows[int(node.value) - n] = (left.value, right.value, gaps[lo : hi - 1].max(), hi - lo)
    return bisected, rows


def build_tree_from_operator(
    cov: CovarianceOperator,
    assets: Sequence[str] | None = None,
    method: Literal["single", "complete", "average", "ward"] = "ward",
    bisection: bool = False,
) -> Dendrogram:
    """Build the hierarchical cluster tree from a covariance operator.

    The correlation distance is ``d_ij = sqrt((1 - rho_ij) / 2)``, as in
    :func:`build_tree`. Ward (the default) and single linkage are built
    **matrix-free**, from the operator's diagonal and ``O(n)`` products with it
    (see :func:`_ward_linkage` and :func:`_single_linkage`), never forming the
    ``n x n`` correlation or distance matrix, so a factor or Gram operator
    clusters a universe whose covariance would not fit in memory. They give
    SciPy's trees. The returned :class:`Dendrogram` carries no ``distance``
    frame.

    With ``bisection=True`` the leaf order is split at its midpoint, as in
    Lopez de Prado's original algorithm, giving ``ceil(log2(n))`` levels; this
    matters for single linkage, whose tree chains to nearly ``n`` levels.
    Matrix-free, the merge height of a bisection node is the cophenetic
    diameter of its range in the clustering tree, not the largest pairwise
    distance :func:`build_tree` reports; the tree is the same, only the plotted
    heights differ.

    Complete and average linkage need all pairwise distances; the correlation
    is then materialised from the operator and the call is delegated to
    :func:`build_tree`, at ``O(n**2)`` memory.

    Args:
        cov (CovarianceOperator): The covariance as an operator.
        assets (Sequence[str], optional): Asset names; defaults to ``"0", "1", ...``.
        method (Literal["single", "complete", "average", "ward"]): Linkage method.
            Defaults to ``"ward"``; ``"ward"`` and ``"single"`` are matrix-free.
        bisection (bool): Whether to rebuild the tree by bisection of the leaf order.

    Returns:
        Dendrogram: The clustering tree over the operator's assets.

    Raises:
        ValueError: If there are fewer than two assets, an asset has no positive
            variance, or ``assets`` has the wrong length.

    Examples:
        >>> import numpy as np
        >>> from pyhrp.dendrogram import build_tree_from_operator
        >>> from pyhrp.operators import DenseCovariance
        >>> sigma = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
        >>> dg = build_tree_from_operator(DenseCovariance(sigma), assets=["A", "B", "C"], method="single")
        >>> dg.names
        ['C', 'A', 'B']
    """
    op, names = as_operator(cov, assets)
    if op.n < 2:
        msg = "Covariance must contain at least two assets."
        raise ValueError(msg)
    vol = _volatilities(op)

    if method in ("single", "ward"):
        links = _single_linkage(op, vol) if method == "single" else _ward_linkage(op, vol)
        root = _to_cluster(sch.to_tree(links, rd=False))
        if bisection:
            root, links = _bisect_linkage(root, links)
        return Dendrogram(root=root, linkage=links, method=method, assets=names)

    idx = np.arange(op.n)
    corr = block(op, idx, idx) / np.outer(vol, vol)
    np.fill_diagonal(corr, 1.0)
    return build_tree(pl.DataFrame(dict(zip(names, corr, strict=True))), method=method, bisection=bisection)
