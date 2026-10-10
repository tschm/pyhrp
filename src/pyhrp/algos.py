"""Portfolio optimization algorithms for hierarchical risk parity.

This module implements various portfolio optimization algorithms:
- risk_parity: The main hierarchical risk parity algorithm
- schur_risk_parity: Schur Complementary Allocation (Cotton, arXiv:2411.05807)
- one_over_n: A simple equal-weight allocation strategy

Allocator contract
------------------
All three allocators take the same inputs — a ``Cluster`` tree (``root``) plus
the asset names — and none of them changes the *shape* of that tree: no node is
added, removed or re-parented. They differ in where the weights land.

``risk_parity`` and ``schur_risk_parity`` share the recursive ``_allocate_with``
scaffolding and write **into the tree they are given**: every node's
``portfolio`` is replaced rather than accumulated into, and the same root object
is returned. Replacing makes them idempotent — re-running on an already weighted
tree, with a different covariance matrix or gamma, gives the same answer as
running on a fresh one — but the caller's tree *is* modified. Note in particular
that ``Dendrogram`` is a frozen dataclass while the ``Cluster`` it holds is not:
passing ``dendrogram.root`` to either allocator rewrites the portfolios inside
that dendrogram. Pass ``copy.deepcopy(dendrogram.root)`` to keep the original
tree unweighted.

``one_over_n`` differs on both counts. It accumulates into a local buffer, so it
leaves the input tree untouched entirely, and its *output* is a generator
yielding the equal-weight portfolio one tree level at a time (see its
docstring), because its purpose is to expose the allocation as it deepens rather
than a single final result.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from copy import deepcopy

import numpy as np
import polars as pl

from .cluster import Cluster, Portfolio, _portfolio
from .operators import CovarianceOperator, as_operator, solve_block

__all__ = ["one_over_n", "risk_parity", "schur_risk_parity"]


def risk_parity(root: Cluster, cov: pl.DataFrame | CovarianceOperator, assets: Sequence[str] | None = None) -> Cluster:
    """Compute hierarchical risk parity weights for a cluster tree.

    This is the main algorithm for hierarchical risk parity. It recursively
    traverses the cluster tree and assigns weights to each node based on
    the risk parity principle.

    Note:
        The tree is modified in place: the portfolio of every node is rebuilt
        from scratch and ``root`` itself is returned, so the function is
        idempotent and a tree can be reused with a different covariance matrix.
        Deep-copy the tree first if you need to keep it unweighted.

    The covariance is reached only through block products, one per child of
    every node, so a factor or Gram :class:`~pyhrp.operators.CovarianceOperator`
    allocates without forming the ``n x n`` matrix.

    Args:
        root (Cluster): The root node of the cluster tree
        cov (pl.DataFrame | CovarianceOperator): Covariance matrix of asset returns,
            as a DataFrame or as an operator
        assets (Sequence[str], optional): Asset names for an operator ``cov``;
            defaults to ``"0", "1", ...``. A DataFrame names its own assets.

    Returns:
        Cluster: The same root node, with portfolio weights assigned

    Examples:
        >>> import polars as pl
        >>> from pyhrp.cluster import Cluster
        >>> from pyhrp.algos import risk_parity
        >>> cov = pl.DataFrame({"A": [4.0, 0.0], "B": [0.0, 1.0]})
        >>> root = Cluster(2, left=Cluster(0), right=Cluster(1))
        >>> cluster = risk_parity(root=root, cov=cov)
        >>> round(cluster.portfolio["B"], 1)
        0.8
    """

    def node_variances(left: Block, right: Block, op: CovarianceOperator) -> tuple[float, float]:
        """Plain block variance of each child sub-portfolio."""
        return _block_variance(left, op), _block_variance(right, op)

    return _allocate_with(root, cov, node_variances, assets)


def schur_risk_parity(
    root: Cluster,
    cov: pl.DataFrame | CovarianceOperator,
    gamma: float = 0.5,
    assets: Sequence[str] | None = None,
) -> Cluster:
    """Compute Schur Complementary Allocation weights for a cluster tree.

    An extension of HRP introduced by Peter Cotton (arXiv:2411.05807) that augments
    sub-covariance matrices with off-diagonal block information via Schur complements.
    At gamma=0 this recovers standard HRP; at gamma=1 it recovers the minimum-variance
    portfolio through the same recursive structure.

    Note:
        The tree is modified in place: the portfolio of every node is rebuilt
        from scratch and ``root`` itself is returned, so the function is
        idempotent and a tree can be reused with a different covariance matrix
        or gamma. Deep-copy the tree first if you need to keep it unweighted.

    The Schur terms need, per node, the two cross-block products
    ``Sigma[L, R] w_R`` and ``Sigma[R, L] w_L`` and one solve against each
    child's principal block, which a factor operator does by Woodbury in
    ``O(|C| k**2)`` rather than ``O(|C|**3)``. The block quadratic forms
    ``B D^{-1} B^T`` are never formed: only their action on the child weights is.

    Args:
        root (Cluster): The root node of the cluster tree
        cov (pl.DataFrame | CovarianceOperator): Covariance matrix of asset returns,
            as a DataFrame or as an operator
        gamma (float): Interpolation parameter in [0, 1]. 0 = HRP, 1 = minimum variance.
        assets (Sequence[str], optional): Asset names for an operator ``cov``;
            defaults to ``"0", "1", ...``. A DataFrame names its own assets.

    Returns:
        Cluster: The same root node, with portfolio weights assigned

    Raises:
        ValueError: If gamma is outside the interval [0, 1].

    Examples:
        >>> import polars as pl
        >>> from pyhrp.cluster import Cluster
        >>> from pyhrp.algos import schur_risk_parity
        >>> cov = pl.DataFrame({"A": [4.0, 0.0], "B": [0.0, 1.0]})
        >>> root = Cluster(2, left=Cluster(0), right=Cluster(1))
        >>> cluster = schur_risk_parity(root=root, cov=cov, gamma=0.5)
        >>> round(cluster.portfolio["B"], 1)
        0.8
    """
    if not 0.0 <= gamma <= 1.0:
        msg = f"gamma must be in [0, 1], got {gamma}"
        raise ValueError(msg)

    def node_variances(left: Block, right: Block, op: CovarianceOperator) -> tuple[float, float]:
        """Schur-augmented block variance of each child, conditioned on the other."""
        (li, w_left), (ri, w_right) = left, right

        v_left = _block_variance(left, op)
        v_right = _block_variance(right, op)
        if gamma == 0.0:
            return v_left, v_right

        # Schur-augmented blocks A - gamma B D^{-1} B^T and D - gamma B^T A^{-1} B
        # (A = Sigma_LL, B = Sigma_LR, D = Sigma_RR), applied to the child weights
        # through one cross product and one block solve each.
        bt_w = op.block_matvec(ri, li, w_left)
        b_w = op.block_matvec(li, ri, w_right)
        v_left -= gamma * float(bt_w @ solve_block(op, ri, bt_w))
        v_right -= gamma * float(b_w @ solve_block(op, li, b_w))
        return v_left, v_right

    return _allocate_with(root, cov, node_variances, assets)


# A child sub-portfolio as (positions into the covariance, weights), in leaf order.
Block = tuple[np.ndarray, np.ndarray]

# Given a node's two children and the covariance operator, return the
# (v_left, v_right) risk pair used to split the node.
NodeVariances = Callable[[Block, Block, CovarianceOperator], tuple[float, float]]


def _allocate_with(
    root: Cluster,
    cov: pl.DataFrame | CovarianceOperator,
    node_variances: NodeVariances,
    assets: Sequence[str] | None = None,
) -> Cluster:
    """Shared scaffolding for the recursive risk-based allocators.

    Wraps the covariance as an operator, then walks the tree bottom-up, splitting
    each node's weight between its children inversely to the ``(v_left, v_right)``
    pair supplied by ``node_variances``. The only thing that distinguishes
    ``risk_parity`` from ``schur_risk_parity`` is that per-node variance rule;
    everything else lives here.

    The walk holds one weight per asset, not one portfolio per node. Every node's
    subtree covers a contiguous slice of the leaf order, so after its children are
    split that slice holds the node's own portfolio; scaling it in place by the
    node's split turns it into the parent's share. Memory is ``O(n)`` for any tree
    shape. Each node keeps only its split, from which its portfolio is rebuilt
    when read (see :class:`~pyhrp.cluster.Cluster`).

    Args:
        root (Cluster): The root node of the cluster tree.
        cov (pl.DataFrame | CovarianceOperator): Covariance matrix of asset returns.
        node_variances (NodeVariances): Per-node rule mapping a node's left/right
            child sub-portfolios (and the covariance operator) to the
            ``(v_left, v_right)`` risk pair used to split that node.
        assets (Sequence[str], optional): Asset names for an operator ``cov``.

    Returns:
        Cluster: The root node with portfolio weights assigned.

    Raises:
        ValueError: If the tree's leaves do not index the covariance columns one-to-one.
    """
    op, names = as_operator(cov, assets)

    # Iterative layout rather than recursion: a chain-degenerate tree is as deep as
    # the universe is wide. The layout validates every non-leaf node, so a malformed
    # tree raises before any weight is written.
    leaves, spans = root._layout()

    # A leaf's value indexes into the assets, so a tree built for a different universe
    # would either silently drop assets or fail with a bare IndexError deep in the walk.
    leaf_values = sorted(int(leaf.value) for leaf in leaves)
    if leaf_values != list(range(op.n)):
        msg = (
            f"Cluster tree does not match the covariance matrix: expected {op.n} leaves "
            f"indexing columns 0..{op.n - 1}, got {len(leaf_values)} leaves with values {leaf_values}"
        )
        raise ValueError(msg)

    position = np.array([int(leaf.value) for leaf in leaves], dtype=np.intp)
    w = np.ones(len(leaves))
    shares: list[float] = []
    for _, lo, mid, hi in spans:
        v_left, v_right = node_variances((position[lo:mid], w[lo:mid]), (position[mid:hi], w[mid:hi]), op)
        share = _left_share(v_left, v_right)
        w[lo:mid] *= share
        w[mid:hi] *= 1.0 - share
        shares.append(share)

    # Every node's portfolio is replaced, never accumulated into, which keeps
    # repeated allocations on the same tree idempotent.
    for leaf in leaves:
        leaf._set_allocation(1.0, names)
    for (node, _, _, _), share in zip(spans, shares, strict=True):
        node._set_allocation(share, names)
    # The root's portfolio is the weight vector just computed; keep it rather than rebuild it.
    root.portfolio = _portfolio(leaves, w, names)
    return root


def _block_variance(block: Block, op: CovarianceOperator) -> float:
    """Compute the variance of a sub-portfolio from one block product with the covariance operator."""
    idx, w = block
    return float(w @ op.block_matvec(idx, idx, w))


def _left_share(v_left: float, v_right: float) -> float:
    """The share of a node's weight that goes to its left child, inversely proportional to risk.

    The split satisfies v_left * alpha_left == v_right * alpha_right with
    alpha_left + alpha_right == 1. If both variances are zero (e.g. riskless
    sub-portfolios), the weight is split equally.

    Args:
        v_left (float): Variance of the left sub-portfolio
        v_right (float): Variance of the right sub-portfolio

    Returns:
        float: ``alpha_left``, in [0, 1]
    """
    total = v_left + v_right
    return v_right / total if total > 0 else 0.5


def one_over_n(root: Cluster, assets: list[str]) -> Generator[tuple[int, Portfolio]]:
    """Generate 1/N (equal-weight) portfolios one tree level at a time.

    This implements a hierarchical 1/N strategy where weights are distributed
    equally among the leaves of each cluster, and the weight budget halves at
    each successive level of the tree.

    Unlike :func:`risk_parity` and :func:`schur_risk_parity` — which rebuild a
    single final allocation and return the root ``Cluster`` — this allocator is
    intentionally a **generator**: its purpose is to expose the equal-weight
    allocation as the tree deepens, yielding one portfolio per level. It shares
    the sibling input contract (a ``Cluster`` tree plus the asset names) and, like
    them, does not mutate the tree: weights accumulate in a local buffer, so a
    leaf that terminates at a shallow level keeps its weight in the deeper levels
    (each yielded portfolio is therefore a complete allocation over all assets),
    and re-running on the same tree yields an identical sequence.

    Args:
        root (Cluster): The root node of the cluster tree.
        assets (list[str]): Asset names; a leaf's value indexes into this list.

    Yields:
        tuple[int, Portfolio]: The level number and the (cumulative) equal-weight
        portfolio at that level.

    Examples:
        >>> import polars as pl
        >>> from pyhrp.hrp import build_tree
        >>> from pyhrp.algos import one_over_n
        >>> cor = pl.DataFrame({"A": [1.0, 0.3], "B": [0.3, 1.0]})
        >>> dg = build_tree(cor, method="ward")
        >>> levels = list(one_over_n(dg.root, dg.assets))
        >>> len(levels) > 0
        True
    """
    # Accumulate into a local buffer so the input tree is never mutated.
    portfolio = Portfolio()

    # Initial weight to distribute
    w: float = 1.0

    # Process each level of the tree
    for n, level in enumerate(root.levels):
        for node in level:
            # Distribute weight equally among all leaves in this node
            for leaf in node.leaves:
                portfolio[assets[leaf.value]] = w / node.leaf_count

        # Reduce weight for the next level
        w *= 0.5

        # Yield the current level number and a deep copy of the portfolio
        yield n, deepcopy(portfolio)
