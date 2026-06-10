"""Hierarchical Risk Parity (HRP) portfolio optimization library.

This package implements the Hierarchical Risk Parity algorithm for portfolio
optimization, as introduced by Marcos Lopez de Prado, and the Schur
Complementary Allocation extension by Peter Cotton.
"""

import importlib.metadata

from .algos import one_over_n, risk_parity, schur_risk_parity
from .cluster import Cluster, Portfolio
from .hrp import Dendrogram, build_tree, compute_corr, compute_cov, hrp, schur_hrp

__version__ = importlib.metadata.version("pyhrp")

__all__ = [
    "Cluster",
    "Dendrogram",
    "Portfolio",
    "__version__",
    "build_tree",
    "compute_corr",
    "compute_cov",
    "hrp",
    "one_over_n",
    "risk_parity",
    "schur_hrp",
    "schur_risk_parity",
]
