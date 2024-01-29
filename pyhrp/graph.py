"""display a dendrogram"""

from __future__ import annotations

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


def dendrogram(links, ax=None, **kwargs):
    """Plot a dendrogram using matplotlib"""
    if ax is None:
        _, ax = plt.subplots(figsize=(25, 20))
    sch.dendrogram(links, ax=ax, **kwargs)
    return ax
