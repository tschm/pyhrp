import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


def dendrogram(links, ax=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(figsize=(25, 20))
    sch.dendrogram(links, ax=ax, **kwargs)
    return ax
