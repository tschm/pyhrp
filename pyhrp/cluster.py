import scipy.cluster.hierarchy as sch


def root(dist, method="ward"):
    link = sch.linkage(dist, method=method, optimal_ordering=True)
    return sch.to_tree(link, rd=False), link
