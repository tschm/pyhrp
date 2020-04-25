# pyhrp

A recursive implementation of the Hierarchical Risk Parity (hrp) approach by Marcos Lopez de Prado.
We take heavily advantage of the scipy.cluster.hierarchy package. 

Here's a simple example

```python
import numpy as np

from pyhrp.graph import dendrogram
from pyhrp.hrp import hrp_feed, linkage, tree

from pyhrp.linalg import dist, correlation_from_covariance

# use a small covariance matrix
cov = np.array([[1, 0.5, 0.2], [0.5, 2, 0.2], [0.2, 0.2, 3]])

# we compute the root(node) of a graph here
link = linkage(dist(correlation_from_covariance(cov)), 'ward')
root = tree(link)

# plot the dendrogram
ax = dendrogram(link, orientation="left")
ax.get_figure().savefig("dendrogram.png")

v, weights = hrp_feed(node=root, cov=cov)

print(weights)
```

## Installation:
```
pip install pyhpr
```
