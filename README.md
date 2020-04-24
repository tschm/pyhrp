# pyhrp

A recursive implementation of the Hierarchical Risk Parity (hrp) approach by Marcos Lopez de Prado.
We take heavily advantage of the scipy.cluster.hierarchy package. 

Here's a simple example

```python
import numpy as np

from pyhrp.cluster import root
from pyhrp.graph import dendrogram
from pyhrp.hrp import dist, hrp_feed

# use a small covariance matrix
cov = np.array([[1, 0.2, 0], [0.2, 2, 0.0], [0, 0, 3]])

# we compute the rootnode of a graph here
# The rootnode points to left and right and has an id attribute.
rootnode, link = root(dist(cov), 'ward')

# plot the dendrogram
ax = dendrogram(link, orientation="left")
ax.get_figure().savefig("dendrogram.png")

v, weights = hrp_feed(rootnode, cov=cov)

print(v)
print(np.linalg.multi_dot([weights, cov, weights]))
print(weights)
print(weights.sum())
```

## Installation:
```
pip install pyhpr
```
