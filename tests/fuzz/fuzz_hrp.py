"""Fuzz the pyhrp Hierarchical Risk Parity pipeline against arbitrary prices.

``hrp`` runs the full pipeline — returns, covariance, correlation, hierarchical
clustering and risk-parity allocation — over an untrusted price frame. It is
contracted to either return a weighted cluster or raise one of its documented
errors on degenerate data (constant/zero-variance series produce NaN
correlations, too few assets, etc.); scipy/numpy may raise ``ValueError`` or
``LinAlgError`` on pathological matrices. Anything else is a crash. This harness
exercises that contract with coverage-guided input.

Run locally:
    pip install atheris polars numpy scipy
    python tests/fuzz/fuzz_hrp.py -atheris_runs=20000

Run in ClusterFuzzLite: this file is built by .clusterfuzzlite/build.sh.
"""

from __future__ import annotations

import contextlib
import sys

import atheris

# Pre-import the heavy native dependencies OUTSIDE the instrumentation block.
# Atheris's bytecode instrumentation miscompiles parts of polars' Python
# machinery, so we let these libraries load uninstrumented and instrument only
# the first-party package under test. Importing the top-level packages (and the
# scipy submodules pyhrp uses) pulls them into the module cache, so pyhrp's own
# imports below hit the cache and are never re-instrumented.
import numpy as np  # pre-imported uninstrumented (see note above)
import polars as pl
import scipy.cluster.hierarchy  # pre-imported uninstrumented
import scipy.spatial.distance  # noqa: F401  # pre-imported uninstrumented

with atheris.instrument_imports():
    from pyhrp.hrp import hrp

_METHODS = ("single", "complete", "average", "ward")
# Errors the pipeline is allowed to raise on degenerate input.
_ALLOWED = (ValueError, TypeError, np.linalg.LinAlgError)


def test_one_input(data: bytes) -> None:
    """Run a fuzzed price frame through the HRP pipeline."""
    fdp = atheris.FuzzedDataProvider(data)
    n_assets = fdp.ConsumeIntInRange(2, 5)
    n_rows = fdp.ConsumeIntInRange(2, 16)
    method = _METHODS[fdp.ConsumeIntInRange(0, len(_METHODS) - 1)]

    prices = pl.DataFrame({f"A{i}": [fdp.ConsumeFloat() for _ in range(n_rows)] for i in range(n_assets)})

    with contextlib.suppress(_ALLOWED):
        hrp(prices, method=method)


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
