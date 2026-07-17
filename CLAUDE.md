# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyhrp is a recursive implementation of Hierarchical Risk Parity (HRP) by Marcos Lopez de Prado,
built on top of `scipy.cluster.hierarchy`. Given a price/return frame it builds a clustering tree
from asset co-movement and allocates risk recursively along that tree, avoiding explicit return
forecasting.

The public API is re-exported from `pyhrp` (`src/pyhrp/__init__.py`).

## Package layout

The `src/pyhrp/` package is split into small, focused modules:

| Module | Responsibility |
| --- | --- |
| `hrp.py` | High-level entry points. Turns a price/return frame into a weighted tree: `compute_cov`/`compute_corr` build the matrices, `build_tree` produces the `Dendrogram`, and `hrp`/`schur_hrp` run the full pipeline end to end. |
| `algos.py` | The allocation algorithms that size a cluster tree: `risk_parity` (recursive HRP), `schur_risk_parity` (Schur Complementary Allocation), and `one_over_n` (equal weight). |
| `cluster.py` | Core data structures — `Cluster` (a node in the hierarchical tree) and `Portfolio` (an asset-to-weight mapping with analysis/plot helpers). |
| `covariance.py` | `compute_cov`/`compute_corr` helpers that turn a returns frame into covariance/correlation matrices. |
| `dendrogram.py` | `build_tree` plus the `Dendrogram` container (the clustering result and its plotting/ordering helpers). |
| `treelib.py` | A minimal generic binary-tree `Node`, kept in-house to avoid a `binarytree` dependency. |
| `plot.py` | Plotly dendrogram rendering (`plot_dendrogram`), kept separate so the optional plotting dependency does not couple the algorithm modules. |
| `__init__.py` | Public API surface — re-exports the functions and classes above and exposes `__version__`. |

## Module layering

Dependencies flow in one direction, from the generic tree up to the public surface:

```text
treelib  ->  cluster  ->  {algos, dendrogram}  ->  hrp  ->  __init__
```

- `treelib` depends on nothing internal; `cluster.Cluster` subclasses `treelib.Node`.
- `algos` and `dendrogram` build on `cluster` (and `covariance`); neither imports `hrp`.
- `hrp` composes `covariance` + `dendrogram` + `algos` into the end-to-end pipeline.
- `plot`/Plotly is imported lazily (inside `Dendrogram.plot`) so importing the allocation
  core stays plotly-free.

### Allocator contract

The three allocators in `algos.py` share one contract: each takes a `Cluster` tree (`root`)
plus the asset names, and never mutates the tree — weights are rebuilt from scratch, so every
allocator is idempotent and a tree can be reused. `risk_parity` and `schur_risk_parity` share the
`_allocate_with` scaffolding and return the fully weighted root `Cluster`. `one_over_n`
intentionally differs only in its output: it is a generator yielding the equal-weight portfolio
one tree level at a time (`Dendrogram.one_over_n()` is the container-level convenience wrapper).

## Development commands

```bash
make install    # Install uv, create .venv, install dependencies
make test       # Run pytest with coverage (fails under the coverage gate)
make fmt        # Run the pre-commit hooks: ruff format + check, markdownlint
make typecheck  # Run mypy in strict mode over src/
make book       # Build the Marimo/mkdocs documentation
```

## Key invariants

CI enforces these gates — keep them green when changing code:

- **100% test coverage** — `make test` fails below the coverage threshold; cover every new branch.
- **100% docstring coverage** — `interrogate --fail-under 100`; every public module, class,
  and function needs a docstring (Google style).
- **Strict typing** — `mypy` runs in strict mode over `src/`; annotate all public signatures.
- **1:1 test/source layout** — each `src/pyhrp/<mod>.py` has a mirrored `tests/pyhrp/test_<mod>.py`.
- **Linting/formatting** — `ruff` (line length 120, Google-style docstrings) plus `markdownlint`.

## Tech stack

- **Package manager**: uv
- **Core dependencies**: NumPy, polars, scipy (Plotly is optional, for plotting only)
- **Testing**: pytest with coverage and Hypothesis property tests
- **Docs**: Marimo notebooks (`book/marimo/`) rendered via mkdocs
- **Rhiza**: `.rhiza/` holds template-managed config — do not edit it directly

## Workflow

1. Run `make install` to set up the environment.
2. Write code in `src/pyhrp/` and mirrored tests in `tests/pyhrp/`.
3. Run `make test` to verify changes at 100% coverage.
4. Run `make typecheck` and `make fmt` before committing.
