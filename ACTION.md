# pyhrp — Action Plan

> Derived from ANALYSIS.md scores. Items ordered by gap from 10 (largest first).
> Original average: **7.0 / 10** across 18 subcategories.
> Last updated: 2026-05-21

---

## ✅ Completed (merged to `main`)

### Contributing & changelog — 2 / 10 → **done**

**PR #666** (`copilot/add-changelog-contributing-code-of-conduct`):

- Added `CHANGELOG.md` seeded from `git-cliff` output (Keep a Changelog format).
- Added `CONTRIBUTING.md` at repo root.
- Added `CODE_OF_CONDUCT.md` at repo root.

---

### Coverage gating — 3 / 10 → **done**

**PR #664** (`copilot/ci-enforce-minimum-coverage-threshold`):

- Added `[tool.coverage.report] fail_under = 90` to `pyproject.toml`.
- Current measured coverage: **100 %** across all modules.

---

### Community health files — 3 / 10 → **done**

*Covered by PR #666 above.*

---

### Edge case & numerical robustness — 3 / 10 → **done**

**PR #668** (`copilot/add-property-based-tests`):

- `tests/test_property.py` (6 tests) using Hypothesis: random correlation matrices for `build_tree()`,
  random covariance matrices for `risk_parity()`, weights-sum-to-1 and `[0, 1]` assertions.
- Explicit unit tests for single-asset, two-asset, near-singular, and empty-input cases.

---

### README quality — 4 / 10 → **done**

**PR #662** (`copilot/expand-readme-with-motivation`):

- Added **Motivation** section (HRP vs. mean-variance, clustering-based diversification).
- Added method comparison table (Ward / single / average linkage + bisection flag).
- Added result-interpretation paragraph (`Cluster` node, `root.left` / `root.right`, weight extraction).
- Added `demo.py` generation script (`book/marimo/demo.py`) so `demo.png` stays in sync.
- Added `kaleido` as a dev dependency for static image export.

---

### Version constraint hygiene — 5 / 10 → **done**

**PR #671** (`copilot/tighten-dependency-version-constraints`):

- `plotly<6.6` → `plotly>=5,<7` (broadened lower bound, explicit upper cap).
- `polars>=1.40.1` → `polars>=1.40.1,<2` (guarded against Polars 2.0 API break).
- `cvx-linalg>=0.5.1` → `cvx-linalg>=0.5.1,<1` (pinned to stable major version).

---

### API design — 6 / 10 → **done**

**PR #667** (`copilot/refactor-add-all-fix-type-hint`):

- Added `__all__` to every module (`hrp.py`, `algos.py`, `cluster.py`, `treelib.py`).
- Fixed `one_over_n(dendrogram: Any)` → `one_over_n(dendrogram: Dendrogram)`.
- Promoted `bisect_tree` and `get_linkage` to module-level private functions
  `_bisect_tree` and `_get_linkage`, each with a docstring.

---

### Type safety — 6 / 10 → **done**

**PR #665** (`copilot/refactor-treelib-node-generic`):

- Made `treelib.Node` generic: `class Node[T: NodeValue]`.
- `Cluster` inherits from `Node[int]`; all `# type: ignore` pragmas removed — **0 remaining**.

---

### Type checking — 6 / 10 → **done**

**PR #669** (`copilot/chore-fix-ruff-target-version`):

- Fixed `ruff.toml`: `target-version = "py311"` → `"py312"` to match `pyproject.toml`.
- Consolidated to `ty` as the single CI type checker; `mypy` configuration removed.

---

### Standard files — 7 / 10 → **done**

*Covered by PR #666 above.*

---

### Docstring coverage — 8 / 10 → **done**

*Covered by PR #667: `_bisect_tree` and `_get_linkage` received docstrings when promoted.*

---

### Test breadth & module coverage — 8 / 10 → **done**

**PR #670** (`copilot/add-direct-unit-tests-compute-cov-corr`):

- Added `tests/test_helpers.py` (2 tests): directly tests `_compute_cov` (symmetry) and
  `_compute_corr` (unit diagonal, column name preservation).
- Added 4 assertions to `test_notebooks.py`: verifies expected output variables in notebook
  namespaces rather than just checking for no exception.

---

### Performance & stress coverage — 1 / 10 → **done**

**PR #654** (`copilot/add-performance-stress-benchmarks`):

- Added `pytest-benchmark` to the dev dependency group; `uv.lock` updated.
- `tests/test_benchmark.py` (5 tests) with benchmarks for 20-, 100-, and 200-asset universes and
  `_bisect_tree` in isolation.
- Weekly GitHub Actions workflow with baseline JSON artifact and 20 % regression gate.

---

## ⬜ Not Yet Started

### Set minimalism — 7 / 10

**Goal:** Reduce the supply-chain surface to the minimum required.

1. Evaluate removing `cvx-linalg`. The sole usage is `a_norm(w, c)` in `cluster.py:61`. Replace with
   `float(np.sqrt(w @ c @ w))` — three tokens, zero extra dependency.
2. If `cvx-linalg` is authored by the same maintainer and intended as a companion library, document
   that relationship in the README so users understand the design decision.

---

### Pipeline completeness — 9 / 10

**Goal:** Cover all platforms in the coverage gate.

1. Coverage is currently uploaded only from ubuntu-latest / Python 3.12. A platform-specific failure
   (e.g., a Windows path bug) would not be caught by the coverage gate. Consider uploading coverage
   from the full matrix and merging reports, or at minimum add a smoke-test gate on all platforms.

---

## Summary table

| Subcategory | Original | Current | Target | Status |
|-------------|----------|---------|--------|--------|
| Performance & stress coverage | 1 | 7 | 7 | ✅ Merged #654 |
| Contributing & changelog | 2 | 9 | 8 | ✅ Merged #666 |
| Coverage gating | 3 | 9 | 8 | ✅ Merged #664 |
| Edge case & numerical robustness | 3 | 7 | 8 | ✅ Merged #668 |
| Community health files | 3 | 9 | 8 | ✅ Merged #666 |
| README quality | 4 | 8 | 8 | ✅ Merged #662 |
| Version constraint hygiene | 5 | 8 | 8 | ✅ Merged #671 |
| API design | 6 | 9 | 9 | ✅ Merged #667 |
| Type safety | 6 | 10 | 9 | ✅ Merged #665 |
| Type checking | 6 | 9 | 9 | ✅ Merged #669 |
| Set minimalism | 7 | 7 | 9 | ⬜ Not started |
| Standard files | 7 | 9 | 9 | ✅ Merged #666 |
| Docstring coverage | 8 | 9 | 9 | ✅ Merged #667 |
| Test breadth & module coverage | 8 | 9 | 9 | ✅ Merged #670 |
| Test quality | 8 | 9 | 9 | ✅ Merged #670 |
| API reference | 8 | 9 | 9 | ✅ Covered by #667 |
| Pipeline completeness | 9 | 9 | 10 | ⬜ Minor |
| Security posture | 10 | 10 | 10 | ✅ Nothing to do |
