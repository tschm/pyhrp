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
- Explicit unit tests for single-asset, two-asset closed-form, near-singular, and empty-input cases.

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

**PR #671** (`copilot/tighten-dependency-version-constraints`) + **PR #672**:

- `plotly<6.6` → `plotly>=5,<7` (broadened lower bound, explicit upper cap).
- `polars>=1.40.1` → `polars>=1.40.1,<2` (guarded against Polars 2.0 API break).
- `cvx-linalg>=0.5.1` → `cvx-linalg>=0.5.1,<1` (pinned to stable major version).
- `numpy>=2.3` → `numpy>=2.3,<3` (explicit upper bound, symmetric with other runtime packages).

---

### API design — 6 / 10 → **done**

**PR #667** (`copilot/refactor-add-all-fix-type-hint`):

- Added `__all__` to every module (`hrp.py`, `algos.py`, `cluster.py`, `treelib.py`).
- Fixed `one_over_n(dendrogram: Any)` → `one_over_n(dendrogram: Dendrogram)`.
- Promoted `bisect_tree` and `get_linkage` to module-level private functions
  `_bisect_tree` and `_get_linkage`, each with a docstring.

---

### Type safety — 6 / 10 → **done**

**PR #665** (`copilot/refactor-treelib-node-generic`) + **PR #672**:

- Made `treelib.Node` generic: `class Node[T: NodeValue]`.
- `Cluster` inherits from `Node[int]`; all `# type: ignore` pragmas removed — **0 remaining**.
- Removed `**kwargs: Any` from `Cluster.__init__` — **0 `Any` annotations** anywhere in the codebase.

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

**PR #667** + **PR #673** (`copilot/plan-docstring-examples`):

- `_bisect_tree` and `_get_linkage` received docstrings when promoted.
- Added `Examples:` blocks to all four primary public functions (`hrp`, `build_tree`, `risk_parity`,
  `one_over_n`).

---

### Test breadth & module coverage — 8 / 10 → **done**

**PR #670** (`copilot/add-direct-unit-tests-compute-cov-corr`):

- Added `tests/test_helpers.py` (2 tests): directly tests `_compute_cov` (symmetry) and
  `_compute_corr` (unit diagonal, column name preservation).
- Added 4 assertions to `test_notebooks.py`: verifies expected output variables in notebook
  namespaces rather than just checking for no exception.

---

### Performance & stress coverage — 1 / 10 → **done**

**PR #654** (`copilot/add-performance-stress-benchmarks`) + **PR #674** (`copilot/plan-test-improvements`):

- Added `pytest-benchmark` to the dev dependency group; `uv.lock` updated.
- `tests/test_benchmark.py` (5 tests) benchmarks 20-, 100-, and 200-asset universes and `_bisect_tree`
  in isolation. Each benchmark captures the return value and asserts weight validity.
- `tests/stress/test_stress.py` (2 tests) exercises 500- and 1000-asset universes (`make stress`).
- Weekly GitHub Actions workflow with baseline JSON artifact and 20 % regression gate.
- `max_examples` raised from 50 → 200 in Hypothesis property tests.
- `sum(weights) ≈ 1.0` and `all(0 ≤ w ≤ 1)` assertions added to `test_notebooks.py`.
- Two additional tests in `test_cluster.py` covering `TypeError` branches in `Cluster.leaves`.
- **Line coverage: 100 %.**

---

### Tooling completeness — 9 / 10 → **done**

**PR #672** (`copilot/plan-trivial-fixes`):

- Added `[lint.pycodestyle] max-doc-length = 120` to `ruff.toml`, matching `line-length = 120`.

---

### Pipeline completeness — 9 / 10 → **done**

Coverage gate (`--cov-fail-under=90`) runs on every matrix leg (all OS × Python combinations) via
`make test` in `.rhiza/make.d/test.mk`. The gate is already cross-platform.

---

### CHANGELOG automation — 9 / 10 → **done**

**PR #675** (`copilot/plan-changelog-automation`):

- Added `update-changelog` job to `rhiza_release.yml` that runs after `finalise-release`.
- Checks out `main`, regenerates `CHANGELOG.md` via `uvx git-cliff`, and pushes the commit back to
  `main` on every release tag.

---

### Set minimalism — deliberate design decision

**Not actioned.** `cvx-linalg` is retained as an intentional companion dependency within the `cvx`
ecosystem. This caps the Dependencies section at **9 / 10**.

---

## Summary table

| Subcategory | Original | Final | Status |
|-------------|----------|-------|--------|
| Performance & stress coverage | 1 | 10 | ✅ PRs #654, #674 |
| Contributing & changelog | 2 | 10 | ✅ PRs #666, #675 |
| Coverage gating | 3 | 10 | ✅ PR #664 |
| Edge case & numerical robustness | 3 | 10 | ✅ PR #668 |
| Community health files | 3 | 10 | ✅ PR #666 |
| README quality | 4 | 10 | ✅ PR #662 |
| Version constraint hygiene | 5 | 10 | ✅ PRs #671, #672 |
| API design | 6 | 10 | ✅ PR #667 |
| Type safety | 6 | 10 | ✅ PRs #665, #672 |
| Type checking | 6 | 10 | ✅ PR #669 |
| Set minimalism | 7 | 7 | — intentional (cvx-linalg retained) |
| Standard files | 7 | 10 | ✅ PR #666 |
| Docstring coverage | 8 | 10 | ✅ PRs #667, #673 |
| Test breadth & module coverage | 8 | 10 | ✅ PRs #670, #674 |
| Test quality | 8 | 10 | ✅ PRs #670, #674 |
| API reference | 8 | 10 | ✅ PRs #667, #673 |
| Pipeline completeness | 9 | 10 | ✅ already handled by `make test` |
| Security posture | 10 | 10 | ✅ nothing to do |
