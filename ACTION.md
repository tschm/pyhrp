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

### API design — 6 / 10 → **done**

**PR #667** (`copilot/refactor-add-all-fix-type-hint`):

- Added `__all__` to every module (`hrp.py`, `algos.py`, `cluster.py`, `treelib.py`).
- Fixed `one_over_n(dendrogram: Any)` → `one_over_n(dendrogram: Dendrogram)`.
- Promoted `bisect_tree` and `get_linkage` to module-level private functions
  `_bisect_tree` and `_get_linkage`, each with a one-line docstring.

---

### Type safety — 6 / 10 → **done**

**PR #665** (`copilot/refactor-treelib-node-generic`):

- Made `treelib.Node` generic: `class Node(Generic[T])`.
- `Cluster` inherits from `Node[int]`; `# type: ignore` pragmas removed from core modules.
- Replaced `ignore_missing_imports = true` with targeted `[[tool.mypy.overrides]]` entries
  for `cvx.*`, `plotly.*`, and `scipy.*`.

---

### Standard files — 7 / 10 → **done**

*Covered by PR #666 above.*

---

### Docstring coverage — 8 / 10 → **done**

*Covered by PR #667: `_bisect_tree` and `_get_linkage` received docstrings when promoted.*

---

## 🔄 In Progress (open PRs, not yet merged)

### Performance & stress coverage — 1 / 10

**Branch:** `copilot/add-performance-stress-benchmarks`

Done so far:
- Added `pytest-benchmark` to the dev dependency group; `uv.lock` updated.
- `tests/test_benchmark.py` with benchmarks for 20-, 100-, and 200-asset universes and
  `_bisect_tree` in isolation.
- Weekly GitHub Actions workflow with baseline JSON artifact and 20 % regression gate.

Remaining:
- Merge the PR and verify the workflow runs successfully on the scheduled trigger.

---

### Edge case & numerical robustness — 3 / 10

**Branch:** `origin/copilot/add-property-based-tests`

Done so far:
- `tests/test_property.py` using Hypothesis: random correlation matrices for `build_tree()`,
  random covariance matrices for `risk_parity()`, weights-sum-to-1 and `[0, 1]` assertions.
- Explicit unit tests for single-asset, two-asset, near-singular, and empty-input cases.
- `zip(..., strict=True)` tightened in generators.

Remaining:
- Merge the PR.

---

### README quality — 4 / 10

**Branch:** `copilot/expand-readme-with-motivation`

Done so far:
- Added **Motivation** section (HRP vs. mean-variance, clustering-based diversification).
- Added method comparison table (Ward / single / average linkage + bisection flag).
- Added result-interpretation paragraph (`Cluster` node, `root.left` / `root.right`, weight extraction).
- Added `demo.py` generation script (`book/marimo/demo.py`) so `demo.png` stays in sync.
- Added `kaleido` as a dev dependency for static image export.

Remaining:
- Merge the PR.

---

## ⬜ Not Yet Started

### Version constraint hygiene — 5 / 10

**Goal:** Prevent silent breakage from upstream releases.

1. **`plotly<6.6`**: Replace the hard upper bound with a tested upper bound — once plotly 6.6 is
   released and verified, bump to `<6.7`. Add a note in the release checklist to re-test plotly
   compatibility on each minor release.
2. **`polars>=1.40.1`**: Tighten to `>=1.40.1, <2` to guard against a potential Polars 2.0 API break.
3. **`cvx-linalg>=0.5.1`**: Add an upper bound once the API is considered stable, or pin to a
   minor version.
4. Set up Renovate (already referenced in the repo) to open PRs when these bounds need updating,
   with auto-merge enabled for patch-level bumps that pass CI.

---

### Type checking — 6 / 10 (partially done)

**Goal:** One authoritative type checker, properly configured for the declared Python version.

Already done: targeted mypy overrides in place of `ignore_missing_imports = true`.

Remaining:
1. **Fix `ruff.toml`**: Change `target-version = "py311"` to `"py312"` to match `pyproject.toml`.
2. **Resolve the `ty` / `mypy` overlap**: Pick one as the CI gate. If `ty` is primary, remove or
   demote `mypy` to advisory-only. Document the decision in a comment in `pyproject.toml`.

---

### Set minimalism — 7 / 10

**Goal:** Reduce the supply-chain surface to the minimum required.

1. Evaluate removing `cvx-linalg`. The sole usage is `a_norm(w, c)` in `cluster.py`. Replace with
   `float(np.sqrt(w @ c @ w))` — three tokens, zero extra dependency.
2. If `cvx-linalg` is authored by the same maintainer and intended as a companion library, document
   that relationship in the README so users understand the design decision.

---

### Test breadth & module coverage — 8 / 10

**Goal:** Test private helpers directly rather than only through integration paths.

1. Add direct unit tests for `_compute_cov` and `_compute_corr` in a dedicated `test_helpers.py`:
   - Verify covariance matrix is symmetric.
   - Verify diagonal of correlation matrix is all 1.0.
   - Verify column names are preserved from the input DataFrame.
2. Add assertions to `test_notebooks.py`: after `runpy.run_path`, inspect the returned namespace
   for at least one expected output variable (e.g., the weights dict or a `Cluster` object).

---

## Summary table

| Subcategory | Original | Current | Target | Status |
|-------------|----------|---------|--------|--------|
| Performance & stress coverage | 1 | 5 | 7 | 🔄 PR open |
| Contributing & changelog | 2 | 8 | 8 | ✅ Merged #666 |
| Coverage gating | 3 | 9 | 8 | ✅ Merged #664 |
| Edge case & numerical robustness | 3 | 6 | 8 | 🔄 PR open |
| Community health files | 3 | 8 | 8 | ✅ Merged #666 |
| README quality | 4 | 6 | 8 | 🔄 PR open |
| Version constraint hygiene | 5 | 5 | 8 | ⬜ Not started |
| API design | 6 | 9 | 9 | ✅ Merged #667 |
| Type safety | 6 | 9 | 9 | ✅ Merged #665 |
| Type checking | 6 | 7 | 9 | ⬜ Ruff + checker TBD |
| Set minimalism | 7 | 7 | 9 | ⬜ Not started |
| Standard files | 7 | 9 | 9 | ✅ Merged #666 |
| Docstring coverage | 8 | 9 | 9 | ✅ Merged #667 |
| Test breadth & module coverage | 8 | 8 | 9 | ⬜ Not started |
| Test quality | 8 | 8 | 9 | ⬜ Not started |
| API reference | 8 | 9 | 9 | ✅ Covered by #667 |
| Pipeline completeness | 9 | 9 | 10 | ⬜ Minor |
| Security posture | 10 | 10 | 10 | ✅ Nothing to do |
