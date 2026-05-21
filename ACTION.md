# pyhrp — Action Plan

> Derived from ANALYSIS.md scores. Items ordered by gap from 10 (largest first).
> Current average: **7.0 / 10** across 18 subcategories.

---

## Critical (score ≤ 3)

### Performance & stress coverage — 1 / 10

**Goal:** Verify algorithmic complexity does not regress as the codebase evolves.

1. Add `pytest-benchmark` to the dev dependency group.
2. Write a `tests/test_benchmark.py` module using the `@pytest.mark.stress` marker (already registered in `pytest.ini`):
   - Benchmark `hrp()` on the existing 20-asset dataset.
   - Benchmark `hrp()` on a synthetic 100-asset and 200-asset universe.
   - Benchmark `build_tree()` separately to isolate clustering cost from weight computation.
3. Run benchmarks in CI on a schedule (weekly) rather than on every push, to avoid slowing the main matrix.
4. Store the baseline JSON artifact and fail the job if runtime regresses by more than 20 %.

---

### Contributing & changelog — 2 / 10

**Goal:** Surface contributor guidance at the repo root and maintain a persistent change history.

1. Add a `CONTRIBUTING.md` at the repo root (symlink or copy from `.rhiza/CONTRIBUTING.md`, then customise for pyhrp-specific conventions: branch naming, commit format, how to run `make test` and `make fmt`).
2. Add a `CHANGELOG.md` maintained in [Keep a Changelog](https://keepachangelog.com) format. Seed it from existing `git-cliff` output (`uvx git-cliff --output CHANGELOG.md`). Update it as part of the release checklist.
3. Add `CODE_OF_CONDUCT.md` at the repo root (copy from `.rhiza/`).
4. Update `pyproject.toml` to reference the changelog URL under `[project.urls]`.

---

### Coverage gating — 3 / 10

**Goal:** Make CI reject PRs that delete tests or reduce coverage.

1. Add a `[tool.coverage.report]` section to `pyproject.toml` (or a `setup.cfg` / `.coveragerc` equivalent) with `fail_under = 90`.
2. Pass `--cov-fail-under=90` to the pytest invocation in `make test` (or configure it in `pytest.ini` via `addopts`).
3. Extend coverage upload to all OS/Python combinations in the matrix, not just ubuntu-latest / 3.12 — or at minimum gate the build on the ubuntu-latest result with an explicit threshold check step.

---

### Edge case & numerical robustness — 3 / 10

**Goal:** Confirm HRP is numerically stable under degenerate inputs.

1. Install `hypothesis` and `hypothesis[numpy]` in the dev dependency group.
2. Write a `tests/test_property.py` module using the `@pytest.mark.property` marker (already registered) with Hypothesis strategies:
   - Random valid correlation matrices (positive semi-definite, diagonal 1) for `build_tree()`.
   - Random covariance matrices for `risk_parity()`.
   - Assert weights sum to 1.0 and all weights are in `[0, 1]`.
3. Add explicit unit tests for degenerate inputs:
   - Single-asset universe → portfolio weight is 1.0.
   - Two-asset universe → verified against hand-calculated risk-parity weights.
   - Near-singular covariance matrix (assets with identical return series).
   - Tree of depth 1 (two assets, no further bisection).

---

### Community health files — 3 / 10

*Covered by the Contributing & changelog action above.*

---

## High priority (score 4–6)

### README quality — 4 / 10

**Goal:** A first-time visitor should understand what HRP is and when to use pyhrp without reading the paper.

1. Add a **Motivation** section (3–5 sentences): what HRP solves vs. mean-variance optimisation, why correlation-based clustering improves diversification.
2. Add a **Method comparison** table: Ward vs. single vs. average linkage — when each is appropriate and what the bisection flag changes.
3. Add a **Interpreting results** paragraph: explain that `hrp()` returns a `Cluster` node (not a weight series), how to navigate `root.left` / `root.right`, and how to extract weights.
4. Replace the static `demo.png` with a generation script (`book/marimo/demo.py` or similar) so the image stays in sync with the code.

---

### Version constraint hygiene — 5 / 10

**Goal:** Prevent silent breakage from upstream releases.

1. **`plotly<6.6`**: Replace the hard upper bound with a tested upper bound — once plotly 6.6 is released and verified, bump to `<6.7`. Add a note in the release checklist to re-test plotly compatibility on each minor release.
2. **`polars>=1.40.1`**: Tighten to `>=1.40.1, <2` to guard against a potential Polars 2.0 API break.
3. **`cvx-linalg>=0.5.1`**: Add an upper bound once the API is considered stable, or pin to a minor version.
4. Set up Renovate (already referenced in the repo) to open PRs when these bounds need updating, with auto-merge enabled for patch-level bumps that pass CI.

---

### API design — 6 / 10

**Goal:** Prevent accidental import of private helpers and fix the `Any` type gap.

1. Add `__all__` to every module:
   - `hrp.py`: `["hrp", "build_tree", "Dendrogram"]`
   - `algos.py`: `["risk_parity", "one_over_n"]`
   - `cluster.py`: `["Portfolio", "Cluster"]`
   - `treelib.py`: `["Node"]`
2. Fix `one_over_n(dendrogram: Any)` in `algos.py:92` — replace `Any` with `Dendrogram` (import from `hrp.py`) or introduce a `Protocol` if a circular import would result.
3. Promote `bisect_tree` and `get_linkage` from nested functions inside `build_tree` to module-level private functions (`_bisect_tree`, `_get_linkage`) so they can be tested and read in isolation.
4. Add a comment to `cluster.py:142–160` explaining why `leaves` is overridden (left-to-right ordering required for the HRP algorithm, which differs from the default post-order traversal in `treelib.Node`).

---

### Type safety — 6 / 10

**Goal:** Eliminate structural `# type: ignore` pragmas by fixing the root cause.

1. Make `treelib.Node` generic: `class Node(Generic[T])` where `T = NodeValue`. This removes the need for type suppression in `cluster.py:155–159` where `Cluster` assigns typed leaf lists.
2. Update `Cluster` to inherit from `Node[int]` and `Portfolio` / `Cluster` leaf-typing will resolve naturally.
3. Re-run `ty` after the change and fix any newly surfaced errors in `hrp.py` and `algos.py`. Target: ≤ 4 pragmas (only legitimate suppression of external library stubs).
4. Remove `ignore_missing_imports = true` from the `[tool.mypy]` config once stubs for `scipy`, `plotly`, and `cvx-linalg` are either available or explicitly stubbed locally.

---

### Type checking — 6 / 10

**Goal:** One authoritative type checker, properly configured for the declared Python version.

1. **Fix `ruff.toml`**: Change `target-version = "py311"` to `target-version = "py312"` to match `pyproject.toml`. This unlocks `UP` rules for Python 3.12 syntax (`type` statement, etc.).
2. **Resolve the `ty` / `mypy` overlap**: Pick one as the CI gate. If `ty` is the primary checker, remove or demote `mypy` to a local-only advisory tool. Document the decision in a comment in `pyproject.toml`.
3. **Remove `ignore_missing_imports = true`** once type safety action above is complete.

---

## Medium priority (score 7–8)

### Set minimalism — 7 / 10

**Goal:** Reduce the supply-chain surface to the minimum required.

1. Evaluate removing `cvx-linalg` entirely. The sole usage is `a_norm(w, c)` in `cluster.py:59`. Replace with `float(np.sqrt(w @ c @ w))` — three tokens, zero extra dependency.
2. If `cvx-linalg` is authored by the same maintainer and intended as a companion library, document that relationship in the README so users understand the design decision.

---

### Standard files — 7 / 10

*Covered by the Contributing & changelog action above.*

---

### Docstring coverage — 8 / 10

**Goal:** Ensure overrides and non-obvious behaviours are documented inline.

1. Add a one-line comment to `cluster.py:142` explaining the `leaves` override (as noted in API design action).
2. Add docstrings to `bisect_tree` and `get_linkage` once they are promoted to module-level private functions.
3. Ensure `__all__` additions (from API design action) do not suppress interrogate's coverage check on newly public symbols.

---

### Test breadth & module coverage — 8 / 10

**Goal:** Test private helpers directly rather than only through integration paths.

1. Add direct unit tests for `_compute_cov` and `_compute_corr` in a dedicated `test_helpers.py`:
   - Verify covariance matrix is symmetric.
   - Verify diagonal of correlation matrix is all 1.0.
   - Verify column names are preserved from the input DataFrame.
2. Add assertions to `test_notebooks.py`: after `runpy.run_path`, inspect the returned namespace for at least one expected output variable (e.g., the weights dict or a `Cluster` object).

---

### Test quality — 8 / 10

*Partially covered by the edge-case and breadth actions above. No additional standalone action required.*

---

### API reference — 8 / 10

*Covered by the API design and docstring actions above.*

---

### Interactive notebooks — 8 / 10

*No action required beyond keeping notebooks in sync with the API.*

---

## Summary table

| Subcategory | Current | Target | Action |
|-------------|---------|--------|--------|
| Performance & stress coverage | 1 | 7 | Add `pytest-benchmark`, stress tests |
| Contributing & changelog | 2 | 8 | Add `CHANGELOG.md`, root `CONTRIBUTING.md` |
| Coverage gating | 3 | 8 | `fail_under = 90` in pytest / CI |
| Edge case & numerical robustness | 3 | 8 | Hypothesis property tests, degenerate inputs |
| Community health files | 3 | 8 | Root `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md` |
| README quality | 4 | 8 | Motivation, method comparison, result guide |
| Version constraint hygiene | 5 | 8 | Tighten polars, fix plotly bound, Renovate |
| API design | 6 | 9 | `__all__`, fix `Any`, promote nested functions |
| Type safety | 6 | 9 | Generic `Node[T]`, reduce type: ignore pragmas |
| Type checking | 6 | 9 | Fix ruff target-version, single checker |
| Set minimalism | 7 | 9 | Inline `a_norm`, remove `cvx-linalg` |
| Standard files | 7 | 9 | Root community files |
| Docstring coverage | 8 | 9 | Document `leaves` override, promote helpers |
| Test breadth & module coverage | 8 | 9 | Direct tests for `_compute_cov`, `_compute_corr` |
| Test quality | 8 | 9 | Assertions in `test_notebooks.py` |
| API reference | 8 | 9 | Covered by API design actions |
| Pipeline completeness | 9 | 10 | Minor: distinguish rhiza vs. project workflows |
| Security posture | 10 | 10 | Nothing to do |
