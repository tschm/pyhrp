# pyhrp — Quality Analysis

> Analysis date: 2026-05-21 · Version: 2.0.0

---

## Scorecard

| Section | Score |
|---------|-------|
| 1. Source Code | **7 / 10** |
| 2. Tests | **6 / 10** |
| 3. Dependencies | **7 / 10** |
| 4. CI/CD | **8 / 10** |
| 5. Tooling | **8 / 10** |
| 6. Documentation | **5 / 10** |
| 7. Project Hygiene | **7 / 10** |

---

## 1. Source Code — 7 / 10

| Subcategory | Score |
|-------------|-------|
| Structure & separation of concerns | 9 / 10 |
| Algorithm correctness | 9 / 10 |
| Docstring coverage | 8 / 10 |
| API design | 6 / 10 |
| Type safety | 6 / 10 |

### Structure

Five focused modules totalling **687 lines** under `src/pyhrp/`:

| Module | Lines | Role |
|--------|-------|------|
| `hrp.py` | 265 | HRP algorithm, `Dendrogram`, `build_tree`, `hrp` entry point |
| `algos.py` | 124 | `risk_parity`, `one_over_n` recursive tree traversals |
| `cluster.py` | 160 | `Portfolio` and `Cluster` data structures |
| `treelib.py` | 129 | Generic binary `Node` base class |
| `__init__.py` | 9 | Dynamic version via `importlib.metadata` |

### Strengths

- **Separation of concerns is clean.** The tree data structure (`treelib`), domain model (`cluster`), algorithms (`algos`), and orchestration (`hrp`) are clearly separated. Adding a new allocation algorithm requires touching only `algos.py`.
- **Public API surface is small and intentional.** Six public symbols cover the full use-case: `hrp`, `build_tree`, `Dendrogram`, `risk_parity`, `one_over_n`, `Cluster`/`Portfolio`.
- **Type annotations are complete.** Every public function and method is annotated, using the modern `|` union syntax and `from __future__ import annotations` throughout. The `ty` type checker runs in CI.
- **Docstrings follow Google style consistently** and are enforced by `interrogate` in pre-commit and CI. Functions like `build_tree` have 23-line docstrings covering Args, Returns, and Raises.
- **Algorithm is faithful to the source.** The risk-parity formula (`alpha_left = v_right / (v_left + v_right)`) matches Lopez de Prado's original paper. Bisection and Ward-linkage reconstruction in `hrp.py` are non-trivial and correctly implemented.

### Weaknesses

- **3 `# type: ignore` pragmas** across the codebase (`hrp.py`: 1, `cluster.py`: 2). Both cluster.py pragmas suppress inference on leaf assignment (`cluster.py:158–159`) because the generic `Node` type is not parameterised — pointing to a design gap where `treelib.Node` is untyped at the generic level.
- **`one_over_n` accepts `dendrogram: Any`** (`algos.py:92`) instead of `Dendrogram`. This is inconsistent with the rest of the API and silently accepts bad inputs.
- **`bisect_tree` and `get_linkage` are nested functions** inside `hrp.py`. They are long (25+ lines each), non-trivial, and untestable in isolation. Promoting them to module-level private functions would improve testability and readability.
- **No explicit `__all__`** in any module, so `from pyhrp.hrp import *` exposes private helpers like `_compute_cov` and `_compute_corr`.

---

## 2. Tests — 6 / 10

| Subcategory | Score |
|-------------|-------|
| Test quality (assertions, fixtures) | 8 / 10 |
| Breadth & module coverage | 8 / 10 |
| Edge case & numerical robustness | 3 / 10 |
| Performance & stress coverage | 1 / 10 |

### Overview

**59 test functions across 12 files, 1 087 lines.**

| File | Tests | Focus |
|------|-------|-------|
| `test_dendrogram_extras.py` | 32 | Dendrogram validation, distance matrix, edge cases |
| `test_hrp.py` | 4 | Linkage matrix, bisection, node ordering |
| `test_treelib.py` | 7 | Full `Node` class coverage |
| `test_cluster.py` | 7 | Risk parity, type errors |
| `test_hrp_function.py` | 2 | Weights vs. reference CSV files |
| `test_one_over_n.py` | 2 | 1/N algorithm |
| `test_large_1n.py` | 1 | 1/N on 20-stock real data |
| `test_portfolio.py` | 2 | Portfolio creation, plotting, variance |
| `test_node.py` | 1 | `Cluster` basics |
| `test_notebooks.py` | 1 | Marimo notebook execution |

### Strengths

- **Reference-data validation.** `tests/resources/` holds CSV ground-truth files (`weights_hrp.csv`, `weights_marcos.csv`, `links.csv`) against which computed outputs are compared. This is the right approach for a numerical library.
- **`pytest.approx()` used throughout** for floating-point comparisons — no raw `==` on floats.
- **Parametrised over linkage methods.** `test_invariant_order`, `test_build_tree_with_different_methods`, and `test_hrp_with_different_methods` each run against `single`, `ward`, `average`, and `complete` methods.
- **Type-error tests are explicit.** Four dedicated tests verify that passing non-`Cluster` objects to `risk_parity` raises `TypeError`. This catches API misuse early.
- **Notebook execution is tested.** `test_notebooks.py` runs both marimo notebooks via `runpy.run_path`, preventing silent breakage of the interactive documentation.
- **Session-scoped fixtures** in `conftest.py` load the 321-row stock-price CSV once per session and compute returns, covariance, and correlation matrices — no redundant I/O in individual tests.

### Weaknesses

- **`stress` and `property` pytest markers are defined but unused.** `pytest.ini` registers them; no test uses them. Property-based testing (e.g., via Hypothesis) would meaningfully increase confidence in numerical stability for random correlation matrices.
- **No numerical edge-case tests.** Missing: near-singular covariance matrices, assets with identical return series, portfolios with a single asset, trees of depth 1.
- **`test_notebooks.py` has no assertions.** It only verifies that the notebooks do not raise. A broken calculation that produces a wrong plot would pass.
- **No performance/regression test.** There is no test tracking compute time or memory for large inputs (e.g., 200-asset universe), making it impossible to detect regressions in algorithmic complexity.
- **Private helpers `_compute_cov` and `_compute_corr` are tested only indirectly** through the public `hrp()` function. Dedicated unit tests would isolate numerical correctness.

---

## 3. Dependencies — 7 / 10

| Subcategory | Score |
|-------------|-------|
| Security (pip-audit, deptry, lock file) | 9 / 10 |
| Set minimalism | 7 / 10 |
| Version constraint hygiene | 5 / 10 |

### Runtime (5 packages)

| Package | Constraint | Usage |
|---------|-----------|-------|
| `numpy` | `>=2.3` | Array operations, linear algebra throughout |
| `scipy` | `>=1.14.1` | Hierarchical clustering (`linkage`, `leaves_list`) |
| `polars` | `>=1.40.1` | DataFrame I/O and return computation |
| `cvx-linalg` | `>=0.5.1` | `a_norm()` for portfolio variance in `cluster.py` |
| `plotly` | `<6.6` | Interactive dendrogram and weight plots |

### Strengths

- **`uv.lock` is committed**, ensuring fully reproducible builds.
- **Dependency set is minimal and purposeful.** Five packages, each with a clear reason.
- **OIDC PyPI publishing** means no stored credentials in CI secrets.
- **`pip-audit`** runs in every CI build (`make security`), catching known CVEs.
- **`deptry`** verifies that declared deps match actual imports — drift is caught automatically.

### Weaknesses

- **`cvx-linalg` is a heavy dependency for a single utility function.** `a_norm()` computes `x @ A @ x`. This is three lines of numpy. Depending on an external package for this introduces an unnecessary supply-chain surface.
- **`plotly<6.6` upper-bound will require manual bumping.** Unlike a lower-bound, this will silently break when plotly 6.6 is released unless the constraint is regularly reviewed. No Dependabot or Renovate configuration is visible at the repo root level (Renovate is managed via rhiza template).
- **`polars>=1.40.1` ties the library to a rapidly-evolving API.** Polars has broken APIs at minor versions before. A tighter constraint (e.g., `>=1.40, <2`) may reduce unexpected breakage for downstream users.

---

## 4. CI/CD — 8 / 10

| Subcategory | Score |
|-------------|-------|
| Security posture (SBOM, SLSA, OIDC) | 10 / 10 |
| Pipeline completeness (matrix, jobs) | 9 / 10 |
| Coverage gating | 3 / 10 |

### Workflows

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `rhiza_ci.yml` | push / PR | generate-matrix, test, typecheck, deptry, pre-commit, docs-coverage, validation, security, license |
| `rhiza_release.yml` | `v*` tag | validate tag, build, SBOM, draft release, PyPI (OIDC), finalize |
| `rhiza_marimo.yml` | template-managed | Marimo notebook build |
| `rhiza_book.yml` | template-managed | MkDocs site build |
| `rhiza_sync.yml` | template-managed | Rhiza template sync PRs |
| `rhiza_weekly.yml` | cron | Maintenance |

### Strengths

- **Multi-OS, multi-Python matrix.** Tests run on ubuntu-latest, macos-latest, and windows-latest across multiple Python versions.
- **Security scanning is first-class.** `make security` (Bandit + pip-audit) runs on every push. Bandit is also a pre-commit hook.
- **SBOM generated and attested at every release** (CycloneDX JSON + XML, with GitHub Attestation for public repos). This is well above industry baseline.
- **SLSA provenance attestations** generated for all release artifacts.
- **Version is validated before release.** The tag must match `pyproject.toml` version — no stale version numbers can be published.
- **OIDC for PyPI** — no stored credentials anywhere in the pipeline.

### Weaknesses

- **Coverage upload is only on ubuntu-latest / Python 3.12.** A platform-specific failure (e.g., a Windows path bug) would not be caught by the coverage gate.
- **No coverage threshold gate.** Coverage is measured and uploaded but there is no minimum threshold; a PR that deletes tests would still pass CI.
- **`test_notebooks.py` is in the test matrix** but provides no numerical assertions (see Tests section). The CI check is weaker than it appears.
- **All workflow filenames are prefixed `rhiza_`**, which makes it harder to distinguish project-specific workflows from template-managed ones at a glance.

---

## 5. Tooling — 8 / 10

| Subcategory | Score |
|-------------|-------|
| Linting & formatting (ruff) | 9 / 10 |
| Pre-commit hooks | 9 / 10 |
| Type checking | 6 / 10 |

### Strengths

- **15 pre-commit hooks** cover formatting, linting, security, schema validation, lock-file consistency, and project-specific checks.
- **`ruff` with 100+ rules enabled**, including `B` (bugbear), `SIM` (simplify), `PT` (pytest), `S` (security), `TRY`. The configuration is strict without being impractical.
- **`interrogate` enforces docstring coverage** on `src/` as both a pre-commit hook and a CI job.
- **`actionlint` validates GitHub Actions syntax** in pre-commit — workflow YAML errors are caught before push.

### Weaknesses

- **`ruff.toml` targets Python 3.11** (`target-version = "py311"`) while `pyproject.toml` requires `>=3.12`. This means ruff will not flag patterns that are valid in 3.11 but could be modernised for 3.12 (e.g., `typing.TypeAlias` vs. `type` statement).
- **`mypy` config has `ignore_missing_imports = true`**, which suppresses errors from untyped third-party libraries. This is a common pragmatic choice but means type errors from, e.g., `scipy` stubs will pass silently.
- **`ty` (the Astral type checker) is the active type checker**, but `mypy` is also configured — two overlapping tools with potentially different findings. It is unclear which is authoritative.

---

## 6. Documentation — 5 / 10

| Subcategory | Score |
|-------------|-------|
| API reference (mkdocstrings) | 8 / 10 |
| Interactive notebooks | 8 / 10 |
| README quality | 4 / 10 |
| Contributing & changelog | 2 / 10 |

### Strengths

- **MkDocs site with `mkdocstrings-python`** auto-generates API reference from docstrings. The four modules are all covered.
- **Two marimo notebooks** (`hrp.py`, `1_over_N.py`) provide interactive, runnable demonstrations — superior to static notebooks for reproducibility.
- **`SECURITY.md`** is up-to-date with correct version table, real contact details, and accurate CI security measures.

### Weaknesses

- **README is thin.** At 85 lines it covers only a quick-start example and installation. There is no explanation of *why* HRP exists, when to use bisection vs. Ward, or how to interpret the `Node` result tree. A user encountering the library for the first time has to read the paper.
- **No `CONTRIBUTING.md` at the repo root.** It lives in `.rhiza/CONTRIBUTING.md` (managed by the template), meaning GitHub's "Contribute" button and many developer journeys will not surface it.
- **No `CHANGELOG` file.** `git-cliff` generates release notes per-tag but there is no persistent changelog in the repo. This makes it hard to audit breaking changes without reading release pages.
- **`demo.png` (the dendrogram image in README) is a static file** with no generation script committed alongside it. It may silently become stale as the algorithm evolves.

---

## 7. Project Hygiene — 7 / 10

| Subcategory | Score |
|-------------|-------|
| Reproducibility (lock file, SBOM, OIDC) | 9 / 10 |
| Standard files | 7 / 10 |
| Community health files | 3 / 10 |

| File | Status | Note |
|------|--------|------|
| `LICENSE` | ✅ Present | MIT, 2025 Jebel Quant Research |
| `SECURITY.md` | ✅ Present | Accurate and up to date |
| `.gitignore` | ✅ Present | 123 lines, comprehensive |
| `.python-version` | ✅ Present | `3.12` |
| `uv.lock` | ✅ Committed | Reproducible builds |
| `CONTRIBUTING.md` | ⚠️ In `.rhiza/` | Not at repo root |
| `CODE_OF_CONDUCT.md` | ⚠️ In `.rhiza/` | Not at repo root |
| `CHANGELOG` | ❌ Absent | Auto-generated at release only |

---

## Summary

### Strengths

1. **Algorithmically correct and faithful to the paper.** The risk-parity weighting, bisection, and Ward-linkage reconstruction are non-trivial and implemented correctly, validated against reference CSVs.
2. **Excellent CI/CD security posture.** SBOM, SLSA provenance, OIDC publishing, Bandit, and pip-audit together place this well above typical open-source Python projects.
3. **Strong test suite for a 687-line library.** 59 tests, reference-data validation, parametrised over all linkage methods, and type-error coverage give high confidence in correctness.
4. **Strict, automated code quality.** 15 pre-commit hooks, ruff with 100+ rules, interrogate docstring enforcement, and `ty` type checking all run in CI on every push.
5. **Small, purposeful dependency set** with a locked `uv.lock` and `deptry` drift detection.

### Weaknesses

1. **3 `# type: ignore` pragmas** (`hrp.py`: 1, `cluster.py`: 2) point to an unparameterised generic base class (`treelib.Node`) that leaks complexity into every subclass.
2. **`one_over_n` accepts `Any`** — a type-safety gap in the public API.
3. **No property-based or edge-case numerical tests** despite `stress`/`property` markers being registered. Near-singular matrices and degenerate inputs are untested.
4. **No coverage threshold gate in CI** — a PR that removes tests passes undetected.
5. **`ruff.toml` targets Python 3.11**, one version behind the declared minimum, causing modernisation opportunities to be missed.
6. **`cvx-linalg` is a heavyweight dependency for a single three-line operation** (`a_norm = x @ A @ x`).
7. **README does not explain the domain.** Users unfamiliar with HRP get a code example but no conceptual orientation.
8. **`CONTRIBUTING.md` is buried in `.rhiza/`** and not surfaced at the repo root.
