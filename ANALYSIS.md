# pyhrp — Quality Analysis

> Analysis date: 2026-05-21 · Version: 2.0.0

---

## Scorecard

| Section | Score |
|---------|-------|
| 1. Source Code | **9 / 10** |
| 2. Tests | **8 / 10** |
| 3. Dependencies | **8 / 10** |
| 4. CI/CD | **9 / 10** |
| 5. Tooling | **9 / 10** |
| 6. Documentation | **8 / 10** |
| 7. Project Hygiene | **9 / 10** |

---

## 1. Source Code — 9 / 10

| Subcategory | Score |
|-------------|-------|
| Structure & separation of concerns | 9 / 10 |
| Algorithm correctness | 9 / 10 |
| Docstring coverage | 9 / 10 |
| API design | 9 / 10 |
| Type safety | 10 / 10 |

### Structure

Five focused modules totalling **677 lines** under `src/pyhrp/`:

| Module | Lines | Role |
|--------|-------|------|
| `hrp.py` | 245 | HRP algorithm, `Dendrogram`, `build_tree`, `hrp` entry point, `_bisect_tree`, `_get_linkage` |
| `algos.py` | 129 | `risk_parity`, `one_over_n` recursive tree traversals |
| `cluster.py` | 165 | `Portfolio` and `Cluster` data structures |
| `treelib.py` | 129 | Generic binary `Node[T]` base class |
| `__init__.py` | 9 | Dynamic version via `importlib.metadata` |

### Strengths

- **Separation of concerns is clean.** The tree data structure (`treelib`), domain model (`cluster`), algorithms (`algos`), and orchestration (`hrp`) are clearly separated. Adding a new allocation algorithm requires touching only `algos.py`.
- **Public API surface is small, explicit, and intentional.** `__all__` is defined in every module. Six public symbols cover the full use-case: `hrp`, `build_tree`, `Dendrogram`, `risk_parity`, `one_over_n`, `Cluster`/`Portfolio`.
- **Zero `# type: ignore` pragmas.** All previous suppressions are resolved. `treelib.Node` is now generic (`class Node[T: NodeValue]`), eliminating the root cause of the type inference issues in subclasses.
- **Type annotations are complete and precise.** `one_over_n` accepts `Dendrogram` (not `Any`). Every public symbol is fully typed, and `ty` in CI enforces it.
- **Docstrings follow Google style consistently** and are enforced by `interrogate` in pre-commit and CI. `_bisect_tree` and `_get_linkage` now carry concise docstrings as module-level private functions.
- **Algorithm is faithful to the source.** The risk-parity formula (`alpha_left = v_right / (v_left + v_right)`) matches Lopez de Prado's original paper.

### Weaknesses

- **`**kwargs: Any` in `Cluster.__init__`** (`cluster.py:122`) is the only remaining `Any` annotation. It is used for pass-through to the parent constructor, but a stricter keyword-argument type could make the contract more explicit.

---

## 2. Tests — 8 / 10

| Subcategory | Score |
|-------------|-------|
| Test quality (assertions, fixtures) | 9 / 10 |
| Breadth & module coverage | 9 / 10 |
| Edge case & numerical robustness | 7 / 10 |
| Performance & stress coverage | 7 / 10 |

### Overview

**75 test functions across 13 files, 1 345 lines.**

| File | Tests | Focus |
|------|-------|-------|
| `test_dendrogram_extras.py` | 32 | Dendrogram validation, distance matrix, edge cases |
| `test_hrp.py` | 7 | Linkage matrix, bisection, node ordering |
| `test_treelib.py` | 7 | Full `Node` class coverage |
| `test_cluster.py` | 7 | Risk parity, type errors |
| `test_property.py` | 6 | Hypothesis property-based tests for tree and risk-parity |
| `test_benchmark.py` | 5 | `pytest-benchmark` performance regression suite |
| `test_hrp_function.py` | 2 | Weights vs. reference CSV files |
| `test_one_over_n.py` | 2 | 1/N algorithm |
| `test_portfolio.py` | 2 | Portfolio creation, plotting, variance |
| `test_helpers.py` | 2 | Direct unit tests for `_compute_cov` and `_compute_corr` |
| `test_large_1n.py` | 1 | 1/N on 20-stock real data |
| `test_node.py` | 1 | `Cluster` basics |
| `test_notebooks.py` | 1 | Marimo notebook execution with output assertions |

### Strengths

- **Reference-data validation.** `tests/resources/` holds CSV ground-truth files (`weights_hrp.csv`, `weights_marcos.csv`, `links.csv`) against which computed outputs are compared.
- **`pytest.approx()` used throughout** for floating-point comparisons.
- **Parametrised over linkage methods.** `test_invariant_order`, `test_build_tree_with_different_methods`, and `test_hrp_with_different_methods` each run against `single`, `ward`, `average`, and `complete` methods.
- **Property-based testing via Hypothesis.** `test_property.py` generates random valid correlation and covariance matrices, asserting weights sum to 1 and lie in `[0, 1]`. Explicit edge cases cover single-asset, two-asset closed-form, near-singular, and depth-one bisection trees.
- **Direct helper tests.** `test_helpers.py` verifies that `_compute_cov` produces a symmetric matrix and that `_compute_corr` has a unit diagonal and preserves column names — previously tested only indirectly through `hrp()`.
- **Notebook assertions.** `test_notebooks.py` inspects the returned namespace after `runpy.run_path`, asserting on the `root` type, `portfolio.weights` type and contents, and the presence of the `app` variable.
- **Performance regression suite.** `test_benchmark.py` benchmarks 20-, 100-, and 200-asset universes via `pytest-benchmark`, with a weekly CI workflow gating against a 20 % regression threshold.
- **Session-scoped fixtures** in `conftest.py` load the 321-row stock-price CSV once per session.

### Weaknesses

- **`test_benchmark.py` has no inline correctness assertions** — it only records timings. A numerically incorrect result for large inputs would pass silently.
- **Hypothesis `max_examples = 50`** is modest for a numerical library. Increasing to 200+ would raise confidence without significantly slowing CI.
- **The `stress` pytest marker is defined but unused.** It was registered in anticipation of stress tests that have not yet been written.

---

## 3. Dependencies — 8 / 10

| Subcategory | Score |
|-------------|-------|
| Security (pip-audit, deptry, lock file) | 9 / 10 |
| Set minimalism | 7 / 10 |
| Version constraint hygiene | 8 / 10 |

### Runtime (5 packages)

| Package | Constraint | Usage |
|---------|-----------|-------|
| `numpy` | `>=2.3` | Array operations, linear algebra throughout |
| `scipy` | `>=1.14.1` | Hierarchical clustering (`linkage`, `leaves_list`) |
| `polars` | `>=1.40.1,<2` | DataFrame I/O and return computation |
| `cvx-linalg` | `>=0.5.1,<1` | `a_norm()` for portfolio variance in `cluster.py` |
| `plotly` | `>=5,<7` | Interactive dendrogram and weight plots |

### Strengths

- **`uv.lock` is committed**, ensuring fully reproducible builds.
- **Dependency set is minimal and purposeful.** Five packages, each with a clear reason.
- **OIDC PyPI publishing** means no stored credentials in CI secrets.
- **`pip-audit`** runs in every CI build (`make security`), catching known CVEs.
- **`deptry`** verifies that declared deps match actual imports.
- **All runtime packages now carry explicit upper bounds.** `polars<2`, `plotly<7`, and `cvx-linalg<1` guard against silent API breakage on major-version bumps.

### Weaknesses

- **`cvx-linalg` is a heavy dependency for a single utility function.** `a_norm(w, c)` computes `sqrt(w @ c @ w)`. This is four tokens of NumPy. Depending on an external package introduces unnecessary supply-chain surface.
- **`numpy` has no upper bound.** An explicit `numpy<3` would make the intent clear and protect against a hypothetical NumPy 3.0 API change.

---

## 4. CI/CD — 9 / 10

| Subcategory | Score |
|-------------|-------|
| Security posture (SBOM, SLSA, OIDC) | 10 / 10 |
| Pipeline completeness (matrix, jobs) | 9 / 10 |
| Coverage gating | 9 / 10 |

### Workflows

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `rhiza_ci.yml` | push / PR | generate-matrix, test, typecheck, deptry, pre-commit, docs-coverage, validation, security, license, benchmark |
| `rhiza_release.yml` | `v*` tag | validate tag, build, SBOM, draft release, PyPI (OIDC), finalize |
| `rhiza_marimo.yml` | template-managed | Marimo notebook build |
| `rhiza_book.yml` | template-managed | MkDocs site build |
| `rhiza_sync.yml` | template-managed | Rhiza template sync PRs |
| `rhiza_weekly.yml` | cron | Maintenance + benchmark regression gate |

### Strengths

- **Multi-OS, multi-Python matrix.** Tests run on ubuntu-latest, macos-latest, and windows-latest across multiple Python versions.
- **Security scanning is first-class.** `make security` (Bandit + pip-audit) runs on every push. Bandit is also a pre-commit hook.
- **SBOM generated and attested at every release** (CycloneDX JSON + XML, with GitHub Attestation). Well above industry baseline.
- **SLSA provenance attestations** generated for all release artifacts.
- **Version is validated before release.** The tag must match `pyproject.toml` version.
- **OIDC for PyPI** — no stored credentials anywhere in the pipeline.
- **Coverage gate enforced.** `fail_under = 90` in `pyproject.toml`; current measured coverage is 100 %.

### Weaknesses

- **Coverage upload is only from ubuntu-latest / Python 3.12.** A platform-specific failure (e.g., a Windows path bug) would not be caught by the coverage gate.
- **`test_benchmark.py` is in the test matrix** but contributes no correctness assertions — only timing data.

---

## 5. Tooling — 9 / 10

| Subcategory | Score |
|-------------|-------|
| Linting & formatting (ruff) | 9 / 10 |
| Pre-commit hooks | 9 / 10 |
| Type checking | 9 / 10 |

### Strengths

- **15 pre-commit hooks** cover formatting, linting, security, schema validation, lock-file consistency, and project-specific checks.
- **`ruff` with 100+ rules enabled**, including `B` (bugbear), `SIM` (simplify), `PT` (pytest), `S` (security), `TRY`. The configuration is strict without being impractical.
- **`ruff.toml` targets Python 3.12**, matching `pyproject.toml`'s `requires-python = ">=3.12"`.
- **`ty` is the single authoritative type checker.** `mypy` configuration has been removed, eliminating ambiguity about which tool is authoritative.
- **`interrogate` enforces docstring coverage** on `src/` as both a pre-commit hook and a CI job.
- **`actionlint` validates GitHub Actions syntax** in pre-commit — workflow YAML errors are caught before push.

### Weaknesses

- **No enforced maximum line length beyond ruff's default 88.** A few long docstring lines pass silently.

---

## 6. Documentation — 8 / 10

| Subcategory | Score |
|-------------|-------|
| API reference (mkdocstrings) | 9 / 10 |
| Interactive notebooks | 8 / 10 |
| README quality | 8 / 10 |
| Contributing & changelog | 9 / 10 |

### Strengths

- **MkDocs site with `mkdocstrings-python`** auto-generates API reference from docstrings. All modules define `__all__`, making the public surface explicit.
- **Two marimo notebooks** (`hrp.py`, `1_over_N.py`) provide interactive, runnable demonstrations.
- **`SECURITY.md`** is up-to-date with correct version table, real contact details, and accurate CI security measures.
- **README now explains the domain.** A Motivation section covers HRP vs. mean-variance, a method comparison table documents all linkage/bisection combinations, and a result-interpretation paragraph explains the `Cluster` tree output.
- **`demo.py` generation script** (`book/marimo/demo.py`) keeps `demo.png` reproducible — it is no longer a static file that can silently go stale.
- **`CONTRIBUTING.md`, `CHANGELOG.md`, and `CODE_OF_CONDUCT.md` are at the repo root**, surfaced by GitHub's community health UI.

### Weaknesses

- **`CHANGELOG.md` is seeded but not wired into the release workflow.** Future release notes appear only on GitHub Releases unless git-cliff is integrated into the release automation.
- **Notebook tests assert on variable presence and type, not numerical values.** A silent regression in computed weights would pass `test_notebooks.py` as long as the `root` and `app` variables are present.

---

## 7. Project Hygiene — 9 / 10

| Subcategory | Score |
|-------------|-------|
| Reproducibility (lock file, SBOM, OIDC) | 9 / 10 |
| Standard files | 9 / 10 |
| Community health files | 9 / 10 |

| File | Status | Note |
|------|--------|------|
| `LICENSE` | ✅ Present | MIT, 2025 Jebel Quant Research |
| `SECURITY.md` | ✅ Present | Accurate and up to date |
| `.gitignore` | ✅ Present | 123 lines, comprehensive |
| `.python-version` | ✅ Present | `3.12` |
| `uv.lock` | ✅ Committed | Reproducible builds |
| `CONTRIBUTING.md` | ✅ Present | At repo root |
| `CODE_OF_CONDUCT.md` | ✅ Present | At repo root |
| `CHANGELOG.md` | ✅ Present | At repo root (seeded from git-cliff) |

### Weaknesses

- **`CHANGELOG.md` is not maintained automatically.** Without a git-cliff hook in the release workflow it will drift after the first new release.

---

## Summary

### Strengths

1. **Algorithmically correct and faithful to the paper.** Risk-parity weighting, bisection, and Ward-linkage reconstruction are non-trivial and validated against reference CSVs.
2. **Excellent CI/CD security posture.** SBOM, SLSA provenance, OIDC publishing, Bandit, and pip-audit together place this well above typical open-source Python projects.
3. **Strong, growing test suite.** 75 tests across 13 files — including Hypothesis property tests, performance benchmarks, direct helper unit tests, and asserted notebook output — give high confidence in correctness and stability.
4. **Zero `# type: ignore` pragmas.** Generic `Node[T]`, explicit `__all__`, and precise type signatures throughout. `ty` is the single authoritative checker in CI.
5. **Strict, automated code quality.** 15 pre-commit hooks, ruff with 100+ rules at `py312` target, interrogate docstring enforcement, and `ty` type checking all run on every push.
6. **Small, purposeful, bounded dependency set** with a committed `uv.lock`, `deptry` drift detection, and explicit major-version upper bounds on all runtime packages.

### Weaknesses

1. **`cvx-linalg` is a heavyweight dependency for a single three-line operation** (`sqrt(w @ c @ w)`). It is the only remaining meaningful supply-chain risk.
2. **Coverage gate is single-platform.** Only ubuntu-latest / Python 3.12 uploads coverage; platform-specific regressions on macOS or Windows pass undetected.
3. **`CHANGELOG.md` is not maintained automatically.** Without a release-workflow hook it will go stale.
4. **Hypothesis `max_examples = 50`** is conservative; higher counts would strengthen property-test guarantees at low CI cost.
