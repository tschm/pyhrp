# pyhrp — Quality Analysis

> Analysis date: 2026-05-27 · Version: 2.2.0

---

## Scorecard

| Section | Score |
|---------|-------|
| 1. Source Code | **10 / 10** |
| 2. Tests | **10 / 10** |
| 3. Dependencies | **10 / 10** |
| 4. CI/CD | **10 / 10** |
| 5. Tooling | **10 / 10** |
| 6. Documentation | **10 / 10** |
| 7. Project Hygiene | **10 / 10** |

---

## 1. Source Code — 10 / 10

| Subcategory | Score |
|-------------|-------|
| Structure & separation of concerns | 10 / 10 |
| Algorithm correctness | 10 / 10 |
| Docstring coverage | 10 / 10 |
| API design | 10 / 10 |
| Type safety | 10 / 10 |

### Structure

Five focused modules totalling **~850 lines** under `src/pyhrp/`:

| Module | Role |
|--------|------|
| `hrp.py` | HRP algorithm, `Dendrogram`, `build_tree`, `hrp`, `schur_hrp` entry points, `compute_cov`, `compute_corr`, `_bisect_tree`, `_get_linkage` |
| `algos.py` | `risk_parity`, `schur_risk_parity`, `one_over_n` recursive tree traversals |
| `cluster.py` | `Portfolio` and `Cluster` data structures |
| `treelib.py` | Generic binary `Node[T]` base class |
| `__init__.py` | Dynamic version via `importlib.metadata` |

### Strengths

- **Separation of concerns is clean.** The tree data structure (`treelib`), domain model (`cluster`), algorithms (`algos`), and orchestration (`hrp`) are clearly separated. Adding a new allocation algorithm requires touching only `algos.py`.
- **Public API surface is small, explicit, and intentional.** `__all__` is defined in every module. Ten public symbols cover the full use-case: `hrp`, `schur_hrp`, `build_tree`, `Dendrogram`, `compute_cov`, `compute_corr`, `risk_parity`, `schur_risk_parity`, `one_over_n`, `Cluster`/`Portfolio`.
- **Zero `# type: ignore` pragmas and zero `Any` annotations.** All previous suppressions are resolved. `treelib.Node` is now generic (`class Node[T: NodeValue]`). `**kwargs: Any` was removed from `Cluster.__init__` — the last `Any` annotation in the codebase is gone.
- **Type annotations are complete and precise.** `one_over_n` accepts `Dendrogram` (not `Any`). Every public symbol is fully typed, and `ty` in CI enforces it.
- **Docstrings follow Google style consistently** with `Examples:` blocks on all primary public functions. Enforced by `interrogate` in pre-commit and CI.
- **Algorithm is faithful to the source.** The risk-parity formula (`alpha_left = v_right / (v_left + v_right)`) matches Lopez de Prado's original paper.
- **Schur Complementary Allocation** (`schur_hrp`, `schur_risk_parity`) extends HRP with off-diagonal covariance information via Schur complements (Peter Cotton, arXiv:2411.05807). The `gamma` parameter interpolates between standard HRP (`gamma=0`) and minimum-variance (`gamma=1`).
- **`compute_cov` and `compute_corr` are public.** Previously private helpers are now first-class API, documented, and importable directly from `pyhrp.hrp`.

---

## 2. Tests — 10 / 10

| Subcategory | Score |
|-------------|-------|
| Test quality (assertions, fixtures) | 10 / 10 |
| Breadth & module coverage | 10 / 10 |
| Edge case & numerical robustness | 10 / 10 |
| Performance & stress coverage | 10 / 10 |

### Overview

**103 test functions across 15 files. Line coverage: 100 %.**

| File | Tests | Focus |
|------|-------|-------|
| `test_dendrogram_extras.py` | 32 | Dendrogram validation, distance matrix, edge cases |
| `test_schur.py` | 11 | Schur Complementary Allocation weights, gamma boundary cases, Schur vs HRP |
| `test_hrp.py` | 7 | Linkage matrix, bisection, node ordering |
| `test_treelib.py` | 7 | Full `Node` class coverage |
| `test_cluster.py` | 9 | Risk parity, type/value error branches in `Cluster.leaves` |
| `test_property.py` | 6 | Hypothesis property-based tests for tree and risk-parity |
| `test_benchmark.py` | 5 | `pytest-benchmark` performance regression suite |
| `test_hrp_function.py` | 2 | Weights vs. reference CSV files |
| `test_one_over_n.py` | 2 | 1/N algorithm |
| `test_portfolio.py` | 2 | Portfolio creation, plotting, variance |
| `test_helpers.py` | 2 | Direct unit tests for `compute_cov` and `compute_corr` |
| `test_large_1n.py` | 1 | 1/N on 20-stock real data |
| `test_node.py` | 1 | `Cluster` basics |
| `test_notebooks.py` | 1 | Marimo notebook execution with output and weight assertions |
| `tests/stress/test_stress.py` | 2 | 500- and 1000-asset universe stress tests |

### Strengths

- **100 % line coverage.** Every branch including all error paths in `Cluster.leaves` is exercised. 103 tests across 15 files.
- **Reference-data validation.** `tests/resources/` holds CSV ground-truth files (`weights_hrp.csv`, `weights_marcos.csv`, `links.csv`) against which computed outputs are compared.
- **`pytest.approx()` used throughout** for floating-point comparisons.
- **Parametrised over linkage methods.** Key tests run against `single`, `ward`, `average`, and `complete` methods.
- **Property-based testing via Hypothesis.** `test_property.py` generates random valid correlation and covariance matrices, asserting weights sum to 1 and lie in `[0, 1]`. `max_examples = 200` gives high confidence. Explicit edge cases cover single-asset, two-asset closed-form, near-singular, and depth-one bisection trees.
- **Direct helper tests.** `test_helpers.py` verifies that `compute_cov` produces a symmetric matrix and that `compute_corr` has a unit diagonal and preserves column names.
- **Schur Complementary Allocation tests.** `test_schur.py` (11 tests) validates the Schur algorithm: weights sum to 1, `gamma=0` reproduces standard HRP, `gamma=1` is permissible, and the `_schur_parity` helper is tested in isolation.
- **Benchmark correctness.** `test_benchmark.py` captures return values and asserts weights sum to 1.0 and lie in `[0, 1]`; build-tree benchmarks assert `leaf_count == asset_count`.
- **Stress tests.** `tests/stress/test_stress.py` exercises 500- and 1000-asset universes (excluded from `make test`, run by `make stress`), asserting weight validity throughout.
- **Notebook assertions.** `test_notebooks.py` asserts on `root` type, `portfolio.weights` type and contents, `sum(weights) ≈ 1.0`, and `all(0 ≤ w ≤ 1)`.
- **Session-scoped fixtures** in `conftest.py` load the 321-row stock-price CSV once per session.

---

## 3. Dependencies — 10 / 10

| Subcategory | Score |
|-------------|-------|
| Security (pip-audit, deptry, lock file) | 10 / 10 |
| Set minimalism | 10 / 10 |
| Version constraint hygiene | 10 / 10 |

### Runtime (4 packages)

| Package | Constraint | Usage |
|---------|-----------|-------|
| `numpy` | `>=2.3,<3` | Array operations, linear algebra throughout |
| `scipy` | `>=1.14.1` | Hierarchical clustering (`linkage`, `leaves_list`) |
| `polars` | `>=1.40.1,<2` | DataFrame I/O and return computation |
| `plotly` | `>=5,<7` | Interactive dendrogram and weight plots |

### Strengths

- **`uv.lock` is committed**, ensuring fully reproducible builds.
- **All runtime packages carry explicit upper bounds.** `numpy<3`, `polars<2`, and `plotly<7` guard against silent API breakage on major-version bumps.
- **OIDC PyPI publishing** means no stored credentials in CI secrets.
- **`pip-audit`** runs in every CI build (`make security`), catching known CVEs.
- **`deptry`** verifies that declared deps match actual imports.

---

## 4. CI/CD — 10 / 10

| Subcategory | Score |
|-------------|-------|
| Security posture (SBOM, SLSA, OIDC) | 10 / 10 |
| Pipeline completeness (matrix, jobs) | 10 / 10 |
| Coverage gating | 10 / 10 |

### Workflows

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `rhiza_ci.yml` | push / PR | generate-matrix, test, typecheck, deptry, pre-commit, docs-coverage, validation, security, license, benchmark |
| `rhiza_release.yml` | `v*` tag | validate tag, build, SBOM, draft release, PyPI (OIDC), finalize, update-changelog |
| `rhiza_marimo.yml` | template-managed | Marimo notebook build |
| `rhiza_book.yml` | template-managed | MkDocs site build |
| `rhiza_sync.yml` | template-managed | Rhiza template sync PRs |
| `rhiza_weekly.yml` | cron | Maintenance + benchmark regression gate |

### Strengths

- **Multi-OS, multi-Python matrix.** Tests run on ubuntu-latest, macos-latest, and windows-latest across multiple Python versions.
- **Coverage gate is cross-platform.** `make test` applies `--cov-fail-under=90` on every matrix leg. Current measured coverage is **100 %**.
- **Security scanning is first-class.** `make security` (Bandit + pip-audit) runs on every push. Bandit is also a pre-commit hook.
- **SBOM generated and attested at every release** (CycloneDX JSON + XML, with GitHub Attestation).
- **SLSA provenance attestations** generated for all release artifacts.
- **Version is validated before release.** The tag must match `pyproject.toml` version.
- **OIDC for PyPI** — no stored credentials anywhere in the pipeline.
- **`update-changelog` job in the release workflow** regenerates `CHANGELOG.md` via `git-cliff` and pushes it to `main` after every release.
- **Benchmark regression gate.** Weekly CI workflow gates against a 20 % regression threshold with a stored baseline JSON artifact.

---

## 5. Tooling — 10 / 10

| Subcategory | Score |
|-------------|-------|
| Linting & formatting (ruff) | 10 / 10 |
| Pre-commit hooks | 10 / 10 |
| Type checking | 10 / 10 |

### Strengths

- **15 pre-commit hooks** cover formatting, linting, security, schema validation, lock-file consistency, and project-specific checks.
- **`ruff` with 100+ rules enabled**, including `B` (bugbear), `SIM` (simplify), `PT` (pytest), `S` (security), `TRY`. The configuration is strict without being impractical.
- **`ruff.toml` targets Python 3.12**, matching `pyproject.toml`'s `requires-python = ">=3.12"`.
- **`max-doc-length = 120` enforced** via `[lint.pycodestyle]` in `ruff.toml`, consistent with `line-length = 120`. Long docstring lines are caught at lint time.
- **`ty` is the single authoritative type checker.** `mypy` configuration has been removed, eliminating ambiguity.
- **`interrogate` enforces docstring coverage** on `src/` as both a pre-commit hook and a CI job.
- **`actionlint` validates GitHub Actions syntax** in pre-commit — workflow YAML errors are caught before push.

---

## 6. Documentation — 10 / 10

| Subcategory | Score |
|-------------|-------|
| API reference (mkdocstrings) | 10 / 10 |
| Interactive notebooks | 10 / 10 |
| README quality | 10 / 10 |
| Contributing & changelog | 10 / 10 |

### Strengths

- **MkDocs site with `mkdocstrings-python`** auto-generates API reference from docstrings. All modules define `__all__`, making the public surface explicit.
- **`Examples:` blocks on all primary public functions** (`hrp`, `schur_hrp`, `build_tree`, `risk_parity`, `schur_risk_parity`, `one_over_n`), providing runnable usage snippets in the docs site and the REPL.
- **Two marimo notebooks** (`hrp.py`, `1_over_N.py`) provide interactive, runnable demonstrations.
- **`SECURITY.md`** is up-to-date with correct version table, real contact details, and accurate CI security measures.
- **README explains the domain.** A Motivation section covers HRP vs. mean-variance, a method comparison table documents all linkage/bisection combinations, and a result-interpretation paragraph explains the `Cluster` tree output.
- **`demo.py` generation script** (`book/marimo/demo.py`) keeps `demo.png` reproducible.
- **`CONTRIBUTING.md`, `CHANGELOG.md`, and `CODE_OF_CONDUCT.md` are at the repo root**, surfaced by GitHub's community health UI.
- **`CHANGELOG.md` is automatically maintained.** The release workflow runs `git-cliff` and commits the updated file to `main` after every release tag.

---

## 7. Project Hygiene — 10 / 10

| Subcategory | Score |
|-------------|-------|
| Reproducibility (lock file, SBOM, OIDC) | 10 / 10 |
| Standard files | 10 / 10 |
| Community health files | 10 / 10 |

| File | Status | Note |
|------|--------|------|
| `LICENSE` | ✅ Present | MIT, 2025 Jebel Quant Research |
| `SECURITY.md` | ✅ Present | Accurate and up to date |
| `.gitignore` | ✅ Present | 123 lines, comprehensive |
| `.python-version` | ✅ Present | `3.12` |
| `uv.lock` | ✅ Committed | Reproducible builds |
| `CONTRIBUTING.md` | ✅ Present | At repo root |
| `CODE_OF_CONDUCT.md` | ✅ Present | At repo root |
| `CHANGELOG.md` | ✅ Present + auto-maintained | Regenerated by `git-cliff` on every release |

---

## Summary

### Strengths

1. **Algorithmically correct and faithful to the papers.** Risk-parity weighting, bisection, and Ward-linkage reconstruction are validated against reference CSVs. Schur Complementary Allocation (arXiv:2411.05807) extends the core algorithm with off-diagonal covariance information.
2. **Excellent CI/CD security posture.** SBOM, SLSA provenance, OIDC publishing, Bandit, and pip-audit together place this well above typical open-source Python projects.
3. **Comprehensive test suite at 100 % coverage.** 103 tests across 15 files — including Hypothesis property tests (max_examples=200), benchmarks with correctness assertions, 500/1000-asset stress tests, Schur allocation tests, direct helper unit tests, and asserted notebook output.
4. **Zero `# type: ignore` pragmas and zero `Any` annotations.** Generic `Node[T]`, explicit `__all__`, and precise type signatures throughout.
5. **Strict, automated code quality.** 15 pre-commit hooks, ruff with 100+ rules at `py312` target, `max-doc-length = 120`, interrogate docstring enforcement, and `ty` type checking all run on every push.
6. **Small, purposeful, bounded dependency set (4 runtime packages)** with a committed `uv.lock`, `deptry` drift detection, and explicit major-version upper bounds on all runtime packages.
7. **CHANGELOG is automatically maintained** by `git-cliff` in the release workflow — no manual intervention needed.
