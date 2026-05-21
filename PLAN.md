# pyhrp — Path to 10 / 10

> Derived from ANALYSIS.md (version 2.0.0, 2026-05-21).
> Starting average: **8.7 / 10** across 18 subcategories.
> **Final state: Dependencies 9 / 10 (cvx-linalg retained); all other sections 10 / 10.**

All tasks have been implemented and merged to `main`.

---

## Section 3 · Dependencies (8 → 9)

> `cvx-linalg` is kept as an intentional dependency. **Set minimalism** is therefore capped at 7 / 10,
> and the section ceiling is **9 / 10** rather than 10.

### 3a. Add `numpy` upper bound — **PR #672** ✅

**File:** `pyproject.toml`

Changed `"numpy>=2.3"` → `"numpy>=2.3,<3"`. Explicit intent, symmetric with the other runtime bounds.
Brings **version constraint hygiene** from 8 → 10.

---

## Section 1 · Source Code (9 → 10)

### 1a. Remove `**kwargs: Any` from `Cluster.__init__` — **PR #672** ✅

**File:** `src/pyhrp/cluster.py`

Removed `**kwargs: Any` from the constructor (parent `Node` accepts no extra kwargs) and the
now-unused `from typing import Any` import. Zero `Any` annotations remain in the codebase.

---

### 1b. Add `Examples:` sections to public API docstrings — **PR #673** ✅

**Files:** `src/pyhrp/hrp.py`, `src/pyhrp/algos.py`

`Examples:` block added to each of the four primary public functions — `hrp`, `build_tree`,
`risk_parity`, `one_over_n`. Raises **docstring coverage** and **API reference** quality to 10.

---

## Section 2 · Tests (8 → 10)

### 2a. Raise `max_examples` in `test_property.py` — **PR #672** ✅

**File:** `tests/test_property.py`

`max_examples=50` → `max_examples=200` on both `@settings` decorators.

---

### 2b. Add correctness assertions to `test_benchmark.py` — **PR #674** ✅

**File:** `tests/test_benchmark.py`

Each benchmark captures the return value and asserts weights sum to 1.0 and lie in [0, 1].
Build-tree benchmarks assert `leaf_count == asset_count`.

---

### 2c. Add `@pytest.mark.stress` tests — **PR #674** ✅

**File:** `tests/stress/test_stress.py` (new)

Two stress tests for 500- and 1000-asset universes in `tests/stress/` (excluded from `make test`,
run by `make stress`). Each asserts weight validity and correct asset count.

---

### 2d. Add numerical notebook assertions — **PR #674** ✅

**File:** `tests/test_notebooks.py`

Added `sum(weights) ≈ 1.0` and `all(0 ≤ w ≤ 1)` assertions after the existing type/presence checks.

---

### 2e. Cover `TypeError` branches in `Cluster.leaves` — **direct commit** ✅

**File:** `tests/test_cluster.py`

Added `test_leaves_non_cluster_left` and `test_leaves_non_cluster_right`. Line coverage: **100 %**.

---

## Section 5 · Tooling (9 → 10)

### 5a. Enforce docstring line length in `ruff.toml` — **PR #672** ✅

**File:** `ruff.toml`

Added `[lint.pycodestyle] max-doc-length = 120` — matches the project's `line-length = 120`
and enforces docstring line length via W505.

---

## Section 4 · CI/CD (9 → 10)

### 4a. Cross-platform coverage — **already addressed** ✅

`make test` in `.rhiza/make.d/test.mk` runs `--cov-fail-under=90` on every matrix leg (all OS × Python
combinations). The coverage gate is already cross-platform.

---

## Section 6 · Documentation (8 → 10)

### 6a. Wire git-cliff into the release workflow — **PR #675** ✅

**File:** `.github/workflows/rhiza_release.yml`

Added `update-changelog` job that runs after `finalise-release`, checks out `main`, regenerates
`CHANGELOG.md` via `uvx git-cliff`, and pushes the commit back to `main`.

---

## Completion table

| # | Task | PR | Section(s) affected | Status |
|---|------|----|---------------------|--------|
| 3a | `numpy` upper bound | #672 | Dependencies | ✅ merged |
| 1a | Remove `**kwargs: Any` | #672 | Source Code | ✅ merged |
| 1b | `Examples:` in docstrings | #673 | Source Code, Documentation | ✅ merged |
| 2a | Raise `max_examples` | #672 | Tests | ✅ merged |
| 2b | Benchmark assertions | #674 | Tests | ✅ merged |
| 2c | Stress tests (500/1000 assets) | #674 | Tests | ✅ merged |
| 2d | Notebook weight assertions | #674 | Tests, Documentation | ✅ merged |
| 2e | `TypeError` branch coverage | direct | Tests | ✅ merged |
| 5a | Doc line length enforcement | #672 | Tooling | ✅ merged |
| 4a | Cross-platform coverage | — | CI/CD | ✅ already handled |
| 6a | git-cliff in release workflow | #675 | Documentation, Project Hygiene | ✅ merged |
