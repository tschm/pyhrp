# pyhrp — Path to 10 / 10

> Derived from ANALYSIS.md (version 2.0.0, 2026-05-21).
> Current average: **8.7 / 10** across 18 subcategories.
> Target: six sections at **10 / 10**; Dependencies capped at **9 / 10** (cvx-linalg retained).

Items are ordered by effort — smallest changes first.

---

## Section 3 · Dependencies (8 → 9)

> `cvx-linalg` is kept as an intentional dependency. **Set minimalism** is therefore capped at 7 / 10,
> and the section ceiling is **9 / 10** rather than 10.

### 3a. Add `numpy` upper bound — *trivial* — **PR #672**

**File:** `pyproject.toml`

Change `"numpy>=2.3"` → `"numpy>=2.3,<3"`. Explicit intent, symmetric with the other runtime bounds.
Brings **version constraint hygiene** from 8 → 10.

---

## Section 1 · Source Code (9 → 10)

### 1a. Tighten `**kwargs: Any` in `Cluster.__init__` — *small* — **PR #672**

**File:** `src/pyhrp/cluster.py:122`

Removes `**kwargs: Any` from the constructor (parent `Node` accepts no extra kwargs) and the
now-unused `from typing import Any` import. The last `Any` annotation in the codebase is gone.

---

### 1b. Add `Examples:` sections to public API docstrings — *small* — **PR #673**

**Files:** `src/pyhrp/hrp.py`, `src/pyhrp/algos.py`

Minimal `Examples:` block added to each of the four primary public functions — `hrp`, `build_tree`,
`risk_parity`, `one_over_n`. Raises **docstring coverage** and **API reference** quality to 10.

---

## Section 2 · Tests (8 → 10)

### 2a. Raise `max_examples` in `test_property.py` — *trivial* — **PR #672**

**File:** `tests/test_property.py`

`max_examples=50` → `max_examples=200` on both `@settings` decorators.

---

### 2b. Add correctness assertions to `test_benchmark.py` — *small* — **PR #674**

**File:** `tests/test_benchmark.py`

Each benchmark now captures the return value and asserts weights sum to 1.0 and lie in [0, 1].
Build-tree benchmarks assert `leaf_count == asset_count`.

---

### 2c. Add `@pytest.mark.stress` tests — *medium* — **PR #674**

**File:** `tests/stress/test_stress.py` (new)

Two stress tests for 500- and 1000-asset universes in `tests/stress/` (excluded from `make test`,
run by `make stress`). Each asserts weight validity and correct asset count.

---

### 2d. Add numerical notebook assertions — *medium* — **PR #674**

**File:** `tests/test_notebooks.py`

Added `sum(weights) ≈ 1.0` and `all(0 ≤ w ≤ 1)` assertions after the existing type/presence checks.

---

## Section 5 · Tooling (9 → 10)

### 5a. Enforce docstring line length in `ruff.toml` — *trivial* — **PR #672**

**File:** `ruff.toml`

Added `[lint.pycodestyle] max-doc-length = 120` — matches the project's existing `line-length = 120`
and enforces docstring line length via W505.

---

## Section 4 · CI/CD (9 → 10)

### 4a. Cross-platform coverage — *already addressed*

`make test` in `.rhiza/make.d/test.mk` runs `--cov-fail-under=90` on every matrix leg (all OS × Python
combinations). The coverage gate is already cross-platform. Only the artifact *upload* is
single-platform, which is separate from enforcement. No workflow change needed.

---

## Section 6 · Documentation (8 → 10)

### 6a. Wire git-cliff into the release workflow — *medium* — **PR #675**

**File:** `.github/workflows/rhiza_release.yml`

Added `update-changelog` job that runs after `finalise-release`, checks out `main`, regenerates
`CHANGELOG.md` via `uvx git-cliff`, and pushes the commit back to `main`.

---

## Completion table

| # | Task | PR | Section(s) affected | Subcategory lift | Status |
|---|------|-----|---------------------|-----------------|--------|
| 3a | `numpy` upper bound | #672 | Dependencies | version constraint hygiene 8 → 10 | 🔄 open |
| 1a | Remove `**kwargs: Any` | #672 | Source Code | eliminates last `Any` annotation | 🔄 open |
| 1b | `Examples:` in docstrings | #673 | Source Code, Documentation | docstring coverage 9 → 10; API reference 9 → 10 | 🔄 open |
| 2a | Raise `max_examples` | #672 | Tests | edge case robustness 7 → 9 | 🔄 open |
| 2b | Benchmark assertions | #674 | Tests | performance & stress 7 → 9 | 🔄 open |
| 2c | Stress tests (500/1000 assets) | #674 | Tests | performance & stress 9 → 10 | 🔄 open |
| 2d | Notebook weight assertions | #674 | Tests, Documentation | test quality 9 → 10; notebooks 8 → 10 | 🔄 open |
| 5a | Doc line length enforcement | #672 | Tooling | linting 9 → 10 | 🔄 open |
| 4a | Cross-platform coverage | — | CI/CD | already handled by `make test` | ✅ done |
| 6a | git-cliff in release workflow | #675 | Documentation, Project Hygiene | changelog 9 → 10; hygiene 9 → 10 | 🔄 open |
