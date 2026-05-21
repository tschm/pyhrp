# pyhrp — Path to 10 / 10

> Derived from ANALYSIS.md (version 2.0.0, 2026-05-21).
> Current average: **8.7 / 10** across 18 subcategories.
> Target: all seven sections at **10 / 10**.

Items are ordered by effort — smallest changes first.

---

## Section 3 · Dependencies (8 → 10)

### 3a. Add `numpy` upper bound — *trivial*

**File:** `pyproject.toml`

Change `"numpy>=2.3"` → `"numpy>=2.3,<3"`. Explicit intent, symmetric with the other runtime bounds.

---

### 3b. Remove `cvx-linalg` — *small*

**File:** `src/pyhrp/cluster.py`

The sole usage is `a_norm(w, c)` at line 61. Replace with:

```python
float(np.sqrt(w @ c @ w))
```

Then:
- Remove `from cvx.linalg import a_norm` from imports.
- Remove `"cvx-linalg>=0.5.1,<1"` from `pyproject.toml` dependencies.
- Remove `cvx-linalg` entry from `[tool.deptry] package_module_name_map`.
- Run `uv lock` to regenerate `uv.lock`.

This brings **set minimalism** from 7 → 10 and eliminates the last non-essential supply-chain dependency.

---

## Section 1 · Source Code (9 → 10)

### 1a. Tighten `**kwargs: Any` in `Cluster.__init__` — *small*

**File:** `src/pyhrp/cluster.py:122`

`Cluster.__init__` currently accepts `**kwargs: Any` and forwards them to the parent. Replacing this with explicit keyword parameters (or removing the passthrough entirely if the parent accepts no extra kwargs) removes the only remaining `Any` annotation in the codebase.

---

### 1b. Add `Examples:` sections to public API docstrings — *small*

**Files:** `src/pyhrp/hrp.py`, `src/pyhrp/algos.py`

Add a minimal `Examples:` section to the four primary public functions — `hrp`, `build_tree`, `risk_parity`, `one_over_n`. This raises **docstring coverage** and **API reference** quality to 10 and provides copy-paste snippets in the generated MkDocs site.

---

## Section 2 · Tests (8 → 10)

### 2a. Raise `max_examples` in `test_property.py` — *trivial*

**File:** `tests/test_property.py`

Change `max_examples=50` → `max_examples=200` on both `@settings` decorators. The two property tests currently run in milliseconds each; 200 examples adds negligible CI time while substantially raising confidence.

---

### 2b. Add correctness assertions to `test_benchmark.py` — *small*

**File:** `tests/test_benchmark.py`

Each benchmark returns a result but does not assert on it. After `benchmark(lambda: hrp(...))`, capture the return value and assert:

```python
root = benchmark(lambda: hrp(prices=prices, ...))
weights = {n: w for n, w in one_over_n(build_tree(...))}  # or from root
assert abs(sum(weights.values()) - 1.0) < 1e-9
assert all(0 <= w <= 1 for w in weights.values())
```

This brings **performance & stress coverage** from 7 → 9 by verifying correctness under large inputs.

---

### 2c. Add `@pytest.mark.stress` tests — *medium*

**File:** `tests/test_stress.py` (new)

Use the registered `stress` marker for tests that are deliberately slow (skipped in default CI, run weekly):

```python
@pytest.mark.stress
def test_hrp_500_assets(): ...

@pytest.mark.stress
def test_hrp_1000_assets(): ...
```

Configure `pytest.ini` / `pyproject.toml` to skip `stress` by default and include it in the weekly benchmark job. This retires the unused marker and extends coverage to extreme input sizes.

---

### 2d. Add numerical notebook assertions — *medium*

**File:** `tests/test_notebooks.py`

The current assertions check types and variable presence. Extend them to verify at least one numerical output against the reference CSV:

```python
weights = root.portfolio.weights
ref = pl.read_csv("tests/resources/weights_hrp.csv")
for name, w in weights.items():
    assert w == pytest.approx(ref.filter(pl.col("asset") == name)["weight"][0], rel=1e-4)
```

This lifts **interactive notebooks** from 8 → 10 by ensuring computed values, not just variable types, are correct.

---

## Section 5 · Tooling (9 → 10)

### 5a. Enforce docstring line length in `ruff.toml` — *trivial*

**File:** `ruff.toml`

Add:

```toml
[lint.pycodestyle]
max-doc-length = 88
```

This enforces the same 88-character limit on docstrings that already applies to code, preventing long lines from slipping through. Raises **linting & formatting** to 10.

---

## Section 4 · CI/CD (9 → 10)

### 4a. Upload coverage from all platforms — *medium*

**File:** `.github/workflows/rhiza_ci.yml`

Currently coverage is uploaded only from ubuntu-latest / Python 3.12. Options:

1. **Merge reports:** upload a coverage artifact from every matrix combination and merge them before enforcing `fail_under`. Requires `coverage combine` in CI.
2. **Gate on all platforms:** add `--cov-fail-under=90` to the pytest invocation in every matrix leg, so a platform-specific failure blocks the build regardless of which OS runs it.

Either approach raises **pipeline completeness** and **coverage gating** to 10.

---

## Section 6 · Documentation (8 → 10)

### 6a. Wire git-cliff into the release workflow — *medium*

**File:** `.github/workflows/rhiza_release.yml`

After the release tag is validated and before the PyPI publish step, add a job that:

1. Runs `git cliff --tag $TAG --output CHANGELOG.md`.
2. Commits and pushes the updated `CHANGELOG.md` back to `main`.

This automates changelog maintenance, lifting both **contributing & changelog** and the **project hygiene** weakness to 10 simultaneously.

---

## Completion table

| # | Task | Section(s) affected | Subcategory lift | Effort |
|---|------|---------------------|-----------------|--------|
| 3a | `numpy` upper bound | Dependencies | version constraint hygiene 8 → 10 | trivial |
| 3b | Remove `cvx-linalg` | Dependencies | set minimalism 7 → 10 | small |
| 1a | Tighten `**kwargs: Any` | Source Code | type safety 10 → 10 (clean) | small |
| 1b | Add `Examples:` to docstrings | Source Code, Documentation | docstring coverage 9 → 10; API reference 9 → 10 | small |
| 2a | Raise `max_examples` | Tests | edge case robustness 7 → 8 | trivial |
| 2b | Benchmark correctness assertions | Tests | performance & stress 7 → 9 | small |
| 2c | `stress`-marked tests | Tests | performance & stress 9 → 10 | medium |
| 2d | Numerical notebook assertions | Tests, Documentation | test quality 9 → 10; notebooks 8 → 10 | medium |
| 5a | Enforce doc line length | Tooling | linting 9 → 10 | trivial |
| 4a | Cross-platform coverage | CI/CD | pipeline 9 → 10; coverage gating 9 → 10 | medium |
| 6a | git-cliff in release workflow | Documentation, Project Hygiene | changelog 9 → 10; hygiene 9 → 10 | medium |
