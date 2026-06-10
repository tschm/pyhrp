## [2.3.1] - 2026-06-10

### 💼 Other

- Bump version 2.3.0 → 2.3.1
## [2.3.0] - 2026-06-10

### 💼 Other

- Bump version 2.2.6 → 2.3.0

### 🚜 Refactor

- Harden API, validate inputs, speed up allocation
## [2.2.6] - 2026-06-08

### 🐛 Bug Fixes

- Align pyproject.toml with rhiza structure tests
- Add explicit type annotations in Node.levels to satisfy ty
- Remove duplicate uses keys in workflow files

### 💼 Other

- Bump version 2.2.5 → 2.2.6

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG.md for v2.2.5 [skip ci]
- Upgrade actions/checkout to v6.0.2
- Remove lowest_deps.yml (now in rhiza_ci)
- Bump rhiza to v0.18.4
- Apply rhiza sync v0.18.4
- Add classifiers to pyproject.toml
- Add pip dependabot entry for .rhiza/requirements
- Bump rhiza reusable workflows from v0.16.0 to v0.18.4
- Bump rhiza to v0.18.8
- Bump rhiza to v0.18.8
## [2.2.5] - 2026-05-27

### 🐛 Bug Fixes

- Inline release jobs to fix PyPI Trusted Publishing with reusable workflows

### 💼 Other

- Bump version 2.2.4 → 2.2.5

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG.md for v2.2.4 [skip ci]
## [2.2.4] - 2026-05-27

### 🐛 Bug Fixes

- Pass required tag input to rhiza release workflow

### 💼 Other

- Bump version 2.2.3 → 2.2.4
## [2.2.3] - 2026-05-27

### 💼 Other

- Bump version 2.2.2 → 2.2.3
## [2.2.2] - 2026-05-27

### 💼 Other

- Bump version 2.2.1 → 2.2.2

### ⚙️ Miscellaneous Tasks

- Relax dependency floors and update package description
- Add workflow to test with lowest-direct dependency floors
## [2.2.1] - 2026-05-27

### 🚀 Features

- Add profiles: github-project to template.yml

### 🐛 Bug Fixes

- Add uv install retry step to CI test job
- Remove duplicate update-changelog job from release workflow
- Use Generic[T] to keep Node subscriptable on older CI Python

### 💼 Other

- Bump version 2.2.0 → 2.2.1

### 🚜 Refactor

- Remove cvx-linalg dependency

### 🧪 Testing

- Increase coverage to 100% by testing TypeError guard branches

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG.md for v2.2.0
- Bump rhiza template to v0.11.0
- Sync with rhiza v0.11.0
- Sync with rhiza v0.16.1
- Bump rhiza to v0.15.3
- Apply rhiza sync v0.15.3
- Remove blank line from .python-version
- Bump rhiza to v0.16.0
- Apply rhiza sync v0.16.0
- Bump rhiza to v0.17.0
- Apply rhiza sync v0.17.0
## [2.2.0] - 2026-05-22

### 🚀 Features

- Implement Schur Complementary Allocation (Peter Cotton, arXiv:2411.05807)

### 💼 Other

- Bump version 2.1.0 → 2.2.0

### 🚜 Refactor

- Promote _compute_cov and _compute_corr to public API

### 📚 Documentation

- Update README example to use public compute_cov / compute_corr
- List public helper APIs in hrp module docstring

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG.md for v2.1.0
- Update marimo notebooks to v0.23.6
## [2.1.0] - 2026-05-21

### 💼 Other

- Add kaleido as dev dependency
- Bump version 2.0.0 → 2.1.0

### 🚜 Refactor

- Avoid private API imports in demo script

### 📚 Documentation

- Align README badges with jquantstats and rewrite SECURITY.md for pyhrp
- Add root contribution docs and changelog
- Expand README and add demo image generation script
- Use portable command path for demo generation
- Address review feedback in README and demo script
- Keep robust README image URL
- Add missing docstrings to public functions in demo.py
- Add inline script metadata header to demo.py
- Add Examples sections to hrp, build_tree, risk_parity, one_over_n

### 🧪 Testing

- Add stress benchmarks and weekly regression workflow
- Add property-based and numerical edge-case HRP tests
- Tighten zip strictness in property test generators
- Add direct helper tests and notebook output assertions
- Benchmark assertions, 500/1000-asset stress tests, notebook weight assertions
- Add TypeError coverage for Cluster.leaves non-Cluster children

### ⚙️ Miscellaneous Tasks

- Update uv.lock with pytest-benchmark dependencies
- Add fail_under = 90 to [tool.coverage.report] in pyproject.toml
- Add docstrings to benchmark tests and remove redundant mypy config
- Remove redundant mypy config from pyproject.toml
- Fix ruff target-version to py312, apply UP fixes, remove mypy config
- Tighten dependency version constraints
- Update uv.lock to reflect tightened dependency constraints
- Update uv.lock with latest dependency versions
- Numpy upper bound, doc line length, remove **kwargs, raise max_examples
- Auto-update CHANGELOG.md after each release via git-cliff
- Regenerate demo.png via book/marimo/demo.py
## [2.0.0] - 2026-05-20

### 🐛 Bug Fixes

- Import private _compute_* inside cell in hrp notebook
- Handle NaN in returns and assert weights by key in tests
- Restore Series-based Portfolio.weights API with dict accessor
- Revert Portfolio.weights to dict[str, float]
- Set requires-python to >=3.11 in hrp marimo script

### 💼 Other

- Bump version 1.6.4 → 2.0.0

### 🚜 Refactor

- Migrate from pandas/matplotlib to polars/plotly

### 📚 Documentation

- Update README example from pandas to polars

### 🎨 Styling

- Apply ruff format to hrp.py

### 🧪 Testing

- Assert portfolio weights by key instead of iteration order

### ⚙️ Miscellaneous Tasks

- Add requires-python constraint to marimo script
## [1.6.4] - 2026-05-20

### 🚀 Features

- Use cvx-linalg for portfolio variance computation

### 💼 Other

- Bump version 1.6.3 → 1.6.4

### ⚙️ Miscellaneous Tasks

- Update Rhiza template to v0.10.3
- Sync with Rhiza template v0.10.3
- Update via rhiza
- Update Rhiza template to v0.10.7
- Sync with Rhiza template v0.10.7
- Add github-book and github-marimo template bundles
- Sync github-book and github-marimo template bundles
- Update Rhiza template to v0.10.9
- Sync with Rhiza template v0.10.9
## [1.6.3] - 2026-04-23

### 🚀 Features

- Add mkdocs.yml with nav for docs, notebooks, and reports

### 🐛 Bug Fixes

- Suppress bandit B404 on intentional subprocess import
- Set MARIMO_FOLDER to book/marimo so notebooks are exported to HTML
- Correct coverage badge URL to GitHub Pages endpoint

### 💼 Other

- Bump version 1.6.2 → 1.6.3

### 📚 Documentation

- Add coverage badge to README
- Add API reference page with mkdocstrings for all modules

### 🧪 Testing

- Bring test coverage to 100%

### ⚙️ Miscellaneous Tasks

- Update rhiza template version to v0.9.5
- Sync with rhiza template v0.9.5
- Update rhiza template version to v0.10.2
- Sync rhiza template to v0.10.2
## [1.6.2] - 2026-03-22

### 💼 Other

- Bump version 1.6.1 → 1.6.2

### ⚙️ Miscellaneous Tasks

- Apply rhiza template v0.8.16 sync
## [1.6.1] - 2026-03-17

### 💼 Other

- Bump version 1.6.0 → 1.6.1

### ⚙️ Miscellaneous Tasks

- Rhiza sync to v0.11.6
- Update via rhiza
- Resolve merge conflicts from rhiza template sync
## [1.6.0] - 2026-02-24

### 🐛 Bug Fixes

- *(deps)* Update dependency polars to v1.35.1 (#483)
- *(deps)* Update dependency polars to v1.35.2 (#488)
- *(deps)* Update dependency polars to v1.36.0 (#517)
- *(deps)* Update dependency polars to v1.36.1
- Resolve ruff linting errors for TRY003 and PT011
- Resolve mypy type checking errors
- Replace assert statements with proper exception handling
- Resolve type checking errors in algos, cluster, and hrp modules
- Change Dendrogram.assets type from list[str] to pd.Index

### 💼 Other

- Bump version 1.5.1 → 1.6.0

### 📚 Documentation

- *(tests)* Document S101 security exception in conftest

### 🎨 Styling

- Split compound assertions into separate statements

### ⚙️ Miscellaneous Tasks

- Sync template files (#472)
- Sync template from tschm/.config-templates@main (#477)
- Sync template files (#481)
- Sync template files (#482)
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Remove deprecated files
- Import rhiza templates
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Add explicit package-module mappings for deptry
- Update via rhiza
## [1.5.0] - 2025-10-17

### 🐛 Bug Fixes

- *(deps)* Update dependency polars to v1.32.2 (#414)
- *(deps)* Update dependency polars to v1.32.3 (#420)
- *(deps)* Update dependency polars to v1.33.0 (#426)
- *(deps)* Update dependency polars to v1.33.1 (#433)
- *(deps)* Update dependency polars to v1.34.0 (#453)

### ⚙️ Miscellaneous Tasks

- Sync config files from .config-templates (#416)
- Sync config files from .config-templates (#418)
- Sync config files from .config-templates (#422)
- Sync config files from .config-templates (#425)
- Sync config files from .config-templates (#431)
- Sync config files from .config-templates (#435)
- Sync config files from .config-templates (#440)
- Sync template files (#446)
- Sync template files (#448)
- Sync template files (#449)
- Sync template files (#455)
## [1.4.0] - 2025-08-11

### ⚙️ Miscellaneous Tasks

- Sync config files from .config-templates (#390)
- Sync config files from .config-templates (#393)
- Sync config files from .config-templates (#395)
- Sync config files from .config-templates (#397)
- Sync config files from .config-templates (#398)
- Sync config files from .config-templates (#408)
- Sync config files from .config-templates (#409)
- Sync config files from .config-templates (#413)
## [1.3.6] - 2025-05-15

### 🐛 Bug Fixes

- *(deps)* Update dependency scipy to v1.15.3 (#331)

### ⚙️ Miscellaneous Tasks

- *(config)* Migrate config .github/renovate.json (#312)
## [0.0.1] - 2020-04-24
