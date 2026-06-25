# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [2.3.3] - 2026-06-25

### New Features
- Add ClusterFuzzLite fuzzing scaffold for pyhrp (#711)

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 5 updates
- Chore(deps)(deps): bump actions/checkout in the github-actions group

### Other Changes
- Bump rhiza template ref v0.19.3 → v0.19.4
- Sync Rhiza template v0.19.3 → v0.19.4
- Merge pull request #707 from tschm/sync/rhiza-v0.19.4
- Merge pull request #708 from tschm/dependabot/github_actions/github-actions-6a98abd9ac
- Merge branch 'main' into dependabot/uv/python-dependencies-9ea60807c1
- Merge pull request #709 from tschm/dependabot/uv/python-dependencies-9ea60807c1
- Cover remaining branches to reach 100% coverage
- Merge pull request #710 from tschm/test/100-percent-coverage
- Support Python 3.11 (#712)
- Extract plotly dendrogram rendering into pyhrp.plot (#713)

## [2.3.2] - 2026-06-16

### Maintenance
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump starlette from 1.1.0 to 1.3.1
- Chore(deps)(deps): bump python-multipart from 0.0.29 to 0.0.31
- Chore(deps)(deps): bump the github-actions group with 8 updates
- Add Rhiza Claude commands (/rhiza_quality, /rhiza_update)

### Other Changes
- Merge pull request #702 from tschm/dependabot/uv/python-dependencies-78f2ec1154
- Merge pull request #703 from tschm/dependabot/uv/starlette-1.3.1
- Merge pull request #704 from tschm/dependabot/uv/python-multipart-0.0.31
- Merge pull request #701 from tschm/dependabot/github_actions/github-actions-28fa30880a
- Merge pull request #700 from tschm/chore/add-rhiza-claude-commands
- Bump rhiza template ref v0.18.8 → v0.19.3
- Resolve rhiza v0.19.3 sync conflicts
- Exempt book/marimo notebooks from ANN ruff rules
- Add docstrings to _synthetic_prices helpers for 100% docs-coverage
- Add ANN2 return annotations (ruff --fix)
- Merge pull request #705 from tschm/sync/rhiza-v0.19.3
- Point mkdocstrings at src/ so the book builds without an editable install
- Merge pull request #706 from tschm/fix/book-mkdocstrings-src-paths
- Bump version 2.3.1 → 2.3.2

## [2.3.1] - 2026-06-10

### Other Changes
- Update git auth action to use specific version
- Bump version 2.3.0 → 2.3.1

## [2.3.0] - 2026-06-10

### Maintenance
- Chore(deps)(deps): bump the github-actions group with 8 updates
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Harden API, validate inputs, speed up allocation

### Other Changes
- Merge pull request #698 from tschm/dependabot/uv/python-dependencies-4b95ba7fc8
- Merge branch 'main' into dependabot/github_actions/github-actions-bbecf9262c
- Merge pull request #697 from tschm/dependabot/github_actions/github-actions-bbecf9262c
- Merge pull request #699 from tschm/fable
- Bump version 2.2.6 → 2.3.0

## [2.2.6] - 2026-06-08

### Bug Fixes
- Align pyproject.toml with rhiza structure tests
- Add explicit type annotations in Node.levels to satisfy ty
- Remove duplicate uses keys in workflow files

### Maintenance
- Upgrade actions/checkout to v6.0.2
- Remove lowest_deps.yml (now in rhiza_ci)
- Apply rhiza sync v0.18.4
- Add classifiers to pyproject.toml
- Add pip dependabot entry for .rhiza/requirements
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump the github-actions group with 9 updates
- Chore(deps)(deps): bump the python-dependencies group with 3 updates

### Other Changes
- Merge origin/main into rhiza_v0.17.0
- Merge pull request #690 from tschm/rhiza_v0.17.0
- Merge pull request #691 from tschm/rhiza_v0.18.4
- Merge pull request #693 from tschm/dependabot/uv/python-dependencies-f80f250a6f
- Merge pull request #695 from tschm/dependabot/uv/python-dependencies-adc89c0a44
- Merge branch 'main' into dependabot/github_actions/github-actions-65749369b6
- Merge pull request #694 from tschm/dependabot/github_actions/github-actions-65749369b6
- Merge pull request #696 from tschm/rhiza_v0.18.8
- Bump version 2.2.5 → 2.2.6

## [2.2.5] - 2026-05-27

### Bug Fixes
- Inline release jobs to fix PyPI Trusted Publishing with reusable workflows

### Other Changes
- Bump version 2.2.4 → 2.2.5

## [2.2.4] - 2026-05-27

### Bug Fixes
- Pass required tag input to rhiza release workflow

### Other Changes
- Bump version 2.2.3 → 2.2.4

## [2.2.3] - 2026-05-27

### Other Changes
- Merge branch 'main' into rhiza_v0.16.0
- Merge pull request #689 from tschm/rhiza_v0.16.0
- Bump version 2.2.2 → 2.2.3

## [2.2.2] - 2026-05-27

### Maintenance
- Relax dependency floors and update package description
- Add workflow to test with lowest-direct dependency floors

### Other Changes
- Bump version 2.2.1 → 2.2.2

## [2.2.1] - 2026-05-27

### New Features
- Add profiles: github-project to template.yml

### Bug Fixes
- Add uv install retry step to CI test job
- Remove duplicate update-changelog job from release workflow
- Use Generic[T] to keep Node subscriptable on older CI Python

### Maintenance
- Update CHANGELOG.md for v2.2.0
- Sync with rhiza v0.11.0
- Sync with rhiza v0.16.1
- Apply rhiza sync v0.15.3
- Remove blank line from .python-version
- Apply rhiza sync v0.16.0
- Apply rhiza sync v0.17.0
- Increase coverage to 100% by testing TypeError guard branches
- Remove cvx-linalg dependency

### Other Changes
- Merge branch 'main' into rhiza
- Merge pull request #679 from tschm/rhiza
- Copy rhiza_*.yml workflows from cvxrisk
- Sync .rhiza/tests from cvxrisk
- Merge pull request #681 from tschm/rhizaMove
- Merge pull request #686 from tschm/rhiza_v0.15.3
- Clean up template list in template.yml
- Merge pull request #687 from tschm/tschm-patch-1
- Merge pull request #688 from tschm/linalg
- Bump version 2.2.0 → 2.2.1

## [2.2.0] - 2026-05-22

### New Features
- Implement Schur Complementary Allocation (Peter Cotton, arXiv:2411.05807)

### Documentation
- Update README example to use public compute_cov / compute_corr
- List public helper APIs in hrp module docstring

### Maintenance
- Update CHANGELOG.md for v2.1.0
- Promote _compute_cov and _compute_corr to public API
- Update marimo notebooks to v0.23.6

### Other Changes
- Merge pull request #676 from tschm/public
- Merge branch 'main' into schur
- Merge pull request #677 from tschm/schur
- Merge branch 'main' into notebooks
- Merge pull request #678 from tschm/notebooks
- Bump version 2.1.0 → 2.2.0

## [2.1.0] - 2026-05-21

### Documentation
- Align README badges with jquantstats and rewrite SECURITY.md for pyhrp
- Add root contribution docs and changelog
- Expand README and add demo image generation script
- Use portable command path for demo generation
- Address review feedback in README and demo script
- Keep robust README image URL
- Add missing docstrings to public functions in demo.py
- Add inline script metadata header to demo.py
- Add Examples sections to hrp, build_tree, risk_parity, one_over_n

### Maintenance
- Add stress benchmarks and weekly regression workflow
- Update uv.lock with pytest-benchmark dependencies
- Add fail_under = 90 to [tool.coverage.report] in pyproject.toml
- Add docstrings to benchmark tests and remove redundant mypy config
- Remove redundant mypy config from pyproject.toml
- Avoid private API imports in demo script
- Add kaleido as dev dependency
- Add property-based and numerical edge-case HRP tests
- Tighten zip strictness in property test generators
- Fix ruff target-version to py312, apply UP fixes, remove mypy config
- Tighten dependency version constraints
- Update uv.lock to reflect tightened dependency constraints
- Update uv.lock with latest dependency versions
- Add direct helper tests and notebook output assertions
- Numpy upper bound, doc line length, remove **kwargs, raise max_examples
- Auto-update CHANGELOG.md after each release via git-cliff
- Benchmark assertions, 500/1000-asset stress tests, notebook weight assertions
- Regenerate demo.png via book/marimo/demo.py
- Add TypeError coverage for Cluster.leaves non-Cluster children

### Other Changes
- Initial plan
- Initial plan
- Merge pull request #664 from tschm/copilot/ci-enforce-minimum-coverage-threshold
- Merge branch 'main' into copilot/add-performance-stress-benchmarks
- Initial plan
- Make treelib Node generic and remove type ignores in core modules
- Merge branch 'main' into copilot/refactor-treelib-node-generic
- Finalize generic Node typing and targeted mypy overrides
- Merge pull request #665 from tschm/copilot/refactor-treelib-node-generic
- Merge branch 'main' into copilot/add-performance-stress-benchmarks
- Initial plan
- Refactor HRP helpers and define module exports
- Initial plan
- Merge branch 'main' into copilot/add-changelog-contributing-code-of-conduct
- Merge pull request #666 from tschm/copilot/add-changelog-contributing-code-of-conduct
- Merge branch 'main' into copilot/refactor-add-all-fix-type-hint
- Adjust leaves comment placement per review
- Handle empty _bisect_tree input and add test
- Merge pull request #667 from tschm/copilot/refactor-add-all-fix-type-hint
- Merge branch 'main' into copilot/add-performance-stress-benchmarks
- Initial plan
- Merge branch 'main' into copilot/expand-readme-with-motivation
- Merge branch 'main' into copilot/expand-readme-with-motivation
- Merge branch 'main' into copilot/expand-readme-with-motivation
- Merge branch 'main' into copilot/expand-readme-with-motivation
- Merge pull request #662 from tschm/copilot/expand-readme-with-motivation
- Merge branch 'main' into copilot/add-performance-stress-benchmarks
- Initial plan
- Merge branch 'main' into copilot/add-property-based-tests
- Merge branch 'main' into copilot/add-property-based-tests
- Merge branch 'main' into copilot/add-property-based-tests
- Initial plan
- Merge branch 'main' into copilot/chore-fix-ruff-target-version
- Merge pull request #669 from tschm/copilot/chore-fix-ruff-target-version
- Merge branch 'main' into copilot/add-property-based-tests
- Initial plan
- Merge pull request #671 from tschm/copilot/tighten-dependency-version-constraints
- Merge branch 'main' into copilot/add-property-based-tests
- Initial plan
- Merge branch 'main' into copilot/add-direct-unit-tests-compute-cov-corr
- Merge pull request #670 from tschm/copilot/add-direct-unit-tests-compute-cov-corr
- Merge branch 'main' into copilot/add-property-based-tests
- Merge pull request #668 from tschm/copilot/add-property-based-tests
- Merge branch 'main' into copilot/add-performance-stress-benchmarks
- Add 'benchmark' job to expected jobs in tests
- Merge pull request #654 from tschm/copilot/add-performance-stress-benchmarks
- Merge pull request #673 from tschm/copilot/plan-docstring-examples
- Merge branch 'main' into copilot/plan-trivial-fixes
- Merge pull request #675 from tschm/copilot/plan-changelog-automation
- Merge branch 'main' into copilot/plan-trivial-fixes
- Merge branch 'main' into copilot/plan-test-improvements
- Merge branch 'main' into copilot/plan-test-improvements
- Merge pull request #674 from tschm/copilot/plan-test-improvements
- Merge branch 'main' into copilot/plan-trivial-fixes
- Merge pull request #672 from tschm/copilot/plan-trivial-fixes
- Bump version 2.0.0 → 2.1.0

## [2.0.0] - 2026-05-20

### Bug Fixes
- Import private _compute_* inside cell in hrp notebook
- Handle NaN in returns and assert weights by key in tests
- Restore Series-based Portfolio.weights API with dict accessor
- Revert Portfolio.weights to dict[str, float]
- Set requires-python to >=3.11 in hrp marimo script

### Documentation
- Update README example from pandas to polars

### Maintenance
- Migrate from pandas/matplotlib to polars/plotly
- Apply ruff format to hrp.py
- Assert portfolio weights by key instead of iteration order
- Add requires-python constraint to marimo script

### Other Changes
- Potential fix for pull request finding
- Potential fix for pull request finding
- Merge pull request #649 from tschm/polars
- Merge branch 'main' into polars
- Merge pull request #650 from tschm/polars
- Bump version 1.6.4 → 2.0.0

## [1.6.4] - 2026-05-20

### New Features
- Use cvx-linalg for portfolio variance computation

### Maintenance
- Update Rhiza template to v0.10.3
- Sync with Rhiza template v0.10.3
- Update via rhiza
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates
- Chore(deps-dev)(deps-dev): bump marimo in the python-dependencies group
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump urllib3 from 2.6.3 to 2.7.0
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump numpy in the python-dependencies group
- Chore(deps)(deps): bump idna from 3.11 to 3.15
- Chore(deps)(deps): bump pymdown-extensions from 10.21.2 to 10.21.3
- Update Rhiza template to v0.10.7
- Sync with Rhiza template v0.10.7
- Add github-book and github-marimo template bundles
- Sync github-book and github-marimo template bundles
- Update Rhiza template to v0.10.9
- Sync with Rhiza template v0.10.9

### Other Changes
- Merge pull request #636 from tschm/rhiza/24970578129
- Merge pull request #637 from tschm/dependabot/uv/python-dependencies-4a4f34f005
- Merge pull request #639 from tschm/dependabot/uv/python-dependencies-189a206c4b
- Merge pull request #638 from tschm/dependabot/github_actions/github-actions-937d73b4db
- Merge pull request #640 from tschm/dependabot/uv/urllib3-2.7.0
- Merge pull request #641 from tschm/dependabot/github_actions/github-actions-8abaa2cbc6
- Merge pull request #642 from tschm/dependabot/uv/python-dependencies-f78d620da7
- Merge pull request #643 from tschm/dependabot/github_actions/github-actions-bcb0c4251a
- Merge pull request #644 from tschm/dependabot/uv/python-dependencies-8cd47ffab5
- Merge pull request #645 from tschm/dependabot/uv/idna-3.15
- Merge pull request #646 from tschm/dependabot/uv/pymdown-extensions-10.21.3
- Merge branch 'main' into update-rhiza-v0.10.3
- Merge pull request #647 from tschm/update-rhiza-v0.10.3
- Merge pull request #648 from tschm/rhiza
- Bump version 1.6.3 → 1.6.4

## [1.6.3] - 2026-04-23

### New Features
- Add mkdocs.yml with nav for docs, notebooks, and reports

### Bug Fixes
- Suppress bandit B404 on intentional subprocess import
- Set MARIMO_FOLDER to book/marimo so notebooks are exported to HTML
- Correct coverage badge URL to GitHub Pages endpoint

### Documentation
- Add coverage badge to README
- Add API reference page with mkdocstrings for all modules

### Maintenance
- Chore(deps)(deps): bump requests from 2.32.5 to 2.33.0
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump pygments from 2.19.2 to 2.20.0
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump numpy in the python-dependencies group
- Chore(deps)(deps): bump docker/login-action in the github-actions group
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Chore(deps-dev)(deps-dev): bump marimo from 0.22.4 to 0.23.0
- Chore(deps)(deps): bump pillow from 12.1.1 to 12.2.0
- Update rhiza template version to v0.9.5
- Sync with rhiza template v0.9.5
- Chore(deps-dev)(deps-dev): bump marimo in the python-dependencies group
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump the github-actions group with 2 updates
- Update rhiza template version to v0.10.2
- Sync rhiza template to v0.10.2
- Bring test coverage to 100%

### Other Changes
- Initial plan
- Update README for badge and wording improvements
- Merge branch 'main' into copilot/create-coverage-badge
- Merge pull request #612 from tschm/copilot/create-coverage-badge
- Merge pull request #614 from tschm/dependabot/uv/python-dependencies-4852de8d5b
- Merge branch 'main' into dependabot/uv/requests-2.33.0
- Merge pull request #615 from tschm/dependabot/uv/requests-2.33.0
- Merge pull request #617 from tschm/dependabot/uv/pygments-2.20.0
- Merge pull request #619 from tschm/dependabot/uv/python-dependencies-1670832c3b
- Merge branch 'main' into dependabot/github_actions/github-actions-fd00acb19b
- Merge pull request #618 from tschm/dependabot/github_actions/github-actions-fd00acb19b
- Merge pull request #623 from tschm/dependabot/uv/python-dependencies-111c77e3aa
- Merge branch 'main' into dependabot/github_actions/github-actions-cb5fd4910d
- Merge pull request #622 from tschm/dependabot/github_actions/github-actions-cb5fd4910d
- Merge pull request #624 from tschm/dependabot/uv/marimo-0.23.0
- Merge pull request #626 from tschm/dependabot/uv/pillow-12.2.0
- Merge pull request #627 from tschm/RRR
- Merge pull request #628 from tschm/dependabot/uv/python-dependencies-8656732d5a
- Merge pull request #631 from tschm/dependabot/github_actions/github-actions-451fd88063
- Merge branch 'main' into dependabot/uv/python-dependencies-020d7024b2
- Merge pull request #632 from tschm/dependabot/uv/python-dependencies-020d7024b2
- Merge pull request #633 from tschm/dependabot/pip/dot-rhiza/requirements/pip-4ea199e985
- Merge pull request #634 from tschm/rhiza2
- Merge pull request #635 from tschm/test
- Delete docs/development directory
- Update mkdocs.yml
- Bump version 1.6.2 → 1.6.3

## [1.6.2] - 2026-03-22

### Maintenance
- Apply rhiza template v0.8.16 sync

### Other Changes
- Update template branch to v0.8.16
- Add license information to pyproject.toml
- Merge pull request #609 from tschm/tschm-patch-1
- Merge branch 'main' into tschm-patch-2
- Merge pull request #610 from tschm/tschm-patch-2
- Bump version 1.6.1 → 1.6.2

## [1.6.1] - 2026-03-17

### Maintenance
- Chore(deps)(deps): bump the github-actions group with 2 updates
- Chore(deps-dev)(deps-dev): bump pyportfolioopt
- Rhiza sync to v0.11.6
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Update via rhiza
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump the github-actions group across 1 directory with 3 updates
- Resolve merge conflicts from rhiza template sync

### Other Changes
- Update template branch to v0.8.5
- Sync
- Merge pull request #598 from tschm/tschm-patch-100
- Merge pull request #600 from tschm/dependabot/uv/python-dependencies-76500248a0
- Merge branch 'main' into dependabot/github_actions/github-actions-aa99a42152
- Merge pull request #599 from tschm/dependabot/github_actions/github-actions-aa99a42152
- Legal back
- Update rhiza
- Sync
- Merge pull request #602 from tschm/dependabot/uv/python-dependencies-41f8a01e75
- Merge pull request #603 from tschm/rhiza/23122486138
- Merge branch 'main' into syncRhiza
- Merge pull request #606 from tschm/dependabot/uv/python-dependencies-a6b05f733b
- Merge branch 'main' into syncRhiza
- Merge pull request #604 from tschm/syncRhiza
- Update template branch version to v0.8.13
- Merge pull request #607 from tschm/tschm-patch-160
- Merge branch 'main' into dependabot/github_actions/github-actions-e7fb33d53e
- Merge pull request #608 from tschm/dependabot/github_actions/github-actions-e7fb33d53e
- Delete .github/workflows/rhiza_benchmarks.yml
- Delete tests/benchmarks directory
- Remove benchmark workflow and test files
- Bump version 1.6.0 → 1.6.1

## [1.6.0] - 2026-02-24

### Bug Fixes
- *(deps)* Update dependency polars to v1.35.1 (#483)
- *(deps)* Update dependency polars to v1.35.2 (#488)
- Fixing README
- *(deps)* Update dependency polars to v1.36.0 (#517)
- *(deps)* Update dependency polars to v1.36.1
- Resolve ruff linting errors for TRY003 and PT011
- Resolve mypy type checking errors
- Replace assert statements with proper exception handling
- Resolve type checking errors in algos, cluster, and hrp modules
- Change Dendrogram.assets type from list[str] to pd.Index

### Documentation
- *(tests)* Document S101 security exception in conftest

### Dependencies
- *(deps)* Lock file maintenance (#471)
- *(deps)* Update dependency python to 3.14 (#473)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.2 (#475)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.5 (#474)
- *(deps)* Lock file maintenance (#476)
- *(deps)* Lock file maintenance (#479)
- *(deps)* Update github artifact actions (#478)
- *(deps)* Lock file maintenance (#480)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.4 (#484)
- *(deps)* Lock file maintenance (#485)
- *(deps)* Lock file maintenance (#486)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.35.0 (#489)
- *(deps)* Lock file maintenance (#490)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.11 (#492)
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.9
- *(deps)* Update pre-commit hook igorshubovych/markdownlint-cli to v0.46.0
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.6
- *(deps)* Update actions/checkout action to v6
- *(deps)* Lock file maintenance (#499)
- *(deps)* Lock file maintenance (#505)
- *(deps)* Update softprops/action-gh-release action to v2.5.0
- *(deps)* Lock file maintenance
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.14
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.15 (#511)
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance (#513)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.16 (#515)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.8 (#516)
- *(deps)* Update pre-commit hook igorshubovych/markdownlint-cli to v0.47.0
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.17
- *(deps)* Lock file maintenance (#523)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.0 (#525)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.10 (#527)
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance (#529)
- *(deps)* Lock file maintenance (#530)
- *(deps)* Update dependency astral-sh/uv to v0.9.20 (#532)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.20 (#533)
- *(deps)* Lock file maintenance (#536)
- *(deps)* Update dependency plotly to v6
- *(deps)* Update dependency astral-sh/uv to v0.9.22 (#539)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.22 (#540)
- *(deps)* Lock file maintenance (#541)
- *(deps)* Update dependency polars to v1.37.1 (#543)
- *(deps)* Lock file maintenance (#545)
- *(deps)* Update dependency astral-sh/uv to v0.9.26 (#547)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.13
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.26
- *(deps)* Lock file maintenance (#550)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.1 (#552)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.27
- *(deps)* Update dependency astral-sh/uv to v0.9.27
- *(deps)* Lock file maintenance (#555)
- *(deps)* Update github/codeql-action action to v4.32.1 (#557)
- *(deps)* Update pre-commit hook abravalheri/validate-pyproject to v0.25
- *(deps)* Lock file maintenance (#559)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.30 (#563)
- *(deps)* Update dependency astral-sh/uv to v0.10.0
- *(deps)* Update astral-sh/setup-uv action to v7.3.0
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.0
- *(deps)* Update dependency polars to v1.38.1
- *(deps)* Update github/codeql-action action to v4.32.2
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.11 (#572)
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance (#574)
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.0
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.2
- *(deps)* Update actions/download-artifact action to v7
- *(deps)* Update dependency astral-sh/uv to v0.10.3 (#579)
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.3
- *(deps)* Update actions/download-artifact action to v7
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.2
- *(deps)* Update dependency astral-sh/uv to v0.10.4 (#584)
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.4
- *(deps)* Update github/codeql-action action to v4.32.4 (#586)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.2 (#587)
- *(deps)* Lock file maintenance (#588)
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.5
- *(deps)* Lock file maintenance
- *(deps)* Update dependency astral-sh/uv to v0.10.5
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.3

### Maintenance
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
- Chore(deps)(deps): bump actions/checkout from 4 to 6
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Split compound assertions into separate statements
- Add explicit package-module mappings for deptry
- Update via rhiza
- Chore(deps)(deps): bump pillow from 12.1.0 to 12.1.1
- Chore(deps-dev)(deps-dev): bump pyarrow in the python-dependencies group
- Chore(deps)(deps): bump scipy in the python-dependencies group

### Other Changes
- CodeRabbit Generated Unit Tests: Expand test_dendrogram_extras with 27 new unit tests (#469)
- Deptry (#470)
- Merge pull request #487 from tschm/template-updates
- Merge pull request #491 from tschm/template-updates
- Update README.md
- Delete tests/test_docs.py
- Merge pull request #494 from tschm/renovate/rhysd-actionlint-1.x
- Merge pull request #495 from tschm/renovate/igorshubovych-markdownlint-cli-0.x
- Merge pull request #493 from tschm/renovate/astral-sh-ruff-pre-commit-0.x
- Merge pull request #498 from tschm/template-updates
- Merge branch 'main' into renovate/actions-checkout-6.x
- Merge branch 'main' into renovate/actions-checkout-6.x
- Merge pull request #496 from tschm/renovate/actions-checkout-6.x
- Remove tests/test_taskfile.py Taskfile.yml taskfiles
- Merge pull request #504 from tschm/template-updates
- Merge branch 'main' into remove-file-6
- Merge pull request #503 from tschm/remove-file-6
- Delete .github/workflows/devcontainer.yml
- Update .github/template.yml with new configuration
- Add exclusions for specific workflow files
- Merge pull request #507 from tschm/tschm-patch-1
- Merge branch 'main' into template-updates
- Merge pull request #506 from tschm/template-updates
- Merge pull request #508 from tschm/renovate/softprops-action-gh-release-2.x
- Merge pull request #509 from tschm/renovate/lock-file-maintenance
- Merge pull request #510 from tschm/renovate/ghcr.io-astral-sh-uv-0.x
- Merge pull request #512 from tschm/renovate/lock-file-maintenance
- Add pytest.ini to the template file
- Delete tests/test_makefile.py
- Delete tests/test_readme.py
- Merge pull request #520 from tschm/renovate/igorshubovych-markdownlint-cli-0.x
- Merge branch 'main' into renovate/polars-1.x
- Merge pull request #519 from tschm/renovate/polars-1.x
- Merge pull request #518 from tschm/renovate/ghcr.io-astral-sh-uv-0.x
- Update template repository in template.yml
- Merge pull request #521 from tschm/template-updates
- Remove comment about repository from LICENSE file
- Add dependencies and tool configuration comments
- Add project dependencies and configuration comments
- Update 1_over_N.py
- Comment out unused dependencies in hrp.py
- Merge pull request #522 from tschm/template-updates
- Delete .github/workflows/_devcontainer.yml
- Delete .github/workflows/devcontainer.yml
- Delete .github/workflows/docker.yml
- Delete .github/workflows/structure.yml
- Delete .github/scripts/build-extras.sh
- Fix formatting of include and exclude lists
- Rhiza
- Delete .github/scripts/sync.sh
- Merge pull request #526 from tschm/cleanup/delete-files
- Merge pull request #528 from tschm/renovate/lock-file-maintenance
- Rhiza
- Missing makefile
- Add CodeQL analysis workflow configuration
- Add 'book' to the template.yml include list
- Remove old rhiza
- Book makefile
- Rhiza
- Rhiza
- Merge pull request #531 from tschm/rhiza/20561403977
- Merge pull request #535 from tschm/dependabot/github_actions/actions/checkout-6
- Dotenv
- Rhiza
- Rhiza
- Sync
- Dependencies
- Plotly missing
- Merge pull request #537 from tschm/rhiza/20701163635
- Merge pull request #538 from tschm/renovate/plotly-6.x
- Sync
- Merge pull request #542 from tschm/rhiza/20904090014
- Delete .rhiza.env
- Merge pull request #544 from tschm/tschm-patch-1
- Merge pull request #549 from tschm/renovate/astral-sh-ruff-pre-commit-0.x
- Merge pull request #548 from tschm/renovate/ghcr.io-astral-sh-uv-0.x
- Update workflow exclusions in template.yml
- Merge branch 'main' into rhiza/21341903194
- Merge pull request #551 from tschm/rhiza/21341903194
- Merge pull request #554 from tschm/renovate/ghcr.io-astral-sh-uv-0.x
- Merge pull request #553 from tschm/renovate/astral-sh-uv-0.x
- Sync
- Missing __init__ in test_rhiza
- Merge branch 'main' into rhiza/21572790754
- Merge pull request #556 from tschm/rhiza/21572790754
- Sync
- Merge pull request #558 from tschm/renovate/abravalheri-validate-pyproject-0.x
- Delete .github/workflows/rhiza_benchmarks.yml
- Delete .github/workflows/codeql.yml
- Remove 'tests' from template inclusion
- Merge pull request #562 from tschm/tschm-patch-2
- Delete tests/test_rhiza directory
- Merge pull request #561 from tschm/tschm-patch-1
- Merge pull request #566 from tschm/renovate/astral-sh-uv-0.x
- Merge pull request #565 from tschm/renovate/astral-sh-setup-uv-7.x
- Merge pull request #568 from tschm/renovate/astral-sh-ruff-pre-commit-0.x
- Merge pull request #564 from tschm/renovate/github-codeql-action-4.x
- Merge branch 'main' into renovate/polars-1.x
- Merge pull request #567 from tschm/renovate/polars-1.x
- Update template.yml
- Sync
- Lfs tests
- Sync
- Merge pull request #569 from tschm/tschm-patch-1
- Merge pull request #570 from tschm/dependabot/uv/pillow-12.1.1
- Merge branch 'main' into sync22
- Merge pull request #571 from tschm/sync22
- Merge pull request #573 from tschm/renovate/lock-file-maintenance
- Merge pull request #576 from tschm/renovate/python-jsonschema-check-jsonschema-0.x
- Merge branch 'main' into renovate/jebel-quant-rhiza-0.x
- Merge branch 'main' into renovate/major-github-artifact-actions
- Merge pull request #578 from tschm/renovate/major-github-artifact-actions
- Merge branch 'main' into renovate/jebel-quant-rhiza-0.x
- Sync
- Merge pull request #577 from tschm/renovate/jebel-quant-rhiza-0.x
- Merge pull request #580 from tschm/renovate/astral-sh-uv-pre-commit-0.x
- Merge pull request #581 from tschm/dependabot/uv/python-dependencies-cd7a56d8b1
- Merge pull request #582 from tschm/renovate/python-jsonschema-check-jsonschema-0.x
- Merge branch 'main' into renovate/major-github-artifact-actions
- Merge pull request #583 from tschm/renovate/major-github-artifact-actions
- Merge pull request #585 from tschm/renovate/astral-sh-uv-pre-commit-0.x
- Merge pull request #591 from tschm/dependabot/uv/python-dependencies-3a2038e1d7
- Merge pull request #594 from tschm/renovate/lock-file-maintenance
- Merge branch 'main' into renovate/astral-sh-uv-pre-commit-0.x
- Merge pull request #593 from tschm/renovate/astral-sh-uv-pre-commit-0.x
- Merge pull request #592 from tschm/renovate/astral-sh-uv-0.x
- Merge pull request #595 from tschm/renovate/jebel-quant-rhiza-0.x
- Sync
- Clean up README by removing empty result sections
- Merge pull request #596 from tschm/tschm-patch-1
- Version number
- Bump version 1.5.1 → 1.6.0

## [1.5.1] - 2025-10-17

### Other Changes
- Remove Coverage Status badge
- Update Python version badge to 3.12+
- Remove PyPI Status badge from README
- Arrange badges
- Fix formatting in README.md
- Additional tests (#468)

## [1.5.0] - 2025-10-17

### Bug Fixes
- *(deps)* Update dependency polars to v1.32.2 (#414)
- *(deps)* Update dependency polars to v1.32.3 (#420)
- *(deps)* Update dependency polars to v1.33.0 (#426)
- *(deps)* Update dependency polars to v1.33.1 (#433)
- *(deps)* Update dependency polars to v1.34.0 (#453)

### Dependencies
- *(deps)* Update actions/checkout action to v5 (#415)
- *(deps)* Lock file maintenance (#417)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.3 (#419)
- *(deps)* Lock file maintenance (#421)
- *(deps)* Update actions/upload-pages-artifact action to v4 (#423)
- *(deps)* Lock file maintenance (#424)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.12 (#427)
- *(deps)* Update softprops/action-gh-release action to v2.3.3 (#428)
- *(deps)* Lock file maintenance (#429)
- *(deps)* Lock file maintenance (#430)
- *(deps)* Lock file maintenance (#432)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.0 (#434)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.34.0 (#437)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.1 (#436)
- *(deps)* Lock file maintenance (#438)
- *(deps)* Lock file maintenance (#439)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.2 (#442)
- *(deps)* Lock file maintenance (#444)
- *(deps)* Lock file maintenance (#447)
- *(deps)* Lock file maintenance (#450)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.3 (#451)
- *(deps)* Update softprops/action-gh-release action to v2.4.0 (#452)
- *(deps)* Lock file maintenance (#454)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.34.1 (#456)
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.8 (#457)
- *(deps)* Update softprops/action-gh-release action to v2.4.1 (#458)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.0 (#459)
- *(deps)* Update astral-sh/setup-uv action to v7 (#460)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.1 (#464)
- *(deps)* Lock file maintenance (#465)
- *(deps)* Lock file maintenance (#466)

### Maintenance
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

### Other Changes
- Delete taskfiles directory
- Delete .github/taskfiles directory
- Delete .github/CODE_OF_CONDUCT.md
- Delete .github/CONTRIBUTING.md
- Refactor sync workflow to use new template sync action (#441)
- Delete .devcontainer directory
- Create template.yml for GitHub repository setup (#443)
- Refactor sync.yml for permissions and template sync (#445)
- Change template branch from '77-hot' to 'main'
- Potential fix for code scanning alert no. 14: Workflow does not contain permissions (#461)
- Fix link formatting in README.md
- 462 fix broken notebook (#463)

## [1.4.0] - 2025-08-11

### Bug Fixes
- Fixing notebooks?
- Fixing notebooks?
- Fixing notebooks?

### Dependencies
- *(deps)* Update tschm/cradle action to v0.1.73 (#377)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.34.0 (#379)
- *(deps)* Update tschm/minibook action to v0.0.16 (#378)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.2 (#380)
- *(deps)* Update tschm/cradle action to v0.1.75 (#381)
- *(deps)* Lock file maintenance (#382)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.2 (#384)
- *(deps)* Update jebel-quant/marimushka action to v0.1.4 (#383)
- *(deps)* Lock file maintenance (#386)
- *(deps)* Update tschm/cradle action to v0.2.1 (#385)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.3 (#387)
- *(deps)* Update tschm/cradle action to v0.3.01 (#388)
- *(deps)* Lock file maintenance (#389)
- *(deps)* Update peter-evans/create-pull-request action to v7 (#392)
- *(deps)* Lock file maintenance (#394)
- *(deps)* Update tschm/.config-templates action to v0.1.6 (#396)
- *(deps)* Lock file maintenance (#399)
- *(deps)* Lock file maintenance (#400)
- *(deps)* Lock file maintenance (#401)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.7 (#402)
- *(deps)* Update tschm/.config-templates action to v0.1.8 (#403)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.35.1 (#404)
- *(deps)* Update tschm/.config-templates action to v0.2.0 (#405)
- *(deps)* Update actions/download-artifact action to v5 (#406)
- *(deps)* Lock file maintenance (#407)

### Maintenance
- Sync config files from .config-templates (#390)
- Sync config files from .config-templates (#393)
- Sync config files from .config-templates (#395)
- Sync config files from .config-templates (#397)
- Sync config files from .config-templates (#398)
- Testing the treelib
- Sync config files from .config-templates (#408)
- Sync config files from .config-templates (#409)
- Sync config files from .config-templates (#413)

### Other Changes
- Updates
- Treelib aftermath
- Update book.yml
- Update release.yml
- Added update
- Adding update
- Template
- Env
- Delete .github/workflows/marimo.yml
- Update via action
- Update sync.yml
- Add python-dotenv
- Update sync.yml
- Fmt
- Delete .devcontainer/startup.sh
- Moving tests
- Remove Script section
- Remove test_env
- Remove Makefile
- Update README

## [1.3.9] - 2025-06-28

### Dependencies
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.1 (#373)
- *(deps)* Update jebel-quant/marimushka action to v0.1.2 (#376)

### Other Changes
- Update book.yml
- Marimushka
- Update book.yml
- Marimushka
- Update book.yml
- Treelib (#375)

## [1.3.8] - 2025-06-26

### Bug Fixes
- Fix marimo in book

### Dependencies
- *(deps)* Lock file maintenance (#366)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.0 (#367)
- *(deps)* Lock file maintenance (#368)
- *(deps)* Lock file maintenance (#369)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.1 (#370)

### Other Changes
- Update Makefile (#362)
- 363 move marimo (#364)
- 363 move marimo (#365)
- Update book.yml
- Update _toc.yml
- Update book.yml
- First book?
- Remove old book
- Fmt
- Remove devcontainer from renovate
- Dependencies with renovate?
- Dependencies with renovate?
- Marimushka
- Marimushka

## [1.3.7] - 2025-06-11

### Dependencies
- *(deps)* Update tschm/cradle action to v0.1.64 (#334)
- *(deps)* Lock file maintenance (#335)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.10 (#336)
- *(deps)* Update pre-commit hook igorshubovych/markdownlint-cli to v0.45.0 (#337)
- *(deps)* Lock file maintenance (#338)
- *(deps)* Update tschm/cradle action to v0.1.68 (#339)
- *(deps)* Lock file maintenance (#340)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.11 (#341)
- *(deps)* Update pre-commit hook asottile/pyupgrade to v3.20.0 (#342)
- *(deps)* Update tschm/cradle action to v0.1.69 (#344)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.12 (#346)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.33.1 (#349)
- *(deps)* Update tschm/cradle action to v0.1.71 (#350)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.13 (#351)
- *(deps)* Lock file maintenance (#352)
- *(deps)* Lock file maintenance (#353)
- *(deps)* Update tschm/cradle action to v0.1.72 (#358)

### Other Changes
- Update README.md (#343)
- Update __init__.py (#345)
- Update pyproject.toml (#347)
- Update book.yml (#348)
- Update book.yml
- Update book.yml
- Update book.yml (#354)
- Potential fix for code scanning alert no. 10: Workflow does not contain permissions (#355)
- Potential fix for code scanning alert no. 12: Workflow does not contain permissions (#356)
- Remove devcontainer (#360)

## [1.3.6] - 2025-05-15

### Bug Fixes
- *(deps)* Update dependency scipy to v1.15.3 (#331)

### Dependencies
- *(deps)* Update pre-commit hook crate-ci/typos to v1.31.1 (#306)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.3 (#307)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.4 (#308)
- *(deps)* Lock file maintenance (#309)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.5 (#310)
- *(deps)* Lock file maintenance (#311)
- *(deps)* Lock file maintenance (#314)
- *(deps)* Update tschm/cradle action to v0.1.60 (#315)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.0 (#316)
- *(deps)* Lock file maintenance (#317)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.6 (#320)
- *(deps)* Lock file maintenance (#321)
- *(deps)* Lock file maintenance (#322)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.7 (#323)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.31.2 (#324)
- *(deps)* Update tschm/cradle action to v0.1.63 (#325)
- *(deps)* Lock file maintenance (#326)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.8 (#327)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.32.0 (#328)
- *(deps)* Lock file maintenance (#329)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.9 (#330)

### Maintenance
- *(config)* Migrate config .github/renovate.json (#312)
- Ci/cd (#318)

### Other Changes
- Lock file maintenance (#298)
- Update pyproject.toml (#299)
- Update tschm/cradle action to v0.1.58 (#300)
- Update pre-commit hooks (#302)
- Update tschm/cradle action to v0.1.59 (#301)
- Update renovate.json (#303)
- Lock file maintenance (#304)
- Lock file maintenance (#305)
- Update renovate.json
- Security as dev dependency
- Security in test
- Update README.md
- Update README.md
- Workflows (#332)
- Testing (#333)
- Update release.yml

## [1.3.5] - 2025-03-26

### Maintenance
- Ci/cd
- Ci/cd
- Ci/cd

### Other Changes
- Lock file maintenance (#282)
- Lock file maintenance (#283)
- Age of dependencies
- Age of dependencies
- Update reports.md
- Update reports.md
- Update book.yml
- Update pre-commit hooks (#284)
- Lock file maintenance (#285)
- Update cvxgrp/.github action to v2.2.6 (#286)
- Update cvxgrp/.github action to v2.2.7 (#287)
- Lock file maintenance (#288)
- Lock file maintenance (#289)
- Update cvxgrp/.github action to v2.2.8 (#290)
- Lock file maintenance (#291)
- Lock file maintenance (#292)
- Lock file maintenance (#293)
- Update tschm/cradle action to v0.1.56 (#294)
- Update tschm/cradle action to v0.1.57 (#295)
- Update tschm/cradle action to v0.1.57 (#296)
- Update tschm/cradle action to v0.1.57 (#297)

## [1.3.4] - 2025-03-02

### Other Changes
- Lock file maintenance (#280)
- Update pre-commit hooks (#281)
- Update dependency scipy to v1.15.2 (#279)

## [1.3.3] - 2025-02-24

### Bug Fixes
- Fix version for ci/cd

### Other Changes
- Marimo as dev dependency
- Update README.md
- Update cvxgrp/.github action to v2.2.4 (#276)
- Update cvxgrp/.github action to v2.2.5 (#278)
- Lock file maintenance (#277)

## [1.3.2] - 2025-02-15

### Other Changes
- Update pyproject.toml

## [1.3.1] - 2025-02-15

### Maintenance
- Testing notebooks
- Testing notebooks (#272)

### Other Changes
- Update README.md
- 270 fix comments in 1n notebook (#271)
- Improve README
- Improve README (#273)

## [1.2.41] - 2025-02-15

### Maintenance
- Testing README (#269)

### Other Changes
- More testing and features for the dendrogram (#263)
- Update README.md
- No longer return dictionary, use yield (#265)
- Update README.md
- Update pre-commit hooks (#266)
- Lock file maintenance (#267)

## [1.2.40] - 2025-02-14

### Other Changes
- Notebook for hrp (#253)
- Update README.md
- Support plot of portfolio weights (#254)
- Graph for README
- Update README.md
- Lock file maintenance (#258)
- 257 try with a generic function executed (#259)
- 260 generic functions (#261)

## [1.2.39] - 2025-02-12

### Other Changes
- 249 bring in binarytree (#250)
- Removing commented code
- Fixing README
- 248 one over n 2 (#251)

## [1.2.38] - 2025-02-11

### Maintenance
- Testing with windows (#219)
- Testing with pyportfolioopt (#230)

### Other Changes
- Update cvxgrp/.github action to v2.1.1 (#208)
- Update ci.yml (#210)
- Update release.yml (#209)
- Update cvxgrp/.github action to v2.1.2 (#211)
- Update pre-commit.yml (#212)
- Update book.yml (#214)
- Update ci.yml (#215)
- Update cvxgrp/.github action to v2.2.1 (#213)
- Lock file maintenance (#216)
- Update cvxgrp/.github action to v2.2.3 (#220)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.9.5 (#221)
- Update mcr.microsoft.com/devcontainers/python Docker tag to v3.13 (#222)
- Lock file maintenance (#223)
- 224 bring src structure (#225)
- 226 introduce data for marimo (#227)
- README updates
- Update marimo.md
- Fmt
- 231 test with pyportfolioopt (#232)
- Remove mock and loguru (#234)
- Remove _linkage (#236)
- 228 remove node class2 (#237)
- Removing the n argument for the bisection
- Lock file maintenance (#239)
- Removing the n argument for the bisection
- Removing the Node class (#241)
- 242 revisit cluster (#243)
- Cleaning tests
- Revisit README
- Hrp notebook
- Hrp demo
- 244 revisit node to linkage (#245)
- Dependencies
- Dendrogram with bisection
- Introducing algos (#247)

## [1.2.37] - 2025-02-02

### Other Changes
- Update release.yml
- Update cvxgrp/.github action to v2.0.16 (#199)
- Update cvxgrp/.github action to v2.0.17 (#200)
- Update pre-commit.yml (#201)
- Update release.yml (#203)
- Update pre-commit.yml (#204)
- Update ci.yml (#205)
- Update book.yml (#206)
- Update cvxgrp/.github action to v2.1.0 (#202)

## [1.2.36] - 2025-02-02

### Other Changes
- Update release.yml
- Delete .python-version (#197)
- Create Makefile (#198)
- Update release.yml

## [1.2.35] - 2025-02-01

### Other Changes
- Update cvxgrp/.github action to v2.0.13 (#196)

## [1.2.34] - 2025-01-31

### Other Changes
- Update renovate.json (#195)

## [1.2.33] - 2025-01-31

### Other Changes
- Update pre-commit.yml (#194)

## [1.2.32] - 2025-01-31

### Other Changes
- Update pre-commit hook crate-ci/typos to v1.29.5 (#193)

## [1.2.31] - 2025-01-31

### Other Changes
- Update renovate.json

## [1.2.30] - 2025-01-30

### Other Changes
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.9.4 (#191)

## [1.2.29] - 2025-01-30

### Other Changes
- Update pre-commit hook python-jsonschema/check-jsonschema to v0.31.1 (#190)

## [1.2.28] - 2025-01-30

### Other Changes
- Update cvxgrp/.github action to v2.0.12 (#189)

## [1.2.27] - 2025-01-29

### Other Changes
- Update cvxgrp/.github action to v2.0.11 (#188)

## [1.2.26] - 2025-01-29

### Other Changes
- Update cvxgrp/.github action to v2.0.10 (#187)

## [1.2.25] - 2025-01-28

### Other Changes
- Update ci.yml

## [1.2.24] - 2025-01-27

### Other Changes
- Update cvxgrp/.github action to v2.0.9 (#185)

## [1.2.18] - 2025-01-27

### Other Changes
- Update release.yml

## [1.2.17] - 2025-01-27

### Other Changes
- Update cvxgrp/.github action to v2.0.8 (#184)

## [1.2.10] - 2025-01-27

### Other Changes
- Tag job

## [1.2.9] - 2025-01-27

### Other Changes
- License and automated release
- License and automated release

## [1.2.7] - 2025-01-27

### Other Changes
- License and automated release

## [1.2.6] - 2025-01-27

### Other Changes
- License and automated release

## [1.2.5] - 2025-01-27

### Other Changes
- License and automated release

## [1.2.4] - 2025-01-27

### Other Changes
- License and automated release
- License and automated release
- License and automated release

## [1.2.3] - 2025-01-27

### Other Changes
- Lock file maintenance (#182)
- Lock file maintenance (#183)

## [1.2.0] - 2025-01-27

### Bug Fixes
- Fixing artifacts
- Fixing deptry

### Maintenance
- Building the framework
- Test data
- Testing with pyportfolioopt
- Testing with pyportfolioopt
- Test resources
- Testing
- Style
- Build and publish

### Other Changes
- Initial commit
- Dockerfile for binder
- Source for hrp
- README.md
- README.md
- README.md
- Hrp refactored
- Hrp refactored
- Hrp refactored
- Hrp refactored
- Hrp refactored
- Hrp refactored
- Hrp refactored
- Hrp refactored
- Comments for all public functions
- Dendrograms support the testing
- Updating docker for pyportfolioopt
- Tree and linkage
- Bisection
- Bisection
- Cluster class
- Cluster class
- Remove read_pd
- Dependencies in setup
- Cluster class
- Cluster
- Cleaning
- Cleaning
- Github actions
- Publish to pypi automated
- Publish to pypi automated
- Publish to pypi automated
- Playing with pylint
- Pylint
- Pylint
- Better windows support
- Merge remote-tracking branch 'origin/master'
- Better windows support
- Slimmer testdata
- Simplified test
- Simplified test
- Remove spy prices
- Remove spy prices
- Add .deepsource.toml
- Merge pull request #3 from tschm/deepsource-config-49f579e1
- 1 remove docker (#2)
- Update main.yml (#7)
- Remove assert statement from non-test files (#8)
- Update main.yml
- Update __init__.py (#9)
- Update cluster.py (#11)
- Create release.yml (#13)
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Delete pypi.yml
- Update release.yml
- Update release.yml
- Update release.yml (#14)
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Pyhrp in pyproject.toml
- Update release.yml (#16)
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update and rename release.yml to release.yml`
- Update release.yml`
- Update release.yml`
- Update and rename release.yml` to release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Update release.yml
- Workflow
- Coverage
- Pre-commit
- Workflows
- Yml workflows
- Book
- Coverage
- Coverage
- Introduce sphinx folder
- Add _static in sphinx
- Conf file
- Create index.rst (#18)
- Update ci.yml
- Update book.yml
- Update README.md
- Update README.md
- Update index.md
- Delete .gitkeep
- Pre-commit
- Lock file
- Delete book/sphinx/_static directory
- Update conf.py
- Pre-commit
- Conf for sphinx
- Update .gitignore
- Update README.md
- Poetry
- Api
- Reports
- Makefile
- Bump certifi from 2023.5.7 to 2023.7.22 (#19)
- Pyyaml problem
- Poetry
- Bump tornado from 6.3.2 to 6.3.3 (#20)
- Deps updates
- Update Makefile
- Pyyaml
- Remove book installation
- Makefile
- Update pyproject.toml
- Poetry updates
- Remove kernel construction
- Download all artifacts in a single step
- Kernel
- Actions
- Check the actions
- Check poetry
- Update .pre-commit-config.yaml
- Update in lock
- Check the actions
- Update ci.yml
- Pre-commit verbose
- Update README.md
- Fmt
- Revisit pre-commit
- Update .pre-commit-config.yaml
- Fmt
- Update pyproject.toml
- Update README.md (#21)
- Update pyproject.toml
- Lock
- Workflow test
- Lock file
- Update pre-commit.yml
- Conduct
- Update pyproject.toml
- Update Makefile
- Update README.md
- Update conf.py
- [pre-commit.ci] pre-commit autoupdate (#22)
- Bump pillow from 10.0.0 to 10.0.1 (#23)
- [pre-commit.ci] pre-commit autoupdate (#24)
- [pre-commit.ci] pre-commit autoupdate (#25)
- [pre-commit.ci] pre-commit autoupdate (#26)
- [pre-commit.ci] pre-commit autoupdate (#27)
- [pre-commit.ci] pre-commit autoupdate (#28)
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Bump fonttools from 4.42.1 to 4.43.0
- [pre-commit.ci] pre-commit autoupdate
- Bump pillow from 10.0.1 to 10.2.0
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] auto fixes from pre-commit.com hooks
- [pre-commit.ci] pre-commit autoupdate
- Create dependabot.yml
- Update book.yml
- Bump pandas from 2.1.0 to 2.2.0
- Bump actions/checkout from 3 to 4
- Bump pre-commit/action from 3.0.0 to 3.0.1
- Bump pytest from 7.4.2 to 8.0.0
- Bump scipy from 1.11.2 to 1.12.0
- Bump matplotlib from 3.7.3 to 3.8.2
- Bump scikit-learn from 1.3.0 to 1.4.0
- [pre-commit.ci] pre-commit autoupdate
- Update pyproject.toml
- Lock file
- Bump pre-commit from 3.6.1 to 3.6.2
- Bump scikit-learn from 1.4.0 to 1.4.1.post1
- Bump pytest from 8.0.0 to 8.0.1
- Bump matplotlib from 3.8.2 to 3.8.3
- [pre-commit.ci] pre-commit autoupdate
- Bump pandas from 2.2.0 to 2.2.1
- Bump pytest from 8.0.1 to 8.0.2
- Bump pytest from 8.0.2 to 8.1.0
- Bump pytest from 8.1.0 to 8.1.1
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Bump pytest-cov from 4.1.0 to 5.0.0
- Bump pre-commit from 3.6.2 to 3.7.0
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Bump pillow from 10.2.0 to 10.3.0
- Bump scipy from 1.12.0 to 1.13.0
- Bump matplotlib from 3.8.3 to 3.8.4
- [pre-commit.ci] pre-commit autoupdate
- Bump pandas from 2.2.1 to 2.2.2
- Bump scikit-learn from 1.4.1.post1 to 1.4.2
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Bump pytest from 8.1.1 to 8.2.0
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Bump pre-commit from 3.7.0 to 3.7.1
- [pre-commit.ci] pre-commit autoupdate
- Bump pytest from 8.2.0 to 8.2.1
- Bump matplotlib from 3.8.4 to 3.9.0
- Bump scikit-learn from 1.4.2 to 1.5.0
- Bump scipy from 1.13.0 to 1.13.1
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Bump pytest from 8.2.1 to 8.2.2
- Update book.yml
- Update pre-commit.yml
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Update ci.yml
- [pre-commit.ci] pre-commit autoupdate
- Bump scikit-learn from 1.5.0 to 1.5.1
- Bump matplotlib from 3.9.0 to 3.9.1
- [pre-commit.ci] pre-commit autoupdate
- Bump zipp from 3.17.0 to 3.19.1
- Bump setuptools from 69.1.0 to 70.0.0
- [pre-commit.ci] pre-commit autoupdate
- Bump pytest from 8.2.2 to 8.3.1
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Bump pytest from 8.3.1 to 8.3.2
- Bump pre-commit from 3.7.1 to 3.8.0
- [pre-commit.ci] pre-commit autoupdate
- Bump matplotlib from 3.9.1 to 3.9.1.post1
- [pre-commit.ci] pre-commit autoupdate
- Bump matplotlib from 3.9.1.post1 to 3.9.2
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Bump pandas from 2.2.2 to 2.2.3
- Bump pytest from 8.3.2 to 8.3.3
- Bump scikit-learn from 1.5.1 to 1.5.2
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Bump pre-commit from 3.8.0 to 4.0.0
- [pre-commit.ci] pre-commit autoupdate
- Bump pre-commit from 4.0.0 to 4.0.1
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Update pre-commit.yml
- Update pre-commit.yml
- [pre-commit.ci] pre-commit autoupdate
- Bump pytest-cov from 5.0.0 to 6.0.0
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Update pyproject.toml
- Lock
- Update ci.yml
- Update pyproject.toml
- Lock
- Update ci.yml
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Move to uv
- Workflows
- Marimo page for book
- [pre-commit.ci] auto fixes from pre-commit.com hooks
- Fmt
- Workflows
- Workflows
- Workflows
- [pre-commit.ci] pre-commit autoupdate
- [pre-commit.ci] pre-commit autoupdate
- Taskfile
- Remove Makefile
- Workflows
- Workflows
- Remove Makefile
- [pre-commit.ci] pre-commit autoupdate
- Project.urls
- README with link to codespaces
- Simplify book
- Contributing with task file
- [pre-commit.ci] pre-commit autoupdate
- Actions
- Improving the book
- Improving the book
- Taskfile
- Remove Makefile
- Workflows
- Workflows
- Workflows
- Merge branch '137-move-to-uv-and-marimo' into main
- Improving the book
- Frozen sync
- Format
- Ruff config
- Fmt
- Typos
- Add renovate.json
- Update renovate.json
- Delete .github/dependabot.yml
- Update .pre-commit-config.yaml
- Rename renovate.json to .github/renovate.json
- Moving code of conduct and contributing
- Update pre-commit hook crate-ci/typos to v1.29.3
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.8.5
- Lock file maintenance
- Update pre-commit hook crate-ci/typos to v1.29.4
- Automerge
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.8.6 (#157)
- Towards cvxgrp/.github
- Update pre-commit hook rhysd/actionlint to v1.7.6 (#160)
- Run tests on a schedule
- V2.0.0 explicit
- Forgot checkout
- Lock file maintenance (#161)
- Update cvxgrp/.github action to v2.0.1
- Update pre-commit hook python-jsonschema/check-jsonschema to v0.31.0 (#164)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.9.0 (#165)
- Update cvxgrp/.github action to v2.0.2
- Update renovate.json
- Update cvxgrp/.github action to v2.0.3
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.9.1 (#169)
- Lock file maintenance (#170)
- Update .pre-commit-config.yaml (#172)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.9.2 (#173)
- Update cvxgrp/.github action to v2.0.6 (#174)
- Update pre-commit hook rhysd/actionlint to v1.7.7 (#175)
- Lock file maintenance (#176)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.9.3 (#178)
- Update release.yml (#179)
- Update cvxgrp/.github action to v2.0.7 (#180)
- Update pre-commit hook igorshubovych/markdownlint-cli to v0.44.0 (#181)
- License and automated release

<!-- generated by git-cliff -->
