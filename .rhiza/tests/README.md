# Rhiza Test Suite (`.rhiza/tests/`)

This directory is **synced from [jebel-quant/rhiza](https://github.com/jebel-quant/rhiza)**
via the `tests` bundle and runs in your project with `make rhiza-test`. Its job is to
validate the parts of *your* repository that Rhiza cares about — the metadata, docs, and
docstrings that vary per project — using the shared fixtures and helpers below.

> Tests that only exercise Rhiza's *own* template files (Makefile targets, workflow stubs,
> the project skeleton) live in Rhiza's mother-repo `tests/` suite and are **not** synced
> here — they would be identical in every consumer and can't be changed downstream. Put
> your project's own tests under your `tests/` directory, not here.

## Layout

The suite is flat — one file per concern:

- `test_pyproject.py` — validates `pyproject.toml` structure and required fields
- `test_readme_validation.py` — executes/syntax-checks `README.md` code blocks (see below)
- `test_docstrings.py` — runs doctests across the modules in your source folder
- `test_git_repo_fixture.py` — self-test for the shared `git_repo` fixture
- `conftest.py` — shared fixtures (`root`, `logger`, `git_repo`)
- `test_utils.py` — shared helpers (`run_make`, `setup_rhiza_git_repo`, `strip_ansi`)
- `stress/` — scaffolding for optional load/concurrency tests (see [stress/README.md](stress/README.md))

### Skipping README code blocks with `+RHIZA_SKIP`

By default, every `python` and `bash` code block in `README.md` is executed or
syntax-checked by `test_readme_validation.py`. To mark a block as intentionally
non-runnable (e.g. illustrative snippets or environment-specific commands), add
`+RHIZA_SKIP` to the opening fence line:

~~~markdown
```python +RHIZA_SKIP
# This block will NOT be executed or syntax-checked
from my_env import some_function
some_function()
```

```bash +RHIZA_SKIP
# This bash block will NOT be syntax-checked
run-something --only-on-ci
```
~~~

Markdown renderers (including GitHub) ignore everything after the first word on
a fence line, so the block still renders as a normal highlighted code block.
Blocks without `+RHIZA_SKIP` continue to be validated as before.

## Running Tests

```bash
make rhiza-test                                  # run this suite (the usual entry point)
uv run pytest .rhiza/tests/                       # equivalent, direct invocation
uv run pytest .rhiza/tests/test_pyproject.py      # a single file
uv run pytest .rhiza/tests/ -v                    # verbose
uv run pytest .rhiza/tests/ -m "not stress"       # skip stress tests
```

Stress tests accept custom parameters (defaults: 100 iterations, 10 workers):

```bash
uv run pytest .rhiza/tests/stress/ -v --iterations=10
```

## Fixtures

Defined in `conftest.py` and available to every test without import:

- `root` — repository root path (session-scoped)
- `logger` — configured logger instance (session-scoped)
- `git_repo` — sandboxed git repository (function-scoped)

Shared helpers live in `test_utils.py` and are imported directly:

```python
from test_utils import strip_ansi, run_make, setup_rhiza_git_repo
```

`.rhiza/tests` is on `pythonpath` (see `pytest.ini`), so `test_utils` imports resolve
without any `sys.path` manipulation.

## Writing Tests

- Use descriptive test names that explain what is being tested
- Group related tests in classes when appropriate
- Add docstrings to test modules and complex test functions
- Use `pytest.mark.skip` for tests that depend on optional features
- Prefer the `git_repo` fixture over touching the working tree
