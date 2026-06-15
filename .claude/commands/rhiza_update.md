---
description: Update the pinned Rhiza version in .rhiza/template.yml, sync, resolve conflicts, and verify
---

Update this repo's Rhiza template to a newer release: bump the pin in
`.rhiza/template.yml`, run the sync, resolve every conflict, verify the quality
gates, and open a PR. Follow the command-execution policy: always prefer
`make <target>`; never invoke `.venv/bin/...` directly.

`$ARGUMENTS` may name a target version (e.g. `v0.19.1`). If empty, target the
**latest** release of the upstream template repo.

## 1. Determine current and target versions

- Read `.rhiza/template.yml`: the `repository:` field is the upstream template
  repo (usually `jebel-quant/rhiza`) and `ref:` is the currently pinned version.
- Resolve the target version:
  - If `$ARGUMENTS` names a version, use it (verify the tag exists:
    `gh api repos/<repository>/git/ref/tags/<ref>`).
  - Otherwise get the latest release:
    `gh release view --repo <repository> --json tagName,publishedAt`.
- If `ref:` already equals the target, report "already up to date" and stop.
- Briefly summarize what's between the two versions when it's cheap to do so
  (`gh release view`/release notes), so the reviewer knows what's landing.

## 2. Bump the pin and commit (the tree must be clean to sync)

- `make sync` refuses to run on a dirty tree, so the bump lands first.
- Branch off the default branch (don't work on `main`/`master` directly):
  `git checkout -b sync/rhiza-<target>`.
- Edit only `ref:` in `.rhiza/template.yml` to the target version.
- Commit just that change (e.g. `Chore: bump rhiza template ref <old> → <target>`).

## 3. Sync

- Run `make sync` (it invokes `rhiza sync`). Expect it to either complete
  cleanly or report conflicts. It writes the refreshed `.rhiza/template.lock`.
- If it completes with no conflicts, skip to step 5.

## 4. Resolve every conflict

The sync is a 3-way merge. Two kinds of leftovers can appear — handle both, and
finish with **zero** `*.rej` files and **zero** conflict markers
(`<<<<<<<` / `=======` / `>>>>>>>`) anywhere tracked
(`git grep -lE '^(<<<<<<<|=======|>>>>>>>)'`).

**`*.rej` files (rejected hunks).** The 3-way merge often *already applied* a
hunk and still drops a duplicate `.rej`. For each, verify whether the change is
already present in the file (the added `+` lines exist; no conflict markers
remain). If it is, the `.rej` is spurious — delete it. If a hunk genuinely did
not apply, apply it by hand, then delete the `.rej`.

**Conflict-marked files.** Resolve by the ownership rule (see `CLAUDE.md` and the
`files:` block of `.rhiza/template.lock` for the authoritative managed-file list):

- **Rhiza-managed files** (the `.github/workflows/*`, `Makefile`,
  `.pre-commit-config.yaml`, `pytest.ini`, the `.rhiza/` engine, etc.): take the
  **incoming/upstream** side — these are owned by the template and should match
  it (`git checkout --theirs -- <file>` then `git add`).
- **Locally-owned or locally-hardened files** (notably `ruff.toml`, plus
  `pyproject.toml`, `README.md`, `src/`, your `tests/`): **merge by hand** —
  keep the local intent (e.g. stricter lint rules) while folding in genuine
  upstream additions, and make the result internally coherent (dedupe, drop
  comments that now contradict the config).

Validate every touched workflow/YAML still parses before moving on.

## 5. Verify the gates and fix fallout

A version bump can tighten the gates (new lint rules, `mypy --strict`, expanded
docs-coverage scope, etc.) and surface pre-existing issues. Run them and get
them green:

1. `make fmt` — pre-commit + lint
2. `make typecheck`
3. `make docs-coverage`
4. `make deptry`
5. `make security`
6. `make test`

**Scope your fixes.** Fix issues only in **locally-owned** files (`src/`,
`tests/`, `pyproject.toml`, locally-hardened config). If a gate fails because of
a **Rhiza-managed** file, that is an upstream problem: fix it in
`jebel-quant/rhiza` and bump again — do **not** edit the synced artifact in
place. Call out any such upstream-owned failure explicitly rather than papering
over it locally.

## 6. Commit, push, open a PR

- Commit the resolution and any in-scope fixes with clear messages (one logical
  change per commit: the conflict resolution, then each gate fix).
- Push the branch and open a PR (`gh pr create`) titled for the bump, e.g.
  `Chore: sync Rhiza template <old> → <target>`. In the body, summarize how each
  conflict was resolved, list any gate fallout you fixed, and flag anything that
  needs an **upstream** fix in Rhiza.
- Report a concise per-gate PASS/FAIL summary. If the workflow files changed,
  note that pushing them needs a token with the `workflow` scope.

Do not merge the PR. Stop after it is open and summarize what landed and what
(if anything) is blocked on an upstream Rhiza change.
