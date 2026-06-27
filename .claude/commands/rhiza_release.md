---
description: Cut a release — choose the version bump, push the tag, and watch the release workflow
allowed-tools: Bash, Read
---

Cut a release for this repository. Follow the command-execution policy: always
prefer `make <target>`; never invoke `.venv/bin/...` directly.

How releasing works here. `make release` delegates to `rhiza-tools release`
(`.rhiza/make.d/releasing.mk`). It **always bumps** the version, folds a freshly
generated `CHANGELOG.md` into the version-bump commit (git-cliff pre-commit
hook), creates a `v*` tag, and pushes it. The tag push triggers the
`rhiza_release.yml` GitHub Actions workflow (validate → build → SBOM → draft →
publish → finalize). There is no separate post-tag changelog commit — the tagged
commit already carries the changelog.

**The bump type is a user decision.** Run interactively, `make release` prompts
the user to pick `MAJOR`, `MINOR`, or `PATCH`. Because this command runs in a
non-interactive shell (where the tool would silently default to `PATCH`), **you
must ask the user which bump to apply** and pass their choice through with
`BUMP=major|minor|patch` (supported by both `make release` and `make bump`).

Do the following, in order:

1. **Pre-flight checks.** Confirm the release is safe to cut and stop with a
   clear explanation if any check fails (do not push a tag from a dirty or
   behind branch):
   - Working tree is clean (`git status --porcelain` is empty).
   - On the default branch (`main`) — or confirm with the user if not.
   - Local `main` is up to date with `origin/main` (`git fetch` then compare).
   - `pyproject.toml` exists (the bump is skipped without it).

2. **Ask the user for the bump type.** Just like `make release` does
   interactively, ask the user to choose **MAJOR**, **MINOR**, or **PATCH**
   (semver: breaking / feature / fix). Skip this question only when
   `$ARGUMENTS` already names a bump type or authorises a default (see below).
   Hold the chosen `<bump>` (lowercase: `major` / `minor` / `patch`) for the
   next steps.

3. **Preview the bump (dry run).** Run `make release DRY_RUN=1 BUMP=<bump>` to
   show the user the planned new version and tag **before** anything is pushed.
   The `DRY_RUN=1` flag previews the bump/tag/push without applying them.
   Summarise what the real run will do (old → new version, tag name).

4. **Confirm.** Unless `$ARGUMENTS` explicitly authorises proceeding (see
   below), stop here and ask the user to confirm the previewed version before
   cutting the real release. Pushing a tag triggers a public release workflow —
   treat it as outward-facing and confirm first.

5. **Cut the release.** Run `make release BUMP=<bump> PUSH=1`. This bumps,
   commits (with changelog), tags, and pushes. The `PUSH=1` flag is **required
   here**: without it the tool asks an interactive `Push tag to remote? [y/N]`
   question that this non-interactive shell auto-declines, leaving the bump
   commit and tag stranded locally (unpushed). Since you already confirmed with
   the user in step 4, `PUSH=1` pushes straight through. Run it in the
   foreground so any failure is visible. If it fails, show the relevant output,
   diagnose the root cause, and stop — do not re-run blindly or hand-craft a tag.

6. **Watch the workflow.** Run `make release-status` to show the release
   workflow run and the latest release info. If the workflow is still in
   progress, tell the user it is running and how to re-check
   (`make release-status`). Report the final release URL once available.

7. **Report.** Tell the user the version that was released, the tag, and the
   release/workflow URL. If anything is still pending (workflow running, draft
   not yet published), say so plainly.

`$ARGUMENTS` handling:

- Empty → run the full ask-bump → preview → confirm flow above.
- `major` / `minor` / `patch` → use that as the bump type and skip the question
  in step 2 (still preview and confirm).
- `dry-run` (or `preview`) → do steps 1–3 only and stop; do **not** cut a real
  release. If no bump type is also given, still ask for it in step 2.
- `yes` / `confirm` / `--force` → skip the interactive confirmation in step 4
  and proceed straight through (still run the dry-run preview in step 3 so the
  user sees what shipped). If no bump type is also given, still ask for it in
  step 2.
- `status` → skip to step 6 and just report current release/workflow status.

Argument tokens combine, e.g. `minor yes` cuts a minor release without the
confirmation prompt; `minor dry-run` previews a minor bump only.
