---
description: "Create a release from the dev branch to production."
---

Ship a release through this repo's configured release stages: one PR per
stage, then a GitHub release tag.

## Release shape is per-repo (not always staging → main)

The repo's `branches:` config decides how many release stages there are —
the CLI derives the flow and `--step auto` emits the right next step. Do
**not** assume a staging hop exists:

- **2-stage** (`dev` + `prod`, no `staging`): one `dev → prod` PR, then tag.
  There is **no** staging PR and **no** staging tag. `--step auto` goes
  straight to `production`.
- **3-tier** (`dev` + `staging` + `prod`): `dev → staging` PR, optional
  staging tag, then `staging → prod` PR, then tag.
- **single-branch** (`dev` only): nothing to release.

Trust `--step auto`. Run whatever step it reports — never force a staging
PR on a repo that has no staging stage.

## Parse arguments

Extract optional run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `""` → RUN_ID=(none)
- `$ARGUMENTS` = `"--run 7"` → RUN_ID=7

If RUN_ID is present, forward it as `--run-id` to all `devwatch --repo "$REPO" release` calls.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Intelligence (what you decide)

1. Read commits since the last tag. Group by type (feat, fix, refactor, other).
2. Determine the version tag (patch vs minor vs major).

## Execution

The CLI auto-detects which step to run. Pass `--tag vX.Y.Z` on every call — the
first PR off `dev` uses it to bump `pyproject.toml`, `dashboard/package.json`,
and `src-electron/package.json` on the dev branch before the PR opens, so the
bump rides the flow to prod. The first PR is the `dev → staging` PR in a 3-tier
repo, or the `dev → prod` PR in a 2-stage repo:

```bash
devwatch --repo "$REPO" release --step auto --tag vX.Y.Z --notes "<release notes>" --run-id <RUN_ID>
```

Omit `--run-id` if no RUN_ID was parsed from arguments.

Possible outcomes (which fire depends on the repo's release shape above):
- **staging** -- 3-tier only: bumps version files, creates staging PR (dev -> staging). Wait for CI + merge.
- **staging-tag** -- 3-tier only, when `tag_staging: true` in repo config: creates a staging tag. Wait for build if `build: true`.
- **production** -- creates the production PR. In a 3-tier repo this is `staging -> prod`; in a 2-stage repo it is `dev -> prod` (and carries the version bump itself). Wait for CI + merge.
- **wait** -- a release PR is already open or a build is pending. Report its status.
- **done** -- all branches in sync. Nothing to release.

A 2-stage repo therefore ships in a single `dev → prod` PR + one tag — no
staging PR and no staging-tag step are attempted.

After the production PR is merged, run the same command again to tag:

```bash
devwatch --repo "$REPO" release --step auto --tag vX.Y.Z --run-id <RUN_ID>
```

Each step is idempotent. Running `/release` again after a partial release picks up where it left off — the bump step skips silently when files already match the target version.

## Boundary

Does NOT build or deploy. Deployment is handled by CI/CD.
