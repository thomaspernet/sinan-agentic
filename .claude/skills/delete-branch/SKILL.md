---
description: "Safely delete the feature branch for issue #$ARGUMENTS. The devwatch CLI is the single authority for whether the branch is merged and safe to delete."
capability: core
---

Safely delete the feature branch for a completed issue.

The pipeline no longer launches this skill — per-child `delete-branch` is a
synchronous server-side action since #2799. This skill is the manual escape
hatch (the issue side-panel "Delete branch" button and the `/delete-branch`
slash command). It does **not** carry its own "is it merged?" gate:
`devwatch delete-branch` is the single authority and refuses unless the branch
is provably integrated — the GitHub issue is closed, a merged PR closing it
exists, or (for an `epic_integration` child) its `merge-to-base` row is `done`.
Do not re-derive that check here; resolve the branch, move HEAD off it if
needed, and let the CLI decide.

## Parse arguments

Extract issue number and optional run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `"42"` → ISSUE=42, RUN_ID=(none)
- `$ARGUMENTS` = `"42 --run 7"` → ISSUE=42, RUN_ID=7

If no issue number is provided, detect from current branch:

```bash
git branch --show-current
```

Extract issue number from branch name (e.g., `fix/42-broken` → ISSUE=42).

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Intelligence (what you decide)

1. Check issue history for context (run `devwatch issue-history --help` for all options):
   ```bash
   devwatch --repo "$REPO" issue-history <ISSUE>
   ```
   This shows the branch name, PR status, and whether cleanup has already been attempted.

2. Find the branch — check agent runs in the DB, then fall back to git:
   ```bash
   git branch -a --list "*fix/<ISSUE>-*" "*feat/<ISSUE>-*" "*refactor/<ISSUE>-*" "*chore/<ISSUE>-*" "*docs/<ISSUE>-*" "*ci-fix/<ISSUE>-*"
   ```
   Call the result `BRANCH`.

3. Validate safety:
   - `BRANCH` must not be one of the repo's pipeline branches (dev / staging / prod from `config.yaml`) or a universally protected name (`main`, `master`, `develop`)

4. **Resolve checkout target.** The CLI refuses to delete the currently-checked-out branch, so the agent must move HEAD itself before invoking deletion — and on epic children that means the epic integration branch, not the dev branch.

   ```bash
   WORKFLOW_JSON=$(devwatch --repo "$REPO" workflow-get --issue <ISSUE>)
   ```

   - If `WORKFLOW_JSON` is non-null and `.strategy == "epic_integration"`, set `TARGET` to the workflow's `canonical_branch` — the `epic/<root>-<slug>` integration branch the child landed on.
   - Otherwise, resolve the configured dev branch:

     ```bash
     TARGET=$(devwatch --repo "$REPO" branches dev)
     ```

     If the repo has no `dev` configured (e.g. release-only repos), fall back to `branches prod`. Do **not** hardcode a branch name (e.g. `dev`) — every repo resolves its own pipeline branches through `config.yaml`.

5. **Switch HEAD if it is on `BRANCH`.** Compare `git branch --show-current` to `BRANCH`; if they match, run:

   ```bash
   git checkout "$TARGET"
   ```

   If HEAD is already on a different branch, leave it alone — the agent may be mid-task on something unrelated.

## Execution

The CLI performs the integration gate (closed issue / merged PR / done `merge-to-base`) and the delete. Run it and report its result verbatim — if it refuses because the branch is not provably merged, surface that error; do not force the deletion.

If a RUN_ID was provided, pass `--run` only (issue is derived from the run):

```bash
devwatch --repo "$REPO" delete-branch --run <RUN_ID>
```

If no RUN_ID, pass `--issue` explicitly:

```bash
devwatch --repo "$REPO" delete-branch --issue <ISSUE>
```

## Boundary

This command deletes branches only. It does not close issues, merge PRs, or modify code.
