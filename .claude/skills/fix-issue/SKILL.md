---
description: "Read GitHub issue #$ARGUMENTS, create a branch, and fix the bug."
---

Fix a bug. Read the issue, diagnose, fix, test, commit, push.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill fix-issue --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

Read this repo's CLAUDE.md for architecture and rules.

## Parse arguments

Extract issue number, optional run ID, optional base branch, and the optional auto-approve flag from `$ARGUMENTS`:
- `$ARGUMENTS` = `"42"` -> ISSUE=42, RUN_ID=(none), BASE_BRANCH=(none), AUTO_APPROVE=false
- `$ARGUMENTS` = `"42 --run 7"` -> ISSUE=42, RUN_ID=7, BASE_BRANCH=(none)
- `$ARGUMENTS` = `"42 --run 7 --base-branch feat/364-something"` -> ISSUE=42, RUN_ID=7, BASE_BRANCH=feat/364-something
- `$ARGUMENTS` = `"42 --run 7 --auto-approve"` -> ISSUE=42, RUN_ID=7, AUTO_APPROVE=true

Use ISSUE for git branch names and GitHub references. Use RUN_ID for all `devwatch` tracking calls. Use BASE_BRANCH (if provided) as the base for the new branch instead of the default dev branch. `--auto-approve` is set by the dispatcher when the owning workflow has the per-workflow auto-approve-gates toggle on (#2349); it only affects the **No-op terminal path** below — it has no effect on normal fix work.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Context loading

The mandatory reads (above) loaded every coding-principle doc this skill needs. Read this repo's CLAUDE.md for architecture and rules.

## Lineage pre-read

If this issue is a `child-of` another issue, pick up the ancestor context —
parent feature body, merged PR description, quality-check reports, sibling
bugs — before touching code. For root issues this is a no-op (the command
prints nothing).

```bash
devwatch --repo "$REPO" lineage-context <ISSUE> --format markdown
```

Treat the output as authoritative grounding. When a parent feature's merged
PR description contradicts what the bug report says happened, the PR wins —
that is what was actually shipped.

**Missing parent nudge.** If `lineage-context` returned no ancestors but the
issue body mentions another issue via phrases like *"while testing #N"*,
*"regression of #N"*, *"found in #N"*, or *"child-of #N"*, surface this to
the user once before coding: *"This issue mentions #N and has no `child-of`
link. Link it before I start? Run: `devwatch link <ISSUE> N --type child-of`"*
Do not auto-create the link. Proceed if the user says no — weaker mentions
like *"related to #N"* or *"see #N"* are not parent relationships.

## Find the reference

Before writing any code:

1. Read the issue: `gh issue view <ISSUE> --repo "$REPO" --json title,body,labels`
2. Identify the affected area from the issue description.
3. Read the existing code in that area to understand current patterns.
4. Do not start coding until you understand the existing patterns.

## Issue history (context awareness)

Check the action timeline to understand what has already happened on this issue:

```bash
devwatch --repo "$REPO" issue-history <ISSUE>
```

This shows prior implementation attempts, quality check results, PRs, and branches.

Available arguments (run `devwatch issue-history --help` to see all):
- `--run <RUN_ID>` — full details for a single run (timestamps, summary, files, commits)
- `--phase <phase>` — filter by phase: `impl`, `quality`, `pr`, `docs`, `release`, `cleanup`
- `--full` — expand all runs with full details
- `--comments` — show Lingtai-relevant issue comments (quality reports, fix summaries)

- **If prior runs exist**: this is a re-implementation. Drill into the details:
  - `devwatch --repo "$REPO" issue-history <ISSUE> --phase quality` — see quality check results
  - `devwatch --repo "$REPO" issue-history <ISSUE> --comments` — read quality failure reports and fix summaries
  - Use the failed check items as your primary guide for what to fix
  - Do NOT just re-validate against the original AC — address the specific feedback
- **If no prior runs exist**: this is a first implementation. Proceed normally.

## Workflow detection

Before creating a branch, look up the workflow that owns this issue. Since #1123 / epic #1116, the workflow row is the **source of truth** for `base_branch` and `strategy` — the skill does not re-derive either.

```bash
WORKFLOW_JSON=$(devwatch --repo "$REPO" workflow-get --issue <ISSUE>)
```

Decision matrix based on the response:

1. **`null` or empty** — no workflow.
   - If the issue body contains `child-of: #<epic>` (i.e. it is an epic child), **refuse**. Print a clear error and stop before touching git:

     ```
     Issue #<ISSUE> is a child of epic #<epic> but no workflow contains it.
     Epic children cannot be implemented without a workflow that owns the
     base branch and branching strategy. Create a workflow first:

       - Dashboard: open epic #<epic> and click "Create workflow"
       - CLI:       devwatch --repo "$REPO" workflow-create --root <epic> ...

     Then re-run /fix-issue <ISSUE>.
     ```

     Exit non-zero. Do not run `git fetch`, `git checkout`, or any mutation.
   - If the issue is standalone (no `child-of`), proceed to the **Standalone branch** block below.

2. **Workflow returned** — parse these fields:
   - `WORKFLOW_STRATEGY = workflow.strategy` — one of `"standalone"` or `"epic_integration"`. This decides which ref the child branch is cut off, so read it before either of the two refs below.
   - `WORKFLOW_BRANCH = workflow.canonical_branch` — the canonical implementation branch (for `epic_integration`, this is the **epic integration branch** `epic/<root>-<slug>` derived from contract fields; for `standalone` it falls through to `current_branch`). Do **not** read `workflow.current_branch` directly — it is overwritten by every child's `agent-update --branch`, so under `epic_integration` it returns whichever sibling ran last (#1611).
   - `WORKFLOW_BASE = workflow.base_branch` — the workflow's *parent* branch (typically `dev` / `local-dev-next`). For `standalone` workflows this is the ref children cut off; for `epic_integration` workflows it is the parent the epic was cut from (kept for drift-pull / merge-target context) and is **not** the right ref for children.

## Branch

**If a workflow was returned**, choose the implementation branch based on `WORKFLOW_STRATEGY`:

- **`standalone`** — the workflow has no shared integration branch. Each child cuts its own `fix/<ISSUE>-<slug>` off the workflow's parent ref:
  ```bash
  BASE=$WORKFLOW_BASE
  BRANCH=$(devwatch --repo "$REPO" workflow-branch-name --issue <ISSUE> --prefix fix)
  git fetch origin && git checkout -b "$BRANCH" origin/${BASE}
  ```
- **`epic_integration`** — children land on the shared epic integration branch cut at workflow creation. The child branch must be cut off `WORKFLOW_BRANCH` (the epic integration branch, `workflow.canonical_branch`), **not** `WORKFLOW_BASE` — cutting off `WORKFLOW_BASE` would skip every previously-merged sibling commit and produce a child PR with no shared history with the epic (#1457). Each child still ships on its own short-lived `fix/<childN>-<child-slug>` cut off the epic branch (#1096):
  ```bash
  BRANCH=$(devwatch --repo "$REPO" workflow-branch-name --issue <ISSUE> --prefix fix)
  git fetch origin && git checkout -b "$BRANCH" origin/${WORKFLOW_BRANCH}
  ```

**If no workflow was returned AND the issue is standalone** (no `child-of`):

Resolve the base branch in this order:
1. `BASE_BRANCH` argument if provided.
2. The nearest non-epic `child-of` ancestor's active branch (`lineage-branch`).
3. The default dev branch.

```bash
if [ -n "<BASE_BRANCH>" ]; then
  BASE=<BASE_BRANCH>
else
  BASE=$(devwatch --repo "$REPO" resolve-issue-base --issue <ISSUE> --drift-pull)
fi
BRANCH=$(devwatch --repo "$REPO" workflow-branch-name --issue <ISSUE> --prefix fix)
git fetch origin && git checkout -b "$BRANCH" origin/${BASE}
```

After creating or checking out the branch, record it (use `--run-id` if available, fall back to `--issue`):
```bash
devwatch --repo "$REPO" agent-update --run-id <RUN_ID> --branch "<your-branch-name>"
```
If no RUN_ID was provided:
```bash
devwatch --repo "$REPO" agent-update --issue <ISSUE> --branch "<your-branch-name>"
```

When the run belongs to a workflow step, `agent-update --branch` also updates `workflow_steps.branch` and `workflows.current_branch` — no separate call needed.

## Intelligence (what you decide)

1. Read the issue. Understand the symptoms, logs, stack traces.
2. Diagnose the root cause by reading the source code. Do not guess.
3. Write a failing test that proves the bug exists.
4. Fix the bug properly. Leave the code you touch better than you found it.
5. Run tests.

## No-op terminal path (#2103)

If — after reading the issue, the lineage, and the existing code — you
conclude **no code change is needed** (the bug is a duplicate of a closed
issue, was already fixed by a sibling PR, the reported behaviour is
working as intended, or the issue is invalid), do NOT exit silently or
take the success path. Both produce wedged runs:

- Exiting silently leaves the agent run as ``closed``, which the
  dispatcher reads as a structural failure and halts the entire workflow.
- Faking a commit and taking the ``ready_for_review`` path is dishonest
  and ships an empty PR.

Take the no-op terminal path instead:

**Confirmation gate.** Closing an issue is consequential and outward-facing, so by default present your no-op conclusion — the reason plus the evidence (the duplicate issue / the fixing PR / the commit, or why the reported behaviour is working as intended) — to the human and wait for approval before running the close below. **If `AUTO_APPROVE` is true** (the `--auto-approve` flag was on `$ARGUMENTS`): skip the confirmation entirely — close and report the no-op immediately, with no prompt and no pause. The owning workflow opted into auto-approval (#2349), which is a standing "yes, close it" for this run. The default stays gated.

1. Close the GitHub issue with a comment explaining why:

   ```bash
   gh issue close <ISSUE> --repo "$REPO" --comment "Closing as <reason>: <one-line explanation, link to the duplicate/fixing PR/commit>."
   ```

2. Report completion as a no-op (no branch, no commits, no files):

   ```bash
   devwatch --repo "$REPO" agent-update \
     --run-id <RUN_ID> \
     --status completed \
     --summary "no-op: <one-line reason — duplicate of #N / already fixed by <commit> / invalid because <reason>>"
   ```

The dispatcher detects the closed GitHub issue at IMPLEMENT-SUCCESS time,
skips the rest of this run's actions (quality / docs / PR), marks the
workflow step done, and advances the chain to the next child. The
workflow stays ``active``.

If RUN_ID is unavailable, fall back to ``--issue <ISSUE>``. Do not call
``agent-comment`` for the no-op — the close comment already explains the
outcome on the issue.

## Wrap up

After the fix is complete and tests pass:

1. Commit and push:
```bash
git add <changed-files>
git commit -m "fix(scope): <description> (closes #<ISSUE>)"
git push -u origin <your-branch-name>
```

2. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

3. Record completion (use `--run-id` if available, fall back to `--issue`):
```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status ready_for_review \
  --summary "<one-line summary of what you fixed>" \
  --files "<comma-separated changed files>" \
  --commits "$(git rev-parse HEAD)"
```

4. Post completion comment to GitHub issue:
```bash
devwatch --repo "$REPO" agent-comment \
  --issue <ISSUE> \
  --body "## Fix Complete\n\n**Summary**: <what you fixed and why>\n**Branch**: <branch-name>\n**Files**: <changed files>\n\nReady for review."
```

## Boundary

This command stops after committing and pushing. Do NOT create a PR. Tell the user to review the branch, then run `/submit-pr`.
