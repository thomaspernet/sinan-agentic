---
description: "Read GitHub issue #$ARGUMENTS, create a branch, and implement the feature."
capability: core
---

Implement a new feature. Read the issue, plan, implement, test, commit, push.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill feat-issue --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

Read this repo's CLAUDE.md for architecture and rules.

## Parse arguments

Extract issue number, optional run ID, and the optional auto-approve flag from `$ARGUMENTS`:
- `$ARGUMENTS` = `"42"` -> ISSUE=42, RUN_ID=(none), AUTO_APPROVE=false
- `$ARGUMENTS` = `"42 --run 7"` -> ISSUE=42, RUN_ID=7
- `$ARGUMENTS` = `"42 --run 7 --auto-approve"` -> ISSUE=42, RUN_ID=7, AUTO_APPROVE=true

Use ISSUE for git branch names and GitHub references. Use RUN_ID for all `devwatch` tracking calls. The base branch is **never** taken from an argument or guessed here — it is owned by the workflow (#2589), read in the **Branch** section below. `--auto-approve` is set by the dispatcher when the owning workflow has the per-workflow auto-approve-gates toggle on (#2349); it only affects the **No-op terminal path** below — it has no effect on normal implementation.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Context loading

The mandatory reads (above) loaded every coding-principle doc this skill needs. Read this repo's CLAUDE.md for architecture and rules.

## Single-step boundary

You own **exactly one** step of the workflow — this implementation step — and nothing else. Implement the feature, then record completion with `devwatch agent-update` and stop. The dispatcher's completion hook chains every later step (quality, propagation, merge, documentation, release) from your recorded completion; that is never your job. Do **not** run another workflow step, trigger another action, or try to drive the rest of the pipeline from inside this run. If the run is genuinely wedged and the dispatcher needs a kick, run `devwatch unblock <ISSUE> resume-run` (see `devwatch unblock-plan <ISSUE>` for the recovery options) — do not improvise the next step yourself.

## Lineage pre-read

If this issue is a `child-of` another issue (typically: a sub-feature of a
larger epic), pick up the ancestor context — parent feature body, merged PR
description, quality-check reports, sibling bugs — before touching code. For
root issues this is a no-op (the command prints nothing).

```bash
devwatch --repo "$REPO" lineage-context <ISSUE> --format markdown
```

Treat the output as authoritative grounding. When a parent feature's merged
PR description fixes an interface you were about to re-derive from the bug
report, follow the PR — that is what is actually in production.

**Missing parent nudge.** If `lineage-context` returned no ancestors but the
issue body indicates this is a sub-feature — phrases like *"sub-feature of #N"*,
*"part of epic #N"*, *"extends #N"*, or *"child-of #N"* — surface this to the
user once before coding: *"This issue mentions #N and has no `child-of` link.
Link it before I start? Run: `devwatch link <ISSUE> N --type child-of`"*
Do not auto-create the link. Proceed if the user says no — weaker mentions
like *"related to #N"* or *"see #N"* are not parent relationships.

## Find the reference

Before writing any code:

1. Read the issue: `gh issue view <ISSUE> --repo "$REPO" --json title,body,labels`
2. Classify the work type (new entity, endpoint, tool, component, page, etc.).
3. Find the closest existing implementation of the same type in the codebase.
4. Read it to understand patterns, conventions, and structure.
5. Do not start coding until you have a reference implementation to follow.

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

Before creating a branch, look up the workflow that owns this issue. Every issue is born into exactly one workflow (#2588), and the workflow row is the **single source of truth** for the branch base (#2589 / #1123 / epic #1116) — the skill never re-derives or guesses a base from lineage, drift, a repo default, or a CLI argument.

```bash
WORKFLOW_JSON=$(devwatch --repo "$REPO" workflow-get --issue <ISSUE>)
```

`workflow-get` returns the owning workflow's JSON. Parse one field:

- `WORKFLOW_BASE = workflow.base_branch_resolved` — the single workflow-owned base resolver (#2589): the shared **epic integration branch** `epic/<root>-<slug>` for an `epic_integration` workflow, otherwise the workflow's `base_branch`. Do **not** read `workflow.current_branch` (overwritten by every child's `agent-update --branch`; #1611) or `workflow.base_branch` directly (for `epic_integration` that is the parent the epic was cut from, **not** the ref children sit on) — `base_branch_resolved` already picks the correct ref for either strategy.

If `workflow-get` returns `null` (no workflow owns this issue — the workflow-birth invariant has been violated, e.g. an unsynced or manually-created issue), **refuse**. Print a clear error and stop before touching git:

```
Issue #<ISSUE> has no workflow. Every issue must be a member of a
workflow that owns its base branch (#2588). Create one first:

  - Dashboard: open the issue and click "Start workflow"
  - CLI:       devwatch --repo "$REPO" workflow-create ...

Then re-run /feat-issue <ISSUE>.
```

Exit non-zero. Do not run `git fetch`, `git checkout`, or any mutation.

If `base_branch_resolved` is `null` (an `epic_integration` workflow whose root-epic binding is missing — unsynced root issue or root not flagged `is_epic`), also refuse: the epic integration branch name cannot be resolved, so there is no safe ref to branch off. Print the binding-repair guidance and exit non-zero rather than guessing a base.

## Branch

Cut the child's feature branch off the workflow-owned base. One path, both strategies — `WORKFLOW_BASE` already resolves to the epic integration branch for `epic_integration` (so the child inherits every previously-merged sibling commit; #1457) and to the workflow's `base_branch` otherwise. The child always ships on its own short-lived `feat/<ISSUE>-<slug>` (#1096):

```bash
BRANCH=$(devwatch --repo "$REPO" workflow-branch-name --issue <ISSUE> --prefix feat)
git fetch origin && git checkout -b "$BRANCH" origin/${WORKFLOW_BASE}
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

1. Read the issue and acceptance criteria. Understand what the feature should do.
2. Assess complexity. If the scope is too large for a single branch, break into sub-issues.
3. Implement the feature following the patterns from your reference implementation.
4. Write tests for every new function/endpoint.
5. Run tests.

## No-op terminal path (#2103)

If — after reading the issue, the lineage, and the existing code — you
conclude **no implementation is needed** (the feature is already in
production via a sibling PR, the request duplicates a closed issue, the
acceptance criteria are already met by existing behaviour, or the issue
is invalid), do NOT exit silently or take the success path. Both produce
wedged runs:

- Exiting silently leaves the agent run as ``closed``, which the
  dispatcher reads as a structural failure and halts the entire workflow.
- Faking a commit and taking the ``ready_for_review`` path is dishonest
  and ships an empty PR.

Take the no-op terminal path instead:

**Confirmation gate.** Closing an issue is consequential and outward-facing, so by default present your no-op conclusion — the reason plus the evidence (the shipping PR / duplicate issue / the commit that already satisfies the acceptance criteria) — to the human and wait for approval before running the close below. **If `AUTO_APPROVE` is true** (the `--auto-approve` flag was on `$ARGUMENTS`): skip the confirmation entirely — close and report the no-op immediately, with no prompt and no pause. The owning workflow opted into auto-approval (#2349), which is a standing "yes, close it" for this run. The default stays gated.

1. Close the GitHub issue with a comment explaining why:

   ```bash
   gh issue close <ISSUE> --repo "$REPO" --comment "Closing as <reason>: <one-line explanation, link to the duplicate/shipping PR/commit>."
   ```

2. Report completion as a no-op (no branch, no commits, no files):

   ```bash
   devwatch --repo "$REPO" agent-update \
     --run-id <RUN_ID> \
     --status completed \
     --summary "no-op: <one-line reason — already shipped by #N / duplicate of #N / acceptance criteria already met by <commit> / invalid because <reason>>"
   ```

The dispatcher detects the closed GitHub issue at IMPLEMENT-SUCCESS time,
skips the rest of this run's actions (quality / docs / PR), marks the
workflow step done, and advances the chain to the next child. The
workflow stays ``active``.

If RUN_ID is unavailable, fall back to ``--issue <ISSUE>``. Do not call
``agent-comment`` for the no-op — the close comment already explains the
outcome on the issue.

## Wrap up

After implementation is complete and tests pass:

1. Commit and push:
```bash
git add <changed-files>
git commit -m "feat(scope): <description> (closes #<ISSUE>)"
git push -u origin <your-branch-name>
```

2. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

3. Emit the run report (advisory — a failed post must never fail the step).

Write the fixed JSON skeleton, filling only the `notes` array with this step's
follow-ups (`follow_up`), risks reviewers should watch (`risk`), and things you
considered but deliberately did not do (`consideration`). Use an empty array
(`[]`) when there is nothing worth recording. Post it **before** the status flip
below so the report exists when completion hooks fire.

```bash
cat > /tmp/devwatch-report-<ISSUE>.json <<'JSON'
{
  "schema_version": 1,
  "notes": [
    {"category": "follow_up", "text": "<a follow-up worth filing later>"},
    {"category": "risk", "text": "<a risk reviewers should watch>"},
    {"category": "consideration", "text": "<something considered but deliberately not done, and why>"}
  ]
}
JSON

devwatch --repo "$REPO" agent-report \
  --run-id <RUN_ID> \
  --file /tmp/devwatch-report-<ISSUE>.json \
  || echo "  agent-report failed (advisory) — continuing"
```
Fall back to `--issue <ISSUE> --branch "$(git branch --show-current)"` if RUN_ID is unavailable.

4. Record completion (use `--run-id` if available, fall back to `--issue`):
```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status ready_for_review \
  --summary "<one-line summary of what you built>" \
  --files "<comma-separated changed files>" \
  --commits "$(git rev-parse HEAD)"
```

5. Post completion comment to GitHub issue:
```bash
devwatch --repo "$REPO" agent-comment \
  --issue <ISSUE> \
  --body "## Feature Complete\n\n**Summary**: <what you built and why>\n**Branch**: <branch-name>\n**Files**: <changed files>\n\nReady for review."
```

## Boundary

This command stops after committing and pushing. Do NOT create a PR. Tell the user to review the branch, then run `/submit-pr`.
