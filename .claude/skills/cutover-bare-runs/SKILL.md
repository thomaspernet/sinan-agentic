---
description: "Ship-time cutover for the workflow-unification (epic #2586): verify the unified server's NOT-NULL workflow_id migration landed clean for this repo, re-deploy this repo's skills so the deleted standalone branch-resolution path is gone, and surface any fail-closed orphan run for a human to resolve. Read-mostly; the only write is the human-approved skill re-deploy."
capability: core
---

Run this **once per watched repo at the deploy that ships the workflow-unification (epic #2586)**. After this cutover there is only the workflow: every run is born into a workflow and `issue_runs.workflow_id` is NOT NULL. "Standalone" as a bare run with no owning workflow no longer exists.

The heavy lifting is the **server**'s job, not this skill's. When the unified devwatch server starts, `_migrate_issue_runs_workflow_id_not_null` runs automatically: it backfills a workflow for every pre-cutover run whose `workflow_id` was NULL (preferring an existing workflow, seeding a one-member `draft` workflow when none exists), then rebuilds `issue_runs` with `workflow_id INTEGER NOT NULL`. The migration is idempotent and re-runs harmlessly on every startup. So at the DB level there is nothing left to clean — this skill **verifies** that landing and handles the two residues the migration cannot fix on its own:

1. **Stale skills in this repo.** The pre-cutover skill templates carry the deleted standalone branch-resolution path. They stay deployed in this repo until you re-run `init-skills --force`.
2. **A fail-closed orphan run.** The migration leaves a run NULL — and the server then refuses to start — only when a run's repo has no resolvable `base_branch` and no existing workflow to bind to. That is a deliberate fail-closed: a human must resolve it, not the migration.

This skill never deletes data and never starts runs. Its only write is the human-approved skill re-deploy in §3.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill cutover-bare-runs --display

The output lists every doc you must read; treat it as if you opened each file directly. Do not proceed until done.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` and `gh` command.

## 1. Confirm the unified server is deployed

The cutover is meaningless against the old server — the migration only runs in the unified build. Confirm the running server is the one that ships epic #2586 before you touch this repo: check the deploy that just shipped, or ask the operator. If the unified server is **not** yet deployed, stop and report that — there is nothing to cut over against.

The server runs the NOT-NULL migration itself on startup. You do **not** run a migration command here; there is none to run from a watched repo.

## 2. Verify the migration landed clean for this repo

If the server started, the migration succeeded — the `issue_runs.workflow_id NOT NULL` rebuild fails the startup loudly rather than dropping rows, so a running unified server is itself the proof that no orphan run blocked it.

Spot-check the issues this repo cares about (any issue you expect to have an in-flight or recent run — open issues, issues with an active branch). For each, confirm a workflow owns it:

```bash
devwatch --repo "$REPO" workflow-get --issue <N>
```

- **Non-null workflow** → bound. Nothing to do.
- **`null`** → the issue has no run yet, which is fine; only issues with runs were touched by the migration. Do not seed a workflow just to make this non-null — birth-into-draft happens at issue creation going forward, not retroactively for issues that never ran.

This step is read-only. File nothing; start nothing.

## 3. Re-deploy this repo's skills (human-approved)

The old templates still carry the deleted standalone branch-resolution path. Re-deploy so this repo's `.claude/skills/` matches the unified contract (single base resolver, no `epic_child_requires_workflow` gate):

> Re-deploy this repo's skills to the unified set? This overwrites `.claude/skills/` for `$REPO`. Approve before I run it.

Only after the operator approves:

```bash
devwatch --repo "$REPO" init-skills --force
```

This rewrites `.claude/skills/` in place. Stage and commit only the `.claude/skills/` changes on a feature branch per this repo's normal flow — do not bundle unrelated changes.

## 4. If the server refused to start — resolve the orphan, then retry

If the unified server **failed** to start with a `workflow_id` NOT NULL error, a run could not be bound. The migration leaves a run NULL only when its repo has no resolvable `base_branch` and no existing workflow. Resolve the root cause, do not patch the row:

- **Repo missing a base branch** — configure the repo's `dev` branch (or its release base) in the project config so the migration can resolve one, then restart the server. Re-running the migration is idempotent.
- **Run on a since-closed / unconfigured issue** — if the issue is genuinely dead, the correct fix is to let the seed path create a one-member `draft` workflow (it does this automatically once a `base_branch` resolves). Never hand-set `workflow_id` to satisfy the constraint.

After the root cause is fixed, restart the server. The migration re-runs and binds the row. Report what you resolved.

## Boundary

This skill verifies the cutover and re-deploys skills (with approval). It does **not** run the migration (the server owns it), does **not** start or resume runs, does **not** delete rows, and does **not** create workflows for issues that never ran. One repo per invocation.
