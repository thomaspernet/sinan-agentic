---
description: "Open the epic-level PR for an epic-rooted workflow."
capability: core
---

Open the epic PR for an epic-rooted workflow (#1884). Wraps the same `gh pr create` + closes-trailer body + `workflows.pr_number` write that used to run in-process — moved into a skill so the dashboard's inline terminal opens like every other workflow / step action.

The readiness gate has already run on the server before this skill launched; if you got here, every child is closed on GitHub and present on the epic / shared branch. Do not re-prompt the user.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill submit-epic-pr --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Parse arguments

`$ARGUMENTS` shape:

```
<epic> --workflow-id <id> [--run <run_id>]
```

Examples:

- `185 --workflow-id 18 --run 42` → open the epic PR for epic #185 on workflow 18, attach to agent-run 42.

Extract `EPIC` (positional, integer), `WORKFLOW_ID` (`--workflow-id <id>`), and `RUN_ID` (`--run <id>`).

If `WORKFLOW_ID` is missing, **stop** — this skill is only for workflow-bound submissions launched from the dashboard. The legacy CLI path (no workflow id) is for direct ad-hoc use, not the dashboard chip.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command.

## Execution

Hand the work to the existing CLI — it dispatches on workflow shape (EPIC_INTEGRATION vs SAME/batch), composes the closes-trailer body, runs `gh pr create`, writes `workflows.pr_number`, and updates the agent run row:

```bash
devwatch --repo "$REPO" submit-epic-pr "$EPIC" \
  --workflow-id "$WORKFLOW_ID" \
  --run-id "$RUN_ID"
```

Omit `--run-id` if no `RUN_ID` was parsed.

The CLI prints the PR URL, the shipped-child list, and the agent-run id. Surface those lines to the user verbatim — they are the only acknowledgement the user gets that the click did something.

### Fold in the workflow's reviewer notes

After the ship PR is open, fold the workflow's accumulated step notes into its body so the reviewer opens it with the risks, decisions, and follow-ups the implementing agents recorded. This edits the PR body the CLI just wrote — no new posting mechanism.

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to the notes section. The digest is assembled from notes earlier agents wrote, so it may carry banned tokens or personal data — review and redact before posting.

2. Read the workflow's notes digest:

   ```bash
   devwatch --repo "$REPO" get-report --workflow "$WORKFLOW_ID"
   ```

   It prints a category-grouped markdown digest (`### Risks` / `### Decisions` / `### Follow-ups`), or nothing when there are no notes.

3. **Empty digest → stop.** Leave the PR body exactly as the CLI wrote it; no section is added.

4. Non-empty digest → append the reviewed digest to the PR body under a `## Reviewer notes` heading. Take the PR number from the CLI's `Epic PR #<n>` line, keep the existing body intact — the `Closes #N` trailers drive auto-close on release, so **never drop them** — and write it back:

   ```bash
   gh pr view <PR_NUMBER> --repo "$REPO" --json body -q .body > /tmp/epic-pr-body-$EPIC.md
   printf '\n\n## Reviewer notes\n\n' >> /tmp/epic-pr-body-$EPIC.md
   # append the reviewed digest (redacted per step 1) below the heading, then:
   gh pr edit <PR_NUMBER> --repo "$REPO" --body-file /tmp/epic-pr-body-$EPIC.md
   ```

### What the CLI does (so you can explain failures)

- **EPIC_INTEGRATION shape**: PR head = `epic/<N>-<slug>` (the epic integration branch), PR base = repo dev branch. Body lists `Closes #<root>` plus `Closes #<child>` per shipped child.
- **SAME/batch shape**: PR head = `workflow.base_branch` (the shared branch every step landed on), PR base = repo dev branch. Body lists `Closes` lines for the root epic and every `done` step.

Both shapes label the PR `epic` and write `workflows.pr_number` so the dashboard's `submit-workflow-pr` chip flips to `done` and **Merge PR** unblocks.

## Boundary

This skill opens the PR. It does not merge, does not run `/release`, and does not file follow-up issues. Tell the user the next step is to wait for CI green, then run `/merge-pr` (or click the workflow chip).
