---
description: "Submit the current branch as a PR."
capability: core
---

Wrap up a branch: commit, push, create PR, record trace.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill submit-pr --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Parse arguments

Extract issue number and optional run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `""` → ISSUE=(none), RUN_ID=(none)
- `$ARGUMENTS` = `"42"` → ISSUE=42, RUN_ID=(none)
- `$ARGUMENTS` = `"42 --run 7"` → ISSUE=42, RUN_ID=7
- `$ARGUMENTS` = `"--run 7"` → ISSUE=(none), RUN_ID=7

If RUN_ID is present, forward it as `--run-id` to the `devwatch submit-pr` call.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Checkout the correct branch

If ISSUE is present, verify the current branch belongs to that issue:

```bash
git branch --show-current
```

The branch name must start with `fix/<ISSUE>-`, `feat/<ISSUE>-`, `refactor/<ISSUE>-`, `chore/<ISSUE>-`, `docs/<ISSUE>-`, or `ci-fix/<ISSUE>-`.

If the current branch does **not** match, find and checkout the correct branch:

```bash
git branch -a | grep -E "(fix|feat|ci-fix)/<ISSUE>-"
```

Checkout the matching branch. If no matching branch exists, **stop** — tell the user no branch exists for issue #ISSUE.

If ISSUE is not present, stay on the current branch.

## Prerequisites

Run `/check-code-quality` if not already done. Do not submit a PR that hasn't passed the quality gate.

Check the issue history for context (run `devwatch issue-history --help` for all options):
```bash
devwatch --repo "$REPO" issue-history <ISSUE>
```

## Intelligence (what you decide)

1. Review the diff: `git diff --stat` and `git diff`. Check: no secrets, no debug logs.
2. Write a commit message: conventional prefix, explains the WHY.
3. Write a PR summary: what changed and how to verify.
4. Fold in the workflow's reviewer notes (PR body). Resolve the workflow id and read its accumulated step notes:

   ```bash
   WORKFLOW_ID=$(devwatch --repo "$REPO" workflow-get --issue <ISSUE> | jq -r '.id')
   devwatch --repo "$REPO" get-report --workflow "$WORKFLOW_ID"
   ```

   `get-report` prints a category-grouped digest (`### Risks` / `### Decisions` / `### Follow-ups`) assembled from the notes earlier agents recorded, or nothing when there are no notes. When the digest is non-empty **and** this issue opens a standalone PR (no epic ancestor — an epic member merges into the integration branch with no PR; see **Member behaviour** below), append it to your `--summary` under a `## Reviewer notes` heading so it lands in the PR body. Omit the section entirely when the digest is empty, and skip it on the epic-member merge path — the workflow ship PR carries the digest via `/submit-epic-pr`. The digest is earlier agents' prose, so the GitHub-writing rules in **Execution** below apply to it: strip banned tokens and personal data before embedding.
5. Choose labels: area labels (`area:backend`, `area:frontend`, etc.). Labels are best-effort — the CLI retries without them if they don't exist in the target repo.

## Execution

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

2. Emit the run report (advisory — a failed post must never fail the step). Write the fixed JSON skeleton, filling `notes` with PR facts reviewers should follow up on (`follow_up`) and any risk in this PR to watch (`risk`). Use an empty array (`[]`) when there is nothing worth recording. Post it **before** `submit-pr` below so the report exists when completion hooks fire.

```bash
cat > /tmp/devwatch-report-<ISSUE>.json <<'JSON'
{
  "schema_version": 1,
  "notes": [
    {"category": "follow_up", "text": "<anything reviewers should follow up after merge>"},
    {"category": "risk", "text": "<a risk in this PR reviewers should watch>"}
  ]
}
JSON

devwatch --repo "$REPO" agent-report \
  --run-id <RUN_ID> \
  --file /tmp/devwatch-report-<ISSUE>.json \
  || echo "  agent-report failed (advisory) — continuing"
```
Fall back to `--issue <ISSUE> --branch "$(git branch --show-current)"` if RUN_ID is unavailable.

3. Submit the PR:

```bash
devwatch --repo "$REPO" submit-pr \
  --message "<your commit message>" \
  --summary "<your PR summary>" \
  --label "<label1>" --label "<label2>" \
  --run-id <RUN_ID>
```

Omit `--run-id` if no RUN_ID was parsed from arguments.

The CLI handles everything deterministically: branch detection, commit, push, PR creation, sync.

### PR body auto-close contract

The CLI prepends `Closes #<issue>` as the first line of the rendered PR body so GitHub auto-closes the linked issue on merge. Do **not** put `Closes #N` (or `Fixes` / `Resolves` variants) in `--message` — that lands in the PR title, where GitHub ignores it. Keep the magic word in the body, where the CLI puts it.

## Workflow-rooted ship

When the current issue is the root of a workflow (i.e. the workflow's `root_issue_number` equals this issue), the dashboard's Submit Workflow button and the `submit-workflow-pr` action route to the ship-PR backend (`devwatch submit-epic-pr <root> --workflow-id <id>`), not a per-member PR. The routing is driven by `workflows.root_issue_number` on the backend — there is no "step 1 is the epic" heuristic, and the root need not carry the `epic` label: a one-member plain-issue workflow ships the same way as a multi-member epic (#2666). For a member of a workflow, `/submit-pr` performs the local merge into the integration branch as described in the framework's pipeline; only the final ship PR goes through `submit-epic-pr`.

### Member behaviour

Every workflow runs the one integration-branch path (#2666): the member branch is merged locally into `epic/<root>-<slug>` and the member issue is **closed immediately** (short comment linking the merge commit). No per-member GitHub PR is opened. The ship PR runs CI once, for the whole integration branch.

This path skips `gh pr create` — only the ship PR opens later via `submit-epic-pr` (one PR from `epic/<root>-<slug>` to the base branch).

## Boundary

This command does NOT merge the PR. Report the PR URL and stop.
