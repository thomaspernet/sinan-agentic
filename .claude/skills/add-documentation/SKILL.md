---
description: "Update documentation for issue #$ARGUMENTS. Checks stale docs and updates them."
---

Update docs affected by issue #$ARGUMENTS.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill add-documentation --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

**Standing authorization**: posting the `devwatch agent-report`, `devwatch agent-update`, and `devwatch agent-comment` calls described below (run report + status update + single completion comment on the target issue) is part of this skill's contract. Run them without asking for confirmation.

## Parse arguments

Extract issue number, optional run ID, and the optional `--epic` flag from `$ARGUMENTS`:
- `$ARGUMENTS` = `"42"` -> ISSUE=42, RUN_ID=(none), EPIC_MODE=false
- `$ARGUMENTS` = `"42 --run 7"` -> ISSUE=42, RUN_ID=7, EPIC_MODE=false
- `$ARGUMENTS` = `"42 --epic"` -> ISSUE=42, RUN_ID=(none), EPIC_MODE=true
- `$ARGUMENTS` = `"42 --epic --run 7"` -> ISSUE=42, RUN_ID=7, EPIC_MODE=true

`--epic` is a workflow-scoped post-merge mode (#1079) — see the **Epic mode** section below. Without it, the per-child flow runs unchanged.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Epic mode (--epic, #1079)

When `EPIC_MODE=true`, this is the workflow-level meta-docs pass triggered post-merge by the Documentation chip on the epic side panel. Skip the per-child flow entirely:

1. Resolve the dev branch from `devwatch --repo "$REPO" repo-info` and check it out (the epic PR has already merged into it).
2. Pull the latest dev branch so the merged epic commits are local:
   ```bash
   git fetch origin
   git checkout <DEV_BRANCH>
   git pull --ff-only origin <DEV_BRANCH>
   ```
3. Read the epic body as the spec — it describes the feature shipped, not a single child:
   ```bash
   gh issue view <ISSUE> --repo "$REPO" --json title,body,labels
   ```
   Confirm the `epic` label is present; bail with a clear message otherwise (the chip should never be triggerable on a non-epic, but defend the contract).
4. Resolve the merged epic PR and its commit range. The epic PR is the most recent merged PR closing this epic:
   ```bash
   PR=$(gh pr list --repo "$REPO" --state merged --search "closes #<ISSUE>" --json number,mergeCommit --jq '.[0]')
   ```
   Use the merge commit's first-parent range to get the diff that landed on dev:
   ```bash
   MERGE_SHA=$(echo "$PR" | jq -r .mergeCommit.oid)
   git diff "${MERGE_SHA}^1..${MERGE_SHA}" --name-only
   ```
5. Map every changed file to its docs page using the doc map in CLAUDE.md, the same way the per-child flow does — but read the **epic body** as the "what shipped and why" spec, not an issue history.
6. For each flagged doc, decide whether the merged behavior, architecture, or APIs are now misrepresented and update accordingly. This is meta-docs: prefer one cohesive update per surface over N narrow per-child updates.
7. Commit directly to the dev branch (the epic PR has already shipped — there is no feature branch to push to):
   ```bash
   git add <changed-files>
   git commit -m "docs: meta-docs for epic #<ISSUE>"
   git push origin <DEV_BRANCH>
   ```
8. Emit the run report (advisory — a failed post must never fail the step), then record completion and post a single completion comment on the epic. Post the report **before** the status flip so it exists when completion hooks fire:
   ```bash
   cat > /tmp/devwatch-report-<ISSUE>.json <<'JSON'
   {
     "schema_version": 1,
     "notes": [
       {"category": "follow_up", "text": "Updated <surface> — <why>"},
       {"category": "consideration", "text": "<surface considered but left as-is — why>"}
     ]
   }
   JSON

   devwatch --repo "$REPO" agent-report \
     --run-id <RUN_ID> \
     --file /tmp/devwatch-report-<ISSUE>.json \
     || echo "  agent-report failed (advisory) — continuing"

   devwatch --repo "$REPO" agent-update \
     --run-id <RUN_ID> --status completed \
     --summary "Meta-docs updated for epic #<ISSUE>" \
     --files "<comma-separated changed files>" \
     --commits "$(git rev-parse HEAD)"
   devwatch --repo "$REPO" agent-comment \
     --issue <ISSUE> \
     --body "## Meta-docs updated\n\n**Summary**: <which surfaces were updated>\n**Files**: <changed files>"
   ```
9. Stop. Do not run the per-child sections below — they target a feature branch that does not exist in epic mode.

## Epic-parent short-circuit (#948)

Child issues of an epic do **not** get a per-child docs pass. The epic
ships as one PR (see `/submit-epic-pr`), and docs run once against that
PR's combined diff. Check for an epic parent before doing anything:

```bash
EPIC=$(devwatch --repo "$REPO" epic-parent --issue <ISSUE>)
if [ -n "$EPIC" ]; then
  echo "Skipped — issue #<ISSUE> is a child of epic #$EPIC."
  echo "Run \`/add-documentation $EPIC\` after the epic PR is ready."
  # If this step belongs to a workflow, mark the docs action as skipped
  # (see the Workflow integration section at the bottom — same lookup, --status skipped).
  exit 0
fi
```

When `<ISSUE>` **is** itself an epic (the `epic` label is on the issue),
`check-docs` below automatically diffs the epic integration branch
against the dev branch so the combined changeset across every merged
child drives the docs check — the rest of this template stays the same.

## Context loading

1. Read the issue: `gh issue view <ISSUE> --repo "$REPO"`
2. Check the issue history (run `devwatch issue-history --help` for all options): `devwatch --repo "$REPO" issue-history <ISSUE>` — see what was implemented and which files changed.
3. Read this repo's CLAUDE.md to find the documentation map (which code areas affect which docs).
4. The mandatory-reads block already loaded this repo's documentation checklist.

## Branch tracking

Record the current branch so the dashboard shows work in progress (use `--run-id` if available, fall back to `--issue`):
```bash
devwatch --repo "$REPO" agent-update --run-id <RUN_ID> --branch "$(git branch --show-current)"
```
If no RUN_ID was provided:
```bash
devwatch --repo "$REPO" agent-update --issue <ISSUE> --branch "$(git branch --show-current)"
```

## Execution (detect stale docs)

```bash
devwatch --repo "$REPO" check-docs --issue <ISSUE>
```

## Intelligence (what you decide)

For each flagged doc:
1. Read the doc
2. Read the changed code on the branch
3. Decide: is this doc actually stale, or is the change internal?
4. If stale: update the doc content to match the new code
5. If not stale: skip

Before committing, run the documentation checklist against your changes if one exists.

## Commit and push

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

After updating docs, commit and push:

```bash
git add <changed-files>
git commit -m "docs: update docs for issue #<ISSUE>"
git push
```

Emit the run report (advisory — a failed post must never fail the step). Write
the fixed JSON skeleton, filling `notes` with the docs you updated and any doc
you considered but deliberately left as-is (with the reason). Use an empty array
(`[]`) when no docs changed. Post it **before** the status flip below so the
report exists when completion hooks fire.

```bash
cat > /tmp/devwatch-report-<ISSUE>.json <<'JSON'
{
  "schema_version": 1,
  "notes": [
    {"category": "follow_up", "text": "Updated <doc> — <why>"},
    {"category": "consideration", "text": "<doc considered but left as-is — why>"}
  ]
}
JSON

devwatch --repo "$REPO" agent-report \
  --run-id <RUN_ID> \
  --file /tmp/devwatch-report-<ISSUE>.json \
  || echo "  agent-report failed (advisory) — continuing"
```
Fall back to `--issue <ISSUE> --branch "$(git branch --show-current)"` if RUN_ID is unavailable.

Record completion (use `--run-id` if available, fall back to `--issue`):
```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "Docs updated for #<ISSUE>" \
  --files "<comma-separated changed files>" \
  --commits "$(git rev-parse HEAD)"
```

Post completion comment to GitHub issue:
```bash
devwatch --repo "$REPO" agent-comment \
  --issue <ISSUE> \
  --body "## Docs Updated\n\n**Summary**: <which docs were updated and why>\n**Files**: <changed files>\n\nDocs are up to date for #<ISSUE>."
```

## Boundary

This command updates docs only. It does not modify application code.
