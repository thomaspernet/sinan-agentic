---
description: "Update documentation for the workflow root #$ARGUMENTS after its integration branch has merged."
capability: core
---

Update the docs affected by the change that just shipped for workflow root #$ARGUMENTS.

This is the workflow's single **Documentation** ship step (#2801, epic #2800). It runs once, post-merge, against the merged integration diff — there is no per-child docs pass and no `--epic` mode anymore. The argument is the workflow's root: an epic issue, or a self-rooted single issue. Either way you document **whatever that root describes** against the change that landed on the dev branch.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill add-documentation --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

**Standing authorization**: posting the `devwatch agent-report`, `devwatch agent-update`, and `devwatch agent-comment` calls described below (run report + status update + single completion comment on the root issue) is part of this skill's contract. Run them without asking for confirmation.

## Parse arguments

Extract the root issue number and optional run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `"42"` -> ISSUE=42, RUN_ID=(none)
- `$ARGUMENTS` = `"42 --run 7"` -> ISSUE=42, RUN_ID=7

ISSUE is the workflow root. There is no `--epic` flag — every workflow runs the same documentation step.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Resolve the merged change

The integration branch has already merged into the dev branch — this step runs after `merge-branch`. Document the change that landed, reading the root issue body as the spec.

1. Resolve the dev branch and check it out (the integration PR has already merged into it):
   ```bash
   DEV_BRANCH="$(devwatch --repo "$REPO" branches dev)"
   git fetch origin
   git checkout "$DEV_BRANCH"
   git pull --ff-only origin "$DEV_BRANCH"
   ```
2. Read the root issue body as the spec — it describes what shipped (an epic body, or a single issue), not a single child:
   ```bash
   gh issue view <ISSUE> --repo "$REPO" --json title,body,labels
   ```
3. Resolve the merged integration PR and its commit range. The integration PR is the most recent merged PR closing this root:
   ```bash
   PR=$(gh pr list --repo "$REPO" --state merged --search "closes #<ISSUE>" --json number,mergeCommit --jq '.[0]')
   ```
   Use the merge commit's first-parent range to get the diff that landed on dev:
   ```bash
   MERGE_SHA=$(echo "$PR" | jq -r .mergeCommit.oid)
   git diff "${MERGE_SHA}^1..${MERGE_SHA}" --name-only
   ```
   If no merged integration PR is found (a `devonly` workflow merges straight into dev with no PR), fall back to `devwatch --repo "$REPO" check-docs --issue <ISSUE>` — when `<ISSUE>` is an epic it diffs the integration branch against dev; otherwise it diffs the issue's merged change. Either source gives you the changed-file set.

## Intelligence (what you decide)

### Reviewer context — the author's implement notes

Before deciding which docs are stale, read the shipping workflow's run-report notes and use them to focus the pass (epic #2913). Each child's `implement` agent recorded `risk` notes ("watch this") and `consideration` notes ("deliberately didn't do X because Y") while the work was fresh — exactly the hand-off that points you at the behaviour or API a doc may now misrepresent. This is **read-only** context enrichment: it sharpens where you look; it is never posted anywhere.

Resolve the workflow that owns this root, then read its rollup digest (every shipped member's notes — you document the whole merged change, so the workflow-scoped report is the right scope):

```bash
WORKFLOW_ID="$(devwatch --repo "$REPO" workflow-get --issue <ISSUE> | jq -r '.id // empty')"
if [ -n "$WORKFLOW_ID" ]; then
  devwatch --repo "$REPO" get-report --workflow "$WORKFLOW_ID"
fi
```

`get-report` prints a category-grouped markdown digest (`### Risks` / `### Decisions` / `### Follow-ups`) assembled from the notes earlier agents recorded, or nothing when there are no notes.

- **Empty digest (or no workflow resolved) → skip.** No author context; map the changed files to docs as usual. Do not add a context block.
- **Non-empty digest → focus the pass.** Treat each **Risks** and **Decisions** entry as a pointer to a surface whose behaviour or contract may have shifted — check the docs for those surfaces first. **Follow-ups** are deferred work, not shipped behaviour; do not document them as if they landed.

Do **not** post these notes to GitHub. This step's only writes are the docs commit and the single completion comment below.

Map every changed file to its docs page using the doc map in CLAUDE.md. Then, for each flagged doc:

1. Read the doc.
2. Read the changed code that landed.
3. Decide: is the merged behavior, architecture, or APIs now misrepresented, or is the change internal?
4. If stale: update the doc content to match what shipped. Prefer one cohesive update per surface over N narrow per-file edits — read the root body as the "what shipped and why" spec, not a file-by-file history.
5. If not stale: skip.

Before committing, run the documentation checklist against your changes if one exists.

## Commit and push

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

The commit and push are mechanical — there is nothing to decide, so you do **not** run raw `git`. `devwatch commit-docs` owns the git for **both** doc modes: it resolves whether this project keeps docs in a standalone repo (external-docs mode) or inside the code repo (in-repo mode), stages exactly the files you pass — never `git add -A` — commits, and pushes to the right branch. Because your Bash call is `devwatch …` and never a raw `git push origin main`, the `--permission-mode auto` classifier never gates it, so this step runs unattended start to finish.

**If you updated no docs**, there is nothing to commit — skip `commit-docs` and set `DOC_SHA="$(git rev-parse HEAD)"` for the completion record below.

**Otherwise**, pass the docs you edited as a comma-separated `--files` list. The absolute paths you gave to Edit/Write are the simplest to pass — `commit-docs` stages them against the correct git root for this project's doc mode and prints the doc-commit SHA:

```bash
RESULT="$(devwatch --repo "$REPO" commit-docs --issue <ISSUE> --files "<abs-path-1>,<abs-path-2>")"
DOC_SHA="$(echo "$RESULT" | jq -r .sha)"
```

Emit the run report (advisory — a failed post must never fail the step). Write the
fixed JSON skeleton, filling `notes` with the docs you updated and any doc you
considered but deliberately left as-is (with the reason). Use an empty array
(`[]`) when no docs changed. Post it **before** the status flip below so the
report exists when completion hooks fire.

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
```
Fall back to `--issue <ISSUE>` if RUN_ID is unavailable.

Record completion (use `--run-id` if available, fall back to `--issue`). `DOC_SHA` was set in the Commit and push step above — the `commit-docs` output SHA when docs changed, or the dev branch HEAD when nothing changed.

```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "Docs updated for #<ISSUE>" \
  --files "<comma-separated changed files>" \
  --commits "$DOC_SHA"
```

Post completion comment to the root issue. The body is your own prose, so pass it through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

```bash
BODY=$(cat <<'BODY_EOF'
## Docs Updated

**Summary**: <which docs were updated and why>
**Files**: <changed files>

Docs are up to date for #<ISSUE>.
BODY_EOF
)

devwatch --repo "$REPO" agent-comment \
  --issue <ISSUE> \
  --body "$BODY"
```

## Boundary

This command updates docs only. It does not modify application code, and it does not open a PR — the change has already merged.
