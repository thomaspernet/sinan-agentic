---
description: "Update documentation for the workflow root #$ARGUMENTS after its integration branch has merged."
capability: core
---

Update the documentation pages affected by the change that just shipped for workflow root #$ARGUMENTS.

This is the workflow's single **Documentation** ship step (#2801, epic #2800). It runs once, post-merge, against the merged integration diff. The argument is the workflow's root: an epic issue, or a self-rooted single issue. Either way you document **whatever that root describes** against the change that landed on the dev branch.

Documentation is a Page, not a file (epic #2014). You read the merged diff, resolve which page covers what changed, and write that page. **Take no other git action** — no branch, no commit, no push. The working tree you read must be exactly as clean when you finish as when you started.

The post-merge ordering is kept deliberately. It is no longer a constraint imposed by `git diff` — a page reads as stale the moment a file under its globs changes, with no merge commit needed — it is a choice: document what actually shipped, rather than what a child intended before it was re-scoped or reverted.

## Mandatory reads — do this first

The `MANDATORY CONTEXT` block in your priming lists every doc and rule this step must read, each with its uuid. Open the ones this work needs with `read_doc(uuid=...)` and `read_rule(uuid=...)` before writing anything. The block carries titles and descriptions only — never bodies — so those two tools are how you read one.

There is no command to run and no tree to scan: the set is resolved from the library's own edges, so a doc named there exists and a doc that does not exist cannot be named.

**Standing authorization**: posting the `devwatch agent-report`, `devwatch agent-update`, and `devwatch agent-comment` calls described below (run report + status update + single completion comment on the root issue) is part of this skill's contract. Run them without asking for confirmation.

## Parse arguments

Extract the root issue number and optional run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `"42"` -> ISSUE=42, RUN_ID=(none)
- `$ARGUMENTS` = `"42 --run 7"` -> ISSUE=42, RUN_ID=7

ISSUE is the workflow root. Every workflow runs the same documentation step.

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
   This is the whole of the git this step takes, and all of it is a read: you are moving to the ref the change landed on so you can see it.
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
   If no merged integration PR is found (a `devonly` workflow merges straight into dev with no PR), fall back to `devwatch --repo "$REPO" check-docs --issue <ISSUE>` — when `<ISSUE>` is an epic it diffs the integration branch against dev; otherwise it diffs the issue's merged change. Either source gives you the changed-file set. That branch has no merge commit to name, so name the ref you read instead:
   ```bash
   MERGE_SHA="$(git rev-parse HEAD)"
   ```
   `MERGE_SHA` is set on both branches, because the completion record below names it either way.

## Resolve which pages cover it

Call `documentation_coverage` — it returns every documentation page of the project with the code globs it covers, whether it is `current` or `stale`, and the paths that changed under it since it was last written or last confirmed accurate. It also returns `never_written`: directories holding changed code no page covers at all.

Match your changed-file set against each page's `covers` globs. A page whose globs match a file you just saw in the merged diff is a page this pass has to decide about. A page the coverage already calls `stale` for paths outside your diff was made stale by earlier work — leave it; this pass documents what this root shipped.

`repos_unread` names checkouts that could not be read. A page covering only those reads as current on a question nobody managed to ask, so say so in the completion comment rather than treating it as clean.

## Intelligence (what you decide)

### Reviewer context — the author's implement notes

Before deciding which pages are stale, read the shipping workflow's run-report notes and use them to focus the pass (epic #2913). Each child's `implement` agent recorded `risk` notes ("watch this") and `consideration` notes ("deliberately didn't do X because Y") while the work was fresh — exactly the hand-off that points you at the behaviour or API a page may now misrepresent. This is **read-only** context enrichment: it sharpens where you look; it is never posted anywhere.

Resolve the workflow that owns this root, then read its rollup digest (every shipped member's notes — you document the whole merged change, so the workflow-scoped report is the right scope):

```bash
WORKFLOW_ID="$(devwatch --repo "$REPO" workflow-get --issue <ISSUE> | jq -r '.id // empty')"
if [ -n "$WORKFLOW_ID" ]; then
  devwatch --repo "$REPO" get-report --workflow "$WORKFLOW_ID"
fi
```

`get-report` prints a category-grouped markdown digest (`### Risks` / `### Decisions` / `### Follow-ups`) assembled from the notes earlier agents recorded, or nothing when there are no notes.

- **Empty digest (or no workflow resolved) → skip.** No author context; map the changed files to pages as usual. Do not add a context block.
- **Non-empty digest → focus the pass.** Treat each **Risks** and **Decisions** entry as a pointer to a surface whose behaviour or contract may have shifted — check the pages for those surfaces first. **Follow-ups** are deferred work, not shipped behaviour; do not document them as if they landed.

Then, for each page the coverage flagged:

1. Read the page: `read(uuid=<page uuid>)`.
2. Read the changed code that landed under its globs.
3. Decide: is the merged behavior, architecture, or API now misrepresented, or is the change internal?
4. **If it is misrepresented**, rewrite the page: `update_page_content(page_uuid=<uuid>, content=<full markdown>)`. Prefer one cohesive update per surface over N narrow per-file edits — read the root body as the "what shipped and why" spec, not a file-by-file history.
5. **If the change was internal** — a refactor, a rename, anything that does not alter what the page says — call `mark_documentation_current(page_uuid=<uuid>)`. This is the other answer a stale page has, and it is not the same as doing nothing: it records that somebody re-read the page against the change, which is what keeps "confirmed accurate" distinguishable from "nobody looked".

Every page the coverage flagged for a path in your diff gets one of those two answers. Leaving one unanswered is the state this step exists to remove.

Do not create pages for `never_written` areas. Which page should exist, and where it belongs in the tree, is a decision made in the Documentation view — report the areas in the completion comment so a reader can act on them.

Before finishing, run the documentation checklist against your changes if one exists.

## Wrap up

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

2. Confirm the working tree is clean. Nothing this step does touches a file, so anything staged or modified is something to explain, not to commit:

```bash
git status --porcelain
```

Emit the run report (advisory — a failed post must never fail the step). Write the
fixed JSON skeleton, filling `notes` with the pages you updated and any page you
considered but deliberately recorded as still accurate (with the reason). Use an
empty array (`[]`) when no page changed. Post it **before** the status flip below
so the report exists when completion hooks fire.

```bash
cat > /tmp/devwatch-report-<ISSUE>.json <<'JSON'
{
  "schema_version": 1,
  "notes": [
    {"category": "follow_up", "text": "Updated <page> — <why>"},
    {"category": "consideration", "text": "<page recorded as still accurate — why the change was internal>"}
  ]
}
JSON

devwatch --repo "$REPO" agent-report \
  --run-id <RUN_ID> \
  --file /tmp/devwatch-report-<ISSUE>.json \
  || echo "  agent-report failed (advisory) — continuing"
```
Omit `--run-id` if RUN_ID is unavailable — the run is resolved from `DEVWATCH_AGENT_RUN_ID` instead.

3. Record completion (omit `--run-id` if RUN_ID is unavailable). There is no doc commit to record — the pages are the artifact, so `--files` names them and `--commits` names the commit the pass read: the integration merge, or the dev branch's own tip when the workflow merged with no pull request.

```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "Docs updated for #<ISSUE>" \
  --files "<comma-separated page names>" \
  --commits "$MERGE_SHA"
```

4. Post completion comment to the root issue. The body is your own prose, so pass it through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

```bash
BODY=$(cat <<'BODY_EOF'
## Docs Updated

**Summary**: <which pages were updated and why>
**Confirmed accurate**: <pages whose change was internal, or "none">
**Not covered by any page**: <directories from never_written, or "none">

Docs are up to date for #<ISSUE>.
BODY_EOF
)

devwatch --repo "$REPO" agent-comment \
  --issue <ISSUE> \
  --body "$BODY"
```

## Boundary

This command updates documentation pages only. It does not modify application code, it takes no git action beyond reading the merged diff, and it does not open a PR — the change has already merged.
