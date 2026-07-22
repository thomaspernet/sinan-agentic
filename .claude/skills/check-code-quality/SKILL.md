---
description: "Run the completion checklists against the current branch changes. Optionally pass an issue number to scope the review."
capability: core
---

Quality gate. Reviews code against the repo's checklists, posts a report to the GitHub issue, and returns pass/fail. This skill is a gate — if it fails, do NOT proceed to `/submit-pr`.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill check-code-quality --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Parse arguments

- `$ARGUMENTS` = `""` → review all changes on the current branch, no issue comment, RUN_ID=(none), BASE_BRANCH=(none)
- `$ARGUMENTS` = `"42"` → scope-check against issue #42 and post report as issue comment
- `$ARGUMENTS` = `"42 --run 7"` → same, with RUN_ID=7
- `$ARGUMENTS` = `"42 --run 7 --base-branch feat/364-something"` → same, with BASE_BRANCH=feat/364-something

If BASE_BRANCH is provided, use it instead of the repo's dev branch for the diff base in step 3.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## 1. Validate branch

If an issue number was provided, verify the current branch belongs to that issue **before doing anything else**:

```bash
git branch --show-current
```

The branch must be the issue's resolved branch. For a standalone issue that means `fix/<ISSUE>-`, `feat/<ISSUE>-`, `refactor/<ISSUE>-`, `chore/<ISSUE>-`, `docs/<ISSUE>-`, or `ci-fix/<ISSUE>-`. For a child of an epic running under EPIC_INTEGRATION the resolved branch is the parent epic's `epic/<EPIC>-<slug>` integration branch — which is what the dashboard already checked out for you when RUN_ID is present.

In every case, **the CLI re-checks this coupling at the writer** (step 7): if the current branch does not belong to the issue, `devwatch check-quality` rejects the row before any GitHub comment is posted. If you suspect the branch is wrong, **stop immediately** — do not gather the diff, do not run checklists, do not post any comment. Tell the user:

> "Current branch `<branch>` does not belong to issue #<ISSUE>. Check out the correct branch first."

## 2. Understand the scope

If an issue number was provided:
```bash
gh issue view <ISSUE> --repo "$REPO" --json title,body,labels
```
Note the issue's purpose — you will verify that changes stay within this scope.

Check prior quality check results to see if this is a re-check (run `devwatch issue-history --help` for all options):
```bash
devwatch --repo "$REPO" issue-history <ISSUE> --phase quality
```
If prior quality checks exist, note what failed before — verify those items are now fixed.

### Reviewer context — the author's implement notes

Before reviewing, read the implement agent's own run-report notes for this issue and use them to focus the review (epic #2913). The implementing agent records `risk` notes ("watch this") and `consideration` notes ("deliberately didn't do X because Y") while the work is fresh — exactly the hand-off a reviewer wants. This is **read-only** context enrichment: it sharpens where you look; it is never posted anywhere.

```bash
devwatch --repo "$REPO" get-report --issue <ISSUE>
```

`get-report` prints a category-grouped markdown digest (`### Risks` / `### Decisions` / `### Follow-ups`) assembled from the notes earlier agents recorded, or nothing when there are no notes.

- **Empty digest → skip.** No author context for this issue; review the diff as usual. Do not add a context block.
- **Non-empty digest → focus the review.** Carry the digest into the checklist pass in step 4:
  - Each **Risks** entry is a hot-spot — verify the author's concern is actually handled in the diff, not merely flagged.
  - Each **Decisions** entry is a claimed scope boundary — confirm the omission is sound and in scope. A "deliberately didn't do X" that should have been done is a scope/quality FAIL, not a free pass.
  - **Follow-ups** are out of scope by the author's intent — do not fail the gate on them.

Do **not** post these notes to GitHub. They already live on the issue's Report tab, and the quality report you post in step 7 is the only GitHub write this skill makes — re-posting would double-comment.

## 3. Gather the diff

```bash
# What changed on this branch vs the base branch.
# Resolve the repo's dev branch from config.yaml, or use BASE_BRANCH if provided.
BASE="${BASE_BRANCH:-$(devwatch --repo "$REPO" branches dev)}"
git diff "origin/${BASE}...HEAD" --stat
git diff "origin/${BASE}...HEAD"

# Any uncommitted changes
git diff --stat
git diff
```

Identify which languages are involved from file extensions:
- `*.py` → Python
- `*.ts` or `*.tsx` → TypeScript
- Both → run both checklists

## 4. Run the checklists

The mandatory-reads block already loaded every checklist this skill needs. Walk each item against the diff:

1. General checklist (scope, quality, decoupling, cleanup, tests)
2. If Python files changed: Python-specific checklist
3. If TypeScript files changed: TypeScript-specific checklist

For each item in each checklist, verify it against the actual diff. Do not skip items.

When the author's implement notes (section 2) flagged risks or decisions, give those items extra scrutiny here — the digest points you straight at where the author was unsure or made a deliberate trade-off.

## 5. Determine status

- **PASS**: every checklist item passes
- **FAIL**: one or more items fail

## 6. Build the report

Build a markdown report following this structure:

```
## Code Quality Check — PASS / FAIL

**Issue**: #<N> — <title> (or "No issue specified")
**Branch**: <branch-name>
**Files changed**: <count>
**Languages**: Python / TypeScript / Both

### General Checklist
- [x] <item description>
- [ ] FAIL: <what's wrong, file, line>

### Python / TypeScript Checklist
- [x] <item description>
- [ ] FAIL: <what's wrong, file, line>

### Result
X/Y checks passed. Z issues found.
```

For each failure, include the specific file/line, what's wrong, and the path to the violated rule.

## 7. Post the report and trace

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

If an issue number was provided, first emit the run report, then record the quality trace.

Emit the run report (advisory — a failed post must never fail the gate). Write
the fixed JSON skeleton, filling `notes` with the verdict plus any item
reviewers should track, and `stats` with the pass/fail counts. Post it
**before** the `check-quality` trace below so the report exists when completion
hooks fire.

```bash
cat > /tmp/devwatch-report-<ISSUE>.json <<'JSON'
{
  "schema_version": 1,
  "notes": [
    {"category": "consideration", "text": "Verdict: <PASS|FAIL> — <one-line>"},
    {"category": "follow_up", "text": "<an item reviewers should track, or drop this line>"}
  ],
  "stats": {"checks_passed": <N>, "checks_failed": <M>}
}
JSON

devwatch --repo "$REPO" agent-report \
  --run-id <RUN_ID> \
  --file /tmp/devwatch-report-<ISSUE>.json \
  || echo "  agent-report failed (advisory) — continuing"
```
Fall back to `--issue <ISSUE> --branch "$(git branch --show-current)"` if RUN_ID is unavailable.

Then post the quality trace:

```bash
# When RUN_ID is available, --run-id carries the context (issue is derived from the run)
devwatch --repo "$REPO" check-quality \
  --status pass|fail \
  --report "<the full markdown report>" \
  --run-id <RUN_ID>

# When RUN_ID is NOT available, pass --issue explicitly
devwatch --repo "$REPO" check-quality \
  --issue <ISSUE> \
  --status pass|fail \
  --report "<the full markdown report>"
```

The CLI handles everything: posts the report as a GitHub issue comment, records the quality check in the database, and exits with code 1 on failure.

If no issue number was provided, just print the report to the terminal.

## 8. Gate

**If PASS**: tell the user "Quality check passed. Ready for `/submit-pr`."

**If FAIL**: tell the user "Quality check failed. Fix the issues listed above, then run `/check-code-quality <N>` again. Do NOT proceed to `/submit-pr` until all checks pass."

## Boundary

This skill reviews and reports. It does NOT fix issues, commit, push, or create PRs. It is a gate — the agent must not proceed to `/submit-pr` if the check fails.
