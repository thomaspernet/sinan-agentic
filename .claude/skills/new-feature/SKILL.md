---
description: "Create a GitHub issue for a feature request."
capability: core
---

Create a GitHub issue for a feature. Record it in devwatch. Stop.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill new-feature --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Parse arguments

Extract description, optional run ID, optional body-file path, optional parent refs, and optional brainstorm session from `$ARGUMENTS`:
- `$ARGUMENTS` = `"add dark mode"` -> DESCRIPTION="add dark mode", RUN_ID=(none), PARENTS=(), FROM_BRAINSTORM=(none)
- `$ARGUMENTS` = `"add dark mode --run 7"` -> DESCRIPTION="add dark mode", RUN_ID=7, PARENTS=()
- `$ARGUMENTS` = `"add dark mode --parent 42"` -> PARENTS=(42)
- `$ARGUMENTS` = `"add dark mode --parent 42 --parent 50"` -> PARENTS=(42, 50) — repeatable
- `$ARGUMENTS` = `"add dark mode --parent owner/repo#42"` -> cross-repo parent accepted
- `$ARGUMENTS` = `"--body-file /tmp/devwatch-issue-body-XXX.json --run 7"` -> read the JSON file for the payload; no inline description
- `$ARGUMENTS` = `"--from-brainstorm 11-05-26/dark-mode-rollout"` -> FROM_BRAINSTORM="11-05-26/dark-mode-rollout" (bare slug also accepted when unambiguous); the skill seeds the issue body from the session contents and the CLI back-links the session to the new issue in the same transaction

Strip `--run <N>`, `--body-file <PATH>`, `--from-brainstorm <SESSION>`, and every `--parent <REF>` from the description before using it.

**If `--body-file <PATH>` is present:** read the file with your Read tool — it is a JSON object with the exact bytes of the request. Use its fields for the feature issue:
- `description` → primary body text (verbatim, no shell quoting to worry about)
- `subject` → short hint for the title (optional)
- `sender` → From: line for the issue body (optional)
- `messages` → full Gmail thread, oldest-first, when the source is the inbox promote flow (optional). Preserve the full conversation in the issue body when present.

The body-file path is generated server-side and only contains filesystem-safe characters. Do NOT paste the file contents into Bash; read it with the Read tool.

The CLI accepts the parent as `N`, `#N`, or `owner/repo#N`. Preserve the exact form the user typed.

## Pull brainstorm context (when `--from-brainstorm` is set)

When `FROM_BRAINSTORM` is provided, the issue title and body get seeded from the session contents:

```bash
SESSION_TEXT=$(devwatch --repo "$REPO" brainstorm-read --session "$FROM_BRAINSTORM" --display)
```

The output streams every file under the session (README first, then alphabetised siblings, then subfolders) with `===== <abs path> =====` headers — the same shape as `doc-read --display`. Summarise it into the issue body:

1. Read the session's README (the first block in the stream) — its `## Summary` for the gist and its `## Files` index for which child files carry the detail.
2. Read each child `.md` file in the stream (`open-questions.md`, topic files, …) for the detailed thinking the README only summarises and indexes.
3. Promote the session's substance — the summary plus the relevant child-file detail — into the issue's description, and any concrete acceptance signals from the session into the `## Acceptance criteria` block.
4. The CLI back-links the session to the new issue automatically — do **not** write `brainstorm: <path>` into the body yourself.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Intelligence (what you decide)

1. Write a clear issue title and structured body (description, acceptance criteria, affected areas, priority).
2. Assess the area and priority.
3. **Detect implicit parent references.** If the user did not pass `--parent` but the description indicates this is a sub-feature of something — phrases like *"sub-feature of #N"*, *"part of epic #N"*, *"extends #N"*, *"child-of #N"* — surface the candidate once: *"This looks like a child of #N — link as `child-of`?"* If confirmed, append `N` to `PARENTS`. Do not auto-link without confirmation — mentions like "related to #N" or "see #N" are not parent relationships.
4. **Detect epic scope.** An epic is a parent issue that spans multiple child issues. Flag the feature as an epic when the scoping shows **any** of:
   - The user explicitly says *"epic"*, *"this is an epic"*, or asks for an epic.
   - The body naturally splits into **3 or more independent workstreams** that each deserve their own issue (e.g., *DB schema + API + dashboard + CLI + docs*).
   - Acceptance criteria spans **multiple areas** (backend + frontend + cli) and can't ship in a single PR without becoming unreviewable.
   - You find yourself writing a *"Suggested child breakdown"* section because the work is too large to implement in one branch.
   - Dependency notes like *"prerequisite for #N"* or *"blocks #N"* where the downstream work is itself multi-part.
   Narrow single-area features are **not** epics, even when the body is long. When in doubt, surface the candidate once: *"This scoping looks like an epic (N workstreams). Mark as epic?"* Set `IS_EPIC=true` only on confirmation or explicit request. If `IS_EPIC=true`, add a `## Suggested child breakdown` section to the body listing the workstreams as a numbered list — these become the child issues downstream.

## Execution

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

```bash
devwatch --repo "$REPO" create-issue \
  --type feature \
  --title "<title>" \
  --body "<structured body>" \
  --area <backend|frontend|agents|infrastructure> \
  --priority <P0-critical|P1-high|P2-medium|P3-low> \
  --parent <N> \
  --epic \
  --from-brainstorm <SESSION> \
  --run-id <RUN_ID>
```

Add one `--parent <REF>` for each entry in `PARENTS` (repeatable). Omit the flag entirely when `PARENTS` is empty. Omit `--run-id` if no RUN_ID was parsed from arguments. Include `--epic` only when `IS_EPIC=true`. Include `--from-brainstorm <SESSION>` when `FROM_BRAINSTORM` is set — the CLI back-links the session to the new issue (writes `linked_issues` in the session README AND appends `brainstorm: <path>` under the issue body's `Links:` block) in the same transaction.

The CLI handles everything deterministically: issue creation, labels, devwatch trace, sync. When `--epic` is passed, the CLI also creates and pushes the `epic/<N>-<slug>` branch on origin so child issues can branch off it (see #942). The `epic` label is authoritative — GitHub is the source of truth and the server derives the `is_epic` column from the label on every sync.

## Boundary

This command creates the issue. It does NOT implement the feature. Report the issue number and ask: "Want me to implement it? I'll run `/feat-issue <N>`"

When you launched from a `--body-file`, delete the file after `devwatch create-issue` succeeds:

```bash
rm -f "$BODY_FILE"
```
