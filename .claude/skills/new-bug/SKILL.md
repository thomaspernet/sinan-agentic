---
description: "Create a GitHub issue for a bug report."
capability: core
---

Create a GitHub issue for a bug. Record it in devwatch. Stop.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill new-bug --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Parse arguments

Extract description, optional run ID, optional body-file path, optional parent refs, optional scenario coordinate, optional explicit child-of link, and optional brainstorm session from `$ARGUMENTS`:
- `$ARGUMENTS` = `"login page broken"` -> DESCRIPTION="login page broken", RUN_ID=(none), PARENTS=(), FROM_SCENARIO=(none), LINK_TO=(none), FROM_BRAINSTORM=(none)
- `$ARGUMENTS` = `"login page broken --run 7"` -> DESCRIPTION="login page broken", RUN_ID=7, PARENTS=()
- `$ARGUMENTS` = `"login page broken --parent 42"` -> DESCRIPTION="login page broken", PARENTS=(42)
- `$ARGUMENTS` = `"login page broken --parent 42 --parent 50"` -> PARENTS=(42, 50) — repeatable
- `$ARGUMENTS` = `"login page broken --parent owner/repo#42"` -> cross-repo parent accepted
- `$ARGUMENTS` = `"--body-file /tmp/devwatch-issue-body-XXX.json --run 7"` -> read the JSON file for the payload; no inline description
- `$ARGUMENTS` = `'--from-scenario "smoke::e2e/smoke-chat.spec.ts::chat toggle button is visible in header" --link-to 1100'` -> FROM_SCENARIO=`<coord>`, LINK_TO=1100, no inline description (the skill prefills title + body from the failed run)
- `$ARGUMENTS` = `"--from-brainstorm 11-05-26/login-bug-repro"` -> FROM_BRAINSTORM="11-05-26/login-bug-repro" (bare slug also accepted when unambiguous); the skill seeds the issue body from the session contents and the CLI back-links the session to the new issue in the same transaction

Strip `--run <N>`, `--body-file <PATH>`, every `--parent <REF>`, `--from-scenario "<COORD>"`, `--link-to <REF>`, and `--from-brainstorm <SESSION>` from the description before using it. The `<COORD>` is double-quoted because it contains `::` separators and (often) spaces in the title segment — preserve the quotes when shelling out and keep the value intact (do not split on `::` yourself; pass it through).

**If `--body-file <PATH>` is present:** read the file with your Read tool — it is a JSON object with the exact bytes of the request. Use its fields for the issue:
- `description` → primary body text (verbatim, no shell quoting to worry about)
- `subject` → short hint for the title (optional)
- `sender` → From: line for the issue body (optional)
- `messages` → full Gmail thread, oldest-first, when the source is the inbox promote flow (optional). Each message has `sender`, `received_at`, `body`. Preserve the full conversation in the issue body when present — the most recent reply often holds the actual question.

The body-file path is generated server-side and only contains filesystem-safe characters — safe to quote with shell single-quotes if you need to shell out. Do NOT paste the file contents into Bash; read it with the Read tool.

The CLI accepts the parent as `N`, `#N`, or `owner/repo#N`. Preserve the exact form the user typed. `--link-to <REF>` is a friendlier alias when promoting from a scenario failure: it is added to `PARENTS` so the rendered body carries a `child-of:` line pointing at the workflow root.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Pull failure context (when `--from-scenario` is set)

When `FROM_SCENARIO` is provided, the issue title and body get prefilled from the latest failed run of that scenario.

```bash
SCENARIO_JSON=$(devwatch --repo "$REPO" scenarios show "$FROM_SCENARIO")
```

`scenarios show` accepts a numeric scenario id today; if `$FROM_SCENARIO` is a coordinate `<suite>::<file>::<title>`, first resolve it to a numeric id by listing the catalogue and matching the coordinate triple, then re-run `show`:

```bash
SCENARIO_ID=$(devwatch --repo "$REPO" scenarios list --filter "<title segment>" \
  | awk -v file="<file segment>" '$0 ~ file {print $1}' \
  | tr -d '#' | head -n1)
SCENARIO_JSON=$(devwatch --repo "$REPO" scenarios show "$SCENARIO_ID")
```

Parse the JSON response (shape: `{scenario, group, suite, last_runs: [...]}`). Pull these fields:

- `scenario.title` → seed the issue title as `Regression: <scenario.title>` when no other title is supplied.
- The first entry in `last_runs` whose `status == "failed"` → the failure to report. Capture its `artifacts_dir` and any failure summary the CLI prints alongside the JSON (`devwatch scenarios show` prints a short failure list to stderr/stdout when the run is red — copy the first ~10 lines verbatim).
- `<artifacts_dir>/trace.zip` → "Trace" link in the body if it exists.
- `<artifacts_dir>/screenshot.png` → "Screenshot" link if it exists.
- `<artifacts_dir>/video.webm` → "Video" link if it exists.

Write a `## Failure` block at the top of the issue body with the truncated error message and the artifact paths. Keep it short — the goal is enough context for `/fix-issue` to start; the dashboard's scenario drawer (#1387) carries the full artifact view.

## Pull brainstorm context (when `--from-brainstorm` is set)

When `FROM_BRAINSTORM` is provided, the issue title and body get seeded from the session contents:

```bash
SESSION_TEXT=$(devwatch --repo "$REPO" brainstorm-read --session "$FROM_BRAINSTORM" --display)
```

The output streams every file under the session (README first, then alphabetised siblings, then subfolders) with `===== <abs path> =====` headers — the same shape as `doc-read --display`. Summarise it into the issue body — read the README's `## Summary` + `## Files` index for orientation, then the child `.md` files (`open-questions.md`, …) for the detail; promote the session's substance into the bug description and any concrete repro steps from the session into the `## Steps to reproduce` block. The CLI back-links the session to the new issue automatically — do **not** write `brainstorm: <path>` into the body yourself.

## Intelligence (what you decide)

1. Write a clear issue title and structured body (description, steps to reproduce, severity). When `FROM_SCENARIO` is set and the user did not pass an inline description, default the title to `Regression: <scenario.title>` and seed the body from the failure block above.
2. Assess the area and priority. Regression bugs are usually `frontend` + `P1-high` unless the failure clearly points elsewhere.
3. **Detect implicit parent references.** If the user did not pass `--parent` or `--link-to` but the description contains a phrase like *"while testing #N"*, *"regression of #N"*, *"found in #N"*, or *"child-of #N"*, surface the candidate to the user once: *"This looks like a child of #N — link as `child-of`?"* If confirmed, append `N` to `PARENTS`. Do not auto-link without confirmation — mentions like "related to #N" or "see #N for context" are not parent relationships.
4. **Detect epic scope.** Most bugs are single-area and **not** epics. Flag the bug as an epic only when the scoping shows **any** of:
   - The user explicitly says *"epic"*, *"this is an epic bug"*, or asks for an epic.
   - The root cause spans **3 or more independent fix workstreams** that each deserve their own issue (e.g., a systemic regression touching DB, server, CLI, and dashboard that must be split into per-area fixes).
   - You find yourself writing a *"Suggested child breakdown"* section because a single branch can't carry the whole fix.
   Narrow bugs are **not** epics, even when the repro is long. When in doubt, surface the candidate once: *"This bug looks like an epic (N workstreams). Mark as epic?"* Set `IS_EPIC=true` only on confirmation or explicit request. If `IS_EPIC=true`, add a `## Suggested child breakdown` section to the body listing the fix workstreams as a numbered list.

## Execution

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

2. When `LINK_TO` is set, append it to `PARENTS` so the rendered body carries `child-of: #<LINK_TO>` (the CLI does the rendering — the skill just passes the parent ref through).

3. Build the `devwatch create-issue` invocation:

```bash
devwatch --repo "$REPO" create-issue \
  --type bug \
  --title "<title>" \
  --body "<structured body>" \
  --area <backend|frontend|agents|infrastructure> \
  --priority <P0-critical|P1-high|P2-medium|P3-low> \
  --parent <N> \
  --regression-scenario "<COORD>" \
  --epic \
  --from-brainstorm <SESSION> \
  --run-id <RUN_ID>
```

Add one `--parent <REF>` for each entry in `PARENTS` (repeatable). Add one `--regression-scenario "<COORD>"` for each scenario coordinate the issue should track — pass `FROM_SCENARIO` here when it is set; the flag is repeatable for issues that cover multiple regressions. Omit `--parent` entirely when `PARENTS` is empty; omit `--regression-scenario` entirely when no scenario is in scope. Omit `--run-id` if no RUN_ID was parsed from arguments. Include `--epic` only when `IS_EPIC=true`. Include `--from-brainstorm <SESSION>` when `FROM_BRAINSTORM` is set — the CLI back-links the session to the new issue (writes `linked_issues` in the session README AND appends `brainstorm: <path>` under the issue body's `Links:` block) in the same transaction.

The CLI handles everything deterministically: issue creation, labels, devwatch trace, sync. The body's `Links:` block is rendered by the CLI — it always emits a single `Links:` header followed by `- child-of: #N` lines (one per parent) and `regression-scenario: <coord>` lines (one per scenario), in that order. The body parser added in #1386 picks the regression line up on the next sync and rebuilds the `scenario_links` cache. When `--epic` is passed, the CLI also creates and pushes the `epic/<N>-<slug>` branch on origin so child issues can branch off it (see #942). The `epic` label is authoritative — GitHub is the source of truth and the server derives the `is_epic` column from the label on every sync.

## Example

Promoting a failing smoke run into a regression-tracked bug under workflow #1100:

```bash
/new-bug --from-scenario "smoke::e2e/smoke-chat.spec.ts::chat toggle button is visible in header" --link-to 1100
```

The skill resolves the scenario, pulls the failure context, then runs:

```bash
devwatch --repo "$REPO" create-issue \
  --type bug \
  --title "Regression: chat toggle button is visible in header" \
  --body "$BODY" \
  --area frontend \
  --priority P1-high \
  --parent 1100 \
  --regression-scenario "smoke::e2e/smoke-chat.spec.ts::chat toggle button is visible in header"
```

Where `$BODY` looks like:

```
## Failure
TimeoutError: locator.click: Timeout 5000ms exceeded
Trace: <artifacts_dir>/trace.zip
Screenshot: <artifacts_dir>/screenshot.png

Links:
- child-of: #1100
regression-scenario: smoke::e2e/smoke-chat.spec.ts::chat toggle button is visible in header
```

(The CLI assembles the `Links:` block — the skill writes only the `## Failure` block plus any free-text. Do not author `Links:` by hand; let `--parent` and `--regression-scenario` render it.)

## Boundary

This command creates the issue. It does NOT fix the bug. Report the issue number and ask: "Want me to fix it? I'll run `/fix-issue <N>`"

When you launched from a `--body-file`, delete the file after `devwatch create-issue` succeeds so temp files don't accumulate:

```bash
rm -f "$BODY_FILE"
```
