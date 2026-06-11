---
description: "Extract a persistent rule from a resolved issue. Writes to .claude/rules/ or a principle doc. Never modifies application code."
---

Turn a resolved issue into a persistent rule. Read the issue context and its diff, decide whether the fix represents a **class** of mistake worth capturing, and write the constraint to the right rule file so the same mistake does not recur.

This skill writes text — rule entries in `.claude/rules/*.md` or principle docs under `documentation/general/principles/`. It does **not** modify application code, run tests, commit code changes, or execute Python.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill issue-to-rule --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done. The mandatory-reads include the authoritative `issue-to-rule` doc and the `rules-checklist` verifier — both are loaded once here and referenced (not re-read) throughout the rest of this skill.

## Parse arguments

- `$ARGUMENTS` = `"42"` → ISSUE=42, MODE=user-direct, RUN_ID=(none), BASE_BRANCH=(none), HEAD_SHA=(none)
- `$ARGUMENTS` = `"42 --mode <mode>"` → ISSUE=42, MODE=\<mode\>
- `$ARGUMENTS` = `"42 --run 7"` → ISSUE=42, RUN_ID=7
- `$ARGUMENTS` = `"42 --run 7 --base-branch feat/364-x"` → ISSUE=42, RUN_ID=7, BASE_BRANCH=feat/364-x
- `$ARGUMENTS` = `"42 --run 7 --base-branch local-dev-next --head 9f3a2b1"` → ISSUE=42, RUN_ID=7, BASE_BRANCH=local-dev-next, HEAD_SHA=9f3a2b1

`--head <sha>` is set by the dispatcher when the action fires inside an `epic_integration` chain (#1916). Since #2353 this skill runs **before** `merge-to-epic` and `delete-branch`, so the child's feature branch still exists and `origin/<base>...<HEAD_SHA>` is the child's *own un-merged change*. The flag pins the diff to the implement run's tip so it is the same diff every time, regardless of which branch the dispatcher left checked out.

Valid `MODE` values:

- `user-direct` (default) — user invoked the skill directly.
- `post-quality` — triggered after `/check-code-quality` failed on a violation that matched no existing rule.
- `self-flagged` — an implementation agent surfaced a generalizable lesson at the end of `/fix-issue` or `/feat-issue`.

For `post-quality` and `self-flagged`, require at least **two concrete instances** of the class — different files, different sessions, or different authors. A single-session, single-file pattern is iteration feedback; record a skipped summary and stop.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command.

## Context loading

1. Read this repo's CLAUDE.md.
2. The mandatory-reads block already loaded the authoritative rule (`issue-to-rule`) and the verifier checklist (`rules-checklist`).
3. Pull the issue's full context from `devwatch`:
   ```bash
   devwatch --repo "$REPO" issue-history <ISSUE> --full --comments
   ```
   This returns the issue's run timeline, prior quality-check reports, fix summaries, and relevant comments — the complete history the skill needs.
4. Read the diff that fixed the issue:
   ```bash
   BASE="${BASE_BRANCH:-$(devwatch --repo "$REPO" branches dev)}"
   if [ -n "$HEAD_SHA" ]; then
     git diff "origin/${BASE}...${HEAD_SHA}"
   else
     git diff "origin/${BASE}...HEAD"
   fi
   ```
   When `--head <sha>` is supplied the diff is pinned to that SHA — deterministic across re-runs and immune to checkout state. The implement run's tip SHA is recorded on the `agent_runs` row by `/fix-issue` / `/feat-issue`; the dispatcher resolves it at trigger time and passes it here.

   Run exactly the command above — nothing else. If it prints **nothing**, the diff is empty: record an `iteration-only` skipped summary and stop. Do **not** reconstruct a diff by other means (`git show <sha>`, `git diff <sha>^ <sha>`, a different base, the working tree) — an empty sanctioned diff means there is no change to extract a rule from, not that you should find one another way.
5. Read every file in the target tier you might write to, so you can detect conflict or subsumption before drafting:
   - `.claude/rules/*.md` — project-wide always-on rules
   - `documentation/general/principles/**/*.mdx` — language / architecture principles

## Decide: extract or skip

Evaluate the diff against the authoritative `issue-to-rule` doc loaded in the mandatory-reads block and classify:

- **`iteration-only`** — the fix was correct but not generalizable. Record a skipped summary via `devwatch agent-update` with the reason and stop.
- **`structural`** — a class of mistake exists and the constraint would prevent its next occurrence. Proceed.

If `MODE` is `post-quality` or `self-flagged`, also verify the two-instance requirement. Without two cited instances, skip.

## Draft the rule

Write a draft with:

- A **one-line statement** of the constraint. Voice matches the target tier — declarative 2nd-person for principle docs, file-referencing imperative for `.claude/rules/`.
- A **`**Why:**`** line — the past incident, the preference, or the guarantee the rule enforces.
- A **`**How to apply:**`** line — when and where the rule kicks in, concrete enough to judge edge cases.
- **Cited instances** — `file:line` references (at least two for agent-triggered modes).
- **Operation and target** — one of `add`, `update`, `supersede`, `flag-stale`, and the file the rule will be written to.

Choose the tier:

| Claim | Target |
|---|---|
| Applies to any project adopting this framework | `documentation/general/principles/<topic>.mdx` |
| Project-wide, always loaded | `.claude/rules/critical.md` |
| Domain-scoped (backend / frontend / CLI / infra) | `.claude/rules/<domain>.md` |

If you cannot pick a tier confidently, the rule is not general enough — skip.

## Run the verifier

Walk the draft through every item of the `rules-checklist` loaded in the mandatory-reads block. Each item is a yes/no question. Record your answers:

- **All items pass** → proceed to write.
- **Any item fails** → revise the draft once and re-run the checklist. If it fails a second time, stop and record the failing items in the `agent-update` summary.

## Write the rule

Depending on the operation:

- **`add`** — append a new entry to the target file. Preserve neighbouring style.
- **`update`** — replace the existing entry in place. Cite the new instance in `**Why:**`.
- **`supersede`** — write the new entry; append a `⚠ superseded by <anchor>` breadcrumb to the old entry. Do not delete the old entry.
- **`flag-stale`** — **do not edit the rule file**. Post a comment on the issue describing the drift and the file/line of the stale reference. Leave the rule alone.

Commit with a conventional message. The commit only touches rule files — never application code:

```bash
git add <rule-file-or-principle-doc>
git commit -m "docs(rules): <operation> rule from #<ISSUE> — <short description>"
```

## Record and comment

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

Update the agent-run trace and post a completion comment (use `--run-id` if available, otherwise `--issue`):

```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "<operation> rule from #<ISSUE> at <path>" \
  --files "<changed rule file>" \
  --commits "$(git rev-parse HEAD)"

devwatch --repo "$REPO" agent-comment \
  --issue <ISSUE> \
  --body "## Rule extracted\n\n**Operation**: <add|update|supersede|flag-stale>\n**Target**: \`<path>\`\n**Summary**: <one line>\n\nSee the commit for the full entry."
```

For skipped runs (iteration-only, insufficient instances, or flag-stale with no file write), still call `agent-update` with `--status completed` and prefix the summary with `Skipped —` so the dashboard can distinguish:

```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "Skipped — <reason, e.g. iteration-only / insufficient instances / flag-stale surfaced without file change>"
```

## Boundary

- **Writes text only.** Rule entries in `.claude/rules/` or principle docs. Never modifies application code, never runs tests, never opens a PR.
- **No autonomous delete.** `remove` is not one of the operations. The strongest deprecation is `supersede`.
- **Agent-triggered invocations are gated.** `post-quality` and `self-flagged` require ≥2 instances; the gate is not a suggestion.
- **Single rule per run.** If the draft wants to write more than one rule, the issue probably covered unrelated concerns — split the work.
