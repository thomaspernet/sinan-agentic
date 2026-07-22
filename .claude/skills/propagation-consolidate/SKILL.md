---
description: "Consolidate this run's flat propagation: issues into one umbrella per (class, helper) key, fold the folded issues' pipeline steps. Never scans the diff."
capability: core
---

Consolidate the per-site `propagation:` issues a `/propagation-scan` just filed (#2518, epic #2514). Group them — plus the epic's open `propagation:` issues sharing the same coarse `(class, helper)` key (#2515) — into **one umbrella per pattern**, close the folded per-site issues onto it, and reconcile their devwatch steps via #2516's fold primitive so the dashboard matches GitHub.

This is the relief valve for the flood (#2514): propagation issues are created one child at a time, so consolidation runs one child at a time, right behind the scan — the umbrella grows per child instead of N flat issues piling up. The dispatcher fires this automatically after a child's scan that found ≥1 site; it is also a manual button in the per-issue side panel.

**This skill never scans the diff.** It is not a discoverer — `/propagation-scan` is the only discoverer. This skill reads issues that already exist on GitHub and regroups them. It files no per-site issue, greps no codebase, reads no `git diff`.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill propagation-consolidate --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done. The mandatory-reads include the authoritative `propagation-scan` rule — the source of truth on the coarse `(class, helper)` key, the threshold policy, and the umbrella marker.

## Parse arguments

- `$ARGUMENTS` = `"42"` → ISSUE=42, RUN_ID=(none), BASE_BRANCH=(none)
- `$ARGUMENTS` = `"42 --run 7"` → ISSUE=42, RUN_ID=7
- `$ARGUMENTS` = `"42 --run 7 --base-branch local-dev-next"` → ISSUE=42, RUN_ID=7, BASE_BRANCH=local-dev-next
- `$ARGUMENTS` = `"42 --run 7 --base-branch local-dev-next --head 9f3a2b1"` → HEAD_SHA=9f3a2b1 (accepted, unused — this skill does not diff)

ISSUE is the **scan target** — the child issue whose pipeline this step sits in. `--run` / `--head` are passed by the dispatcher for symmetry with the scan; this skill records completion against `--run` but never diffs against `--head`.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` and `gh` command.

## 1. Collect this run's per-site propagation issues

List the open per-site `propagation:` issues — the scan filed these as flat `child-of` children of the scan target (and of its epic, when ISSUE is itself an epic child):

```bash
gh issue list --repo "$REPO" --state open --limit 200 \
  --search "propagation: in:title" --json number,title,state,body,labels
```

Keep only **per-site** `propagation:` issues — their title is `propagation: <summary>` and their body carries `Surfaced by /propagation-scan on #<...>` plus the `**Class:**` and `**Candidate site:**` lines. **Drop umbrella issues** — an umbrella's body carries a `propagation-umbrella: <class>:<helper-or-signal>` marker line; it is the accumulator, not a folding target.

Keep issues for this scan target **and** for the epic's other children: the coarse key is codebase-wide, so consolidation spans every child that filed for the pattern (that is the whole point — grouping *across* children). Resolve the epic via `child-of` on the scan target; an issue belongs to this consolidation when it is a `propagation:` child of ISSUE or of any sibling child under the same epic.

If there are no open per-site `propagation:` issues for the key set, this run has nothing to consolidate — go to §6 and record completion. (The dispatcher's scan-gate normally prevents an empty consolidate from firing; a manual re-run on an already-folded run lands here and is a clean no-op.)

## 2. Derive the coarse key for each issue

For each kept per-site issue, read its body and serialise its coarse key exactly as §4 of the `propagation-scan` rule defines it:

- **Class** — the `**Class:**` value (`new_helper` / `new_pattern` / `perf_fix` / `bugfix_shape`).
- **Helper-or-signal** — the named primitive the change introduced (the symbol for `new_helper`; the lower-kebab signal for the others).
- **Serialised coarse key** — `<class>:<helper-or-signal>`.

The **fine key** — `(file, line range)` from the `**Candidate site:**` line — identifies the individual site within a coarse-key group.

Group the kept issues by serialised coarse key.

## 3. Apply the threshold policy (#2517)

For each coarse-key group, decide whether it collapses into an umbrella:

- **Mechanical sweep — `new_helper` / `new_pattern` / `perf_fix`.** A find-and-replace refactor whose natural scope is the whole codebase. **Collapse the group into one umbrella.** Every site in the group folds onto a single umbrella checklist.
- **`bugfix_shape` — never collapse.** A recurring bug-shape is semantically distinct per site (each guard may differ and merits individual review). **Leave each `bugfix_shape` issue exactly as it is** — open, individual, untouched. Do not umbrella it, do not close it.

The class is the issue's recorded `**Class:**` — never inferred from the group's site count.

For each mechanical-sweep group, first discover whether an **open umbrella already exists** for the key (so this run appends rather than minting a second):

```bash
gh issue list --repo "$REPO" --state open --limit 200 \
  --search "propagation-umbrella: in:body" --json number,title,body
```

Match an umbrella when its body contains a line **exactly equal** to `propagation-umbrella: <class>:<helper-or-signal>` — exact on the serialised key, never substring. At most one open umbrella may match; if two carry the same marker, that is a data error — use the lowest-numbered and note the collision in the summary, do not append to both.

## 4. Promote each mechanical-sweep group into its umbrella

For each mechanical-sweep coarse-key group:

**Case A — an open umbrella already exists (append).** For every site in the group not already on the umbrella's checklist, append it via comment (additive and auditable — never rewrite the umbrella body):

```bash
gh issue comment <umbrella> --repo "$REPO" \
  --body "Appending propagation site (consolidated by /propagation-consolidate on #<ISSUE>):

- [ ] \`<file>:<line range>\` — <one-line summary>"
```

**Case B — no umbrella exists yet (create one).** Mint a single umbrella carrying the marker line, seeding its `## Sites` checklist with **every** site in the group (deduped by fine key):

```bash
devwatch --repo "$REPO" create-issue \
  --type feature \
  --title "propagation umbrella: <class>:<helper-or-signal>" \
  --body "## Propagation umbrella

propagation-umbrella: <class>:<helper-or-signal>

Consolidates the mechanical \`<class>\` sweep for \`<helper-or-signal>\` (#2517). One find-and-replace refactor across the codebase; each checklist item is one site, picked up individually through the normal pipeline.

## Sites

- [ ] \`<file>:<line range>\` — <one-line summary>
- [ ] \`<file>:<line range>\` — <one-line summary>
" \
  --area <area> \
  --priority <P2-medium|P3-low> \
  --parent <ISSUE> \
  --run-id <RUN_ID> \
  --no-claim-run
```

The umbrella is a flat `child-of` child of the scan target exactly like a per-site issue — `create-issue` derives the epic edge for you (pass only `--parent <ISSUE>`). `--no-claim-run` is mandatory: this skill may create one umbrella per group against one `--run-id`; without the flag each `create-issue` would overwrite the run's `github_issue`/`summary`. One umbrella per coarse key — never a second umbrella for a marker that already exists.

## 5. Close the folded per-site issues and reconcile their devwatch steps

Once a mechanical-sweep group's sites live on the umbrella (appended or seeded), every **open** per-site `propagation:` issue in the group is redundant. Close each one onto the umbrella:

```bash
gh issue close <folded> --repo "$REPO" \
  --comment "Closing as not planned — folded into propagation umbrella #<umbrella> for the \`<class>:<helper-or-signal>\` mechanical sweep. The site (\`<file>:<line range>\`) is tracked on the umbrella's checklist."
```

**Close, never edit.** The only mutation to a folded issue is closing it onto the umbrella. Never re-title, re-label, re-parent, re-link, or reopen one. Never close a `bugfix_shape` issue. Never close the umbrella.

Then **fold the closed issues' devwatch steps.** Closing a per-site issue out-of-band leaves its workflow step `pending` forever — the run-lifecycle fold only fires for an *active* run (the exact #770 divergence). Reconcile the folded steps on demand with #2516's primitive, scoped to exactly the issues just closed:

```bash
fold_closed_issue_steps(only_issues={(<folded>, "$REPO"), …})
```

Pass the closed issues as the `only_issues` set so the fold touches only them — the on-demand path, not the every-step tick. Each folded step reconciles to `CANCELLED` via the single writer `set_workflow_step_status` (#2516); the dashboard no longer shows a phantom `pending` step for a closed propagation issue.

## 6. Summary comment and record completion

Post a single summary comment on the scan target `#<ISSUE>`:

```bash
gh issue comment <ISSUE> --repo "$REPO" --body "## Propagation consolidate — <U> umbrellas, <F> folded

- Umbrella #<N> (\`<class>:<helper-or-signal>\`) — <created|appended>, <K> sites, closed #<a>, #<b>, …
- Left individual: <J> \`bugfix_shape\` sites (#<x>, #<y>)

See the propagation-scan rule for the coarse-key + threshold policy."
```

Drop a line that would print zeros — when no mechanical sweep crossed into an umbrella but `bugfix_shape` sites were left individual, the comment still records that nothing was folded.

Then emit the run report (advisory — a failed post must never fail the step), then flip the run status:

```bash
cat > /tmp/devwatch-report-<ISSUE>.json <<'JSON'
{
  "schema_version": 1,
  "notes": [
    {"category": "follow_up", "text": "Folded #<a>, #<b> into umbrella #<N>"}
  ],
  "stats": {"umbrellas": <U>, "folded": <F>, "left_individual": <J>}
}
JSON

devwatch --repo "$REPO" agent-report \
  --run-id <RUN_ID> \
  --file /tmp/devwatch-report-<ISSUE>.json \
  || echo "  agent-report failed (advisory) — continuing"

devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "Consolidated <F> propagation sites into <U> umbrellas under #<ISSUE>"
```

Fall back to `--issue <ISSUE> --branch "$(git branch --show-current)"` if RUN_ID is unavailable. When the run had nothing to consolidate (no open per-site issues), still flip to `completed` with a `Consolidated 0 —` summary so the dashboard records the no-op.

## Boundary

- **Never scans the diff.** This skill is a consolidator, not a discoverer. It reads no `git diff`, greps no codebase, files no per-site issue. `/propagation-scan` is the only discoverer; all per-site filing lives there.
- **Operates on filed issues only.** Its input is the open per-site `propagation:` issues the scan filed (this target + the epic's other children, by coarse key). Nothing else.
- **Mechanical sweep → umbrella; `bugfix_shape` → untouched.** It collapses only `new_helper` / `new_pattern` / `perf_fix` groups. It never umbrellas, closes, or otherwise touches a `bugfix_shape` issue.
- **One umbrella per coarse key.** Appends to the existing open umbrella when one carries the marker; mints at most one when none does. Never a second umbrella for the same marker.
- **Closes folded issues, folds their steps, edits nothing else.** Permitted mutations: appending a site to / creating an umbrella, closing folded per-site issues onto it, and reconciling their devwatch steps via #2516's `fold_closed_issue_steps(only_issues=…)`. Never re-titles, re-labels, re-parents, re-links, reopens, or rewrites the body of any issue; never closes an umbrella or a `bugfix_shape` issue; never creates an epic.
- **No code, no commits, no PR.** File / issue mutations only — never edits source, never commits, never opens a PR.
