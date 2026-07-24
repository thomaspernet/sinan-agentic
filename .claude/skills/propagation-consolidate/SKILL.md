---
description: "Consolidate this run's flat propagation: issues into one umbrella per (class, helper) key and fold the folded issues' pipeline steps. Never scans the diff, never routes."
capability: core
---

Consolidate the per-site `propagation:` issues a `/propagation-scan` just filed (#2518, epic #2514). Group them — plus the epic's open `propagation:` issues sharing the same coarse `(class, helper)` key (#2515) — into **one umbrella per pattern**, close the folded per-site issues onto it, and reconcile their devwatch steps via #2516's fold primitive so the dashboard matches GitHub. **Where each finding is tracked is not this skill's decision** — `/propagation-scan` routes every finding to its destination at file time (its §9.5, #3798/#3800), before consolidation runs. This skill only groups and folds what is already filed and attached.

This is the relief valve for the flood (#2514): propagation issues are created one child at a time, so consolidation runs one child at a time, right behind the scan — the umbrella grows per child instead of N flat issues piling up. The dispatcher fires this automatically after a child's scan that found ≥1 site; it is also a manual button in the per-issue side panel.

**This skill never scans the diff.** It is not a discoverer — `/propagation-scan` is the only discoverer. This skill reads issues that already exist on GitHub and groups them into umbrellas. It files no per-site issue, greps no codebase, reads no `git diff`.

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
- **`bugfix_shape` — individual by default; never collapsed on your own judgment.** A recurring bug-shape is semantically distinct per site (each guard may differ and merits individual review). **Leave each `bugfix_shape` issue exactly as it is** — open, individual, untouched. Do not umbrella it, do not close it. Two recorded decisions lift that default, both below: the key has **graduated**, or a human has made the settled-class call. Neither is a judgment you make from this run.

The class is the issue's recorded `**Class:**` — never inferred from the group's site count.

**A graduated `bugfix_shape` key collapses like a mechanical sweep (#155).** "Each guard merits individual review" is a justification spent once that review has converged on one rule. A key **graduates** when both hold, exactly as the `propagation-scan` rule's §6c defines it:

1. A rule under `.claude/rules/` carries a line **exactly equal** to `propagation-key: <class>:<helper-or-signal>` for the key — same exact-match discipline as the `propagation-umbrella:` marker, never a substring or a fuzzy title match.
2. That rule section specifies the **per-site treatment** — enough that a reader picks the right fix at a new site without re-deriving it. A rule that only names the hazard has not graduated the key.

```bash
rg -n 'propagation-key: <class>:<helper-or-signal>' .claude/rules/
```

Both conditions are a file read, not a judgment about the class's history — which is why this one *is* yours to evaluate. A graduated key takes **Case A or Case B** below exactly as a mechanical sweep does, including minting its umbrella when none exists.

**The settled-class exception — for an ungraduated key, a human's call, never yours (#3553).** A human may decide a `bugfix_shape` class with no committed rule has **settled**: its fix is byte-identical at every site, and per-site review across prior waves has stopped surfacing per-site divergence. They record that decision the only way this machinery reads — by hand-minting an umbrella carrying the class's marker line. #3535, #3552, and #3563 each did exactly this for the quoted-heredoc class.

So when an ungraduated `bugfix_shape` group's coarse key **already matches an open umbrella** (§3's discovery above), the collapse decision has already been made, by a human, and Case A applies to it exactly as to a mechanical sweep: append the group's untracked sites and fold them. You are executing a decision, not making one.

Absent both the `propagation-key:` marker and an open umbrella, the default stands: leave the group individual. **Case B is closed to an ungraduated `bugfix_shape` key — you never mint its umbrella.** That asymmetry is the whole gate: minting is the judgment, appending is the bookkeeping. Unrecorded "settledness" is a claim about a class's history across waves, which this run cannot see.

**This section governs collapsing only (#3800).** Everything above decides whether a group's sites fold onto one umbrella and close. It does not decide **where the group is tracked** — that was already decided at file time by `/propagation-scan`'s §9.5 routing (#3798/#3800), which moves open issues between workflows without folding or closing any of them, and which applies to `bugfix_shape` like any other class. This skill never re-decides tracking; the class gate here binds only the fold.

For each collapsing group — a mechanical sweep, or a graduated `bugfix_shape` key — first discover whether an **open umbrella already exists** for the key (so this run appends rather than minting a second):

```bash
gh issue list --repo "$REPO" --state open --limit 200 \
  --search "propagation-umbrella: in:body" --json number,title,body
```

Match an umbrella when its body contains a line **exactly equal** to `propagation-umbrella: <class>:<helper-or-signal>` — exact on the serialised key, never substring. At most one open umbrella may match; if two carry the same marker, that is a data error — use the lowest-numbered and note the collision in the summary, do not append to both.

## 4. Promote each collapsing group into its umbrella

For each collapsing coarse-key group — a mechanical sweep, or a graduated `bugfix_shape` key:

**Case A — an open umbrella already exists (append).** For every site in the group not already on the umbrella's checklist, append it via comment (additive and auditable — never rewrite the umbrella body). The summary is your own prose and sits beside a backticked path, so pass the body through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

```bash
BODY=$(cat <<'BODY_EOF'
Appending propagation site (consolidated by /propagation-consolidate on #<ISSUE>):

- [ ] `<file>:<line range>` — <one-line summary>
BODY_EOF
)

gh issue comment <umbrella> --repo "$REPO" --body "$BODY"
```

**Case B — no umbrella exists yet (create one).** Mint a single umbrella carrying the marker line, seeding its `## Sites` checklist with **every** site in the group (deduped by fine key). The body is your own prose and its checklist names paths in backticks, so pass it through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

```bash
BODY=$(cat <<'BODY_EOF'
## Propagation umbrella

propagation-umbrella: <class>:<helper-or-signal>

Consolidates the mechanical `<class>` sweep for `<helper-or-signal>` (#2517). One find-and-replace refactor across the codebase; each checklist item is one site, picked up individually through the normal pipeline.

## Sites

- [ ] `<file>:<line range>` — <one-line summary>
- [ ] `<file>:<line range>` — <one-line summary>
BODY_EOF
)

devwatch --repo "$REPO" create-issue \
  --type feature \
  --title "propagation umbrella: <class>:<helper-or-signal>" \
  --body "$BODY" \
  --area <area> \
  --priority <P2-medium|P3-low> \
  --parent <ISSUE> \
  --run-id <RUN_ID> \
  --no-claim-run
```

The umbrella is a flat `child-of` child of the scan target exactly like a per-site issue — `create-issue` derives the epic edge for you (pass only `--parent <ISSUE>`). `--no-claim-run` is mandatory: this skill may create one umbrella per group against one `--run-id`; without the flag each `create-issue` would overwrite the run's `github_issue`/`summary`. One umbrella per coarse key — never a second umbrella for a marker that already exists.

## 5. Close the folded per-site issues and reconcile their devwatch steps

Once a collapsing group's sites live on the umbrella (appended or seeded), every **open** per-site `propagation:` issue in the group is redundant. Close each one onto the umbrella **and** reconcile its now-orphaned devwatch step — **one command does both**, applied once per folded issue:

```bash
devwatch --repo "$REPO" close-folded-issue <folded> \
  --comment "Folded into propagation umbrella #<umbrella> for the \`<class>:<helper-or-signal>\` mechanical sweep. The site (\`<file>:<line range>\`) is tracked on the umbrella's checklist."
```

The server closes `<folded>` as **not planned** with your comment and — only on a successful close — reconciles its workflow step to `CANCELLED` via #2516's fold. `CANCELLED`, never `DONE`, is honest: the step never ran its work. Closing a per-site issue any other way leaves its step `pending` — the run-lifecycle fold fires only for an *active* run (the exact #770 divergence), and only the 10-minute reconcile tick would eventually catch up (#3574). Folding it into the close is why the dashboard never shows a phantom `pending` step beside a `Closed` stage. A step whose issue has an active run is left untouched: that run's own drain owns it. Re-running after a successful close-and-fold is a clean no-op.

**Close, never edit.** The only mutation to a folded issue is closing it onto the umbrella. Never re-title, re-label, re-parent, re-link, or reopen one. Never close the umbrella. Never close a `bugfix_shape` issue **unless its key has graduated** (§3, a `.claude/rules/` section carrying the key's `propagation-key:` marker and its per-site treatment) **or a human has already minted an umbrella carrying its coarse key's marker** (§3's settled-class exception) — either one is the recorded decision this fold executes. With neither, a `bugfix_shape` issue stays open and untouched.

**`close-folded-issue` is the only verb that closes-and-folds here — never substitute another (#3574).** It closes as *not planned* and reconciles exactly the one issue you name; nothing else. In particular `mark-done` is not its cousin: it sweeps every step at or before the one named, converges live runs, and stamps `DONE` on steps that never ran (#3539, #3540). If `close-folded-issue` fails, say so in the summary and stop — a folded issue with an unreconciled step is a phantom `pending` row someone can fix; a live run stamped `done` is lost work.

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

Omit `--run-id` if RUN_ID is unavailable — the run is resolved from `DEVWATCH_AGENT_RUN_ID` instead. When the run had nothing to consolidate (no open per-site issues), still flip to `completed` with a `Consolidated 0 —` summary so the dashboard records the no-op.

## Boundary

- **Never scans the diff.** This skill is a consolidator, not a discoverer. It reads no `git diff`, greps no codebase, files no per-site issue. `/propagation-scan` is the only discoverer; all per-site filing lives there.
- **Operates on filed issues only.** Its input is the open per-site `propagation:` issues the scan filed (this target + the epic's other children, by coarse key). Nothing else.
- **Eligible group → umbrella; ungraduated `bugfix_shape` → never minted.** It mints an umbrella for a `new_helper` / `new_pattern` / `perf_fix` group, and for a `bugfix_shape` group whose key has **graduated** — a `.claude/rules/` section carrying the key's `propagation-key:` marker and specifying the per-site treatment (§3, #155). It **never mints** an ungraduated `bugfix_shape` umbrella — that call is the human's, never yours (§3's settled-class exception). Where a human has already minted one carrying the class's marker, appending the group's untracked sites and folding them onto it is Case A bookkeeping, permitted exactly as for a mechanical sweep. With neither the marker nor an umbrella, a `bugfix_shape` issue stays open, individual, and untouched.
- **One umbrella per coarse key.** Appends to the existing open umbrella when one carries the marker; mints at most one when none does. Never a second umbrella for the same marker.
- **Closes folded issues, folds their steps, edits nothing else.** Permitted mutations: appending a site to / creating an umbrella, and closing each folded per-site issue onto it while reconciling its devwatch step in one operation via `devwatch close-folded-issue <folded>` (close as not-planned + #2516's fold, #3574). Never re-titles, re-labels, reopens, or rewrites the body of any issue; never closes an umbrella; never closes a `bugfix_shape` issue **except onto an umbrella its key has graduated into, or a human-minted one carrying its coarse key's marker** (§3, §5).
- **Never routes, never moves a member, never mints an epic.** Where each finding is tracked was decided at file time by `/propagation-scan`'s §9.5 routing (#3798/#3800/#3801) — this skill re-decides nothing. It creates no epic, moves no member between workflows, and calls no `regroup-onto-*-epic` command. A finding that is on the wrong epic is corrected by re-running the scan or by an operator's re-home, never here.
- **Closes-and-folds with the one verb, never a substitute.** `devwatch close-folded-issue` is the only command this skill uses to close a folded issue and converge its step. It never calls `gh issue close` separately (the close and the fold must land together, #3574), and never calls `mark-done` (or any other terminal-step verb) to reconcile a folded issue — that one sweeps live runs and writes `DONE` to steps that never ran (#3539, #3540).
- **No code, no commits, no PR.** File / issue mutations only — never edits source, never commits, never opens a PR.
