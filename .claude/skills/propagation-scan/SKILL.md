---
description: "Scan the codebase for sites where the current diff's change could apply elsewhere. Files child issues, never edits inline."
capability: core
---

After a fix or feature lands, find every other site where the same change could apply — and file each site as its own tracked unit of work. File-only: never edits code, never commits, never opens a PR.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill propagation-scan --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done. The mandatory-reads include the authoritative `propagation-scan` rule — that doc is loaded once here and referenced (not re-read) throughout the rest of this skill.

## Parse arguments

- `$ARGUMENTS` = `"42"` → ISSUE=42, RUN_ID=(none), BASE_BRANCH=(none), HEAD_SHA=(none), CAP=5, AUTO_APPROVE=false
- `$ARGUMENTS` = `"42 --run 7"` → ISSUE=42, RUN_ID=7
- `$ARGUMENTS` = `"42 --run 7 --base-branch feat/364-x"` → ISSUE=42, RUN_ID=7, BASE_BRANCH=feat/364-x
- `$ARGUMENTS` = `"42 --run 7 --base-branch local-dev-next --head 9f3a2b1"` → ISSUE=42, RUN_ID=7, BASE_BRANCH=local-dev-next, HEAD_SHA=9f3a2b1
- `$ARGUMENTS` = `"42 --cap 3"` → CAP=3 (overrides the default top-5 cap)
- `$ARGUMENTS` = `"42 --run 7 --head 9f3a2b1 --auto-approve"` → AUTO_APPROVE=true

`--head <sha>` is set by the dispatcher when the action fires inside an `epic_integration` chain (#1916). Since #2353 this skill runs **before** `merge-to-base` and `delete-branch`, so the child's feature branch still exists and `origin/<epic>...<HEAD_SHA>` is the child's *own un-merged change*. The flag pins the diff to the implement run's tip so it is the same diff every time, regardless of which branch the dispatcher left checked out. (Pre-#2353 this skill ran after the merge, when `<HEAD_SHA>` was already an ancestor of `origin/<epic>` — the three-dot diff collapsed to empty and the agent was tempted to reconstruct a diff by other means. §3 forbids that; the ordering fix removes the temptation.)

`--auto-approve` is set by the dispatcher when the owning workflow has the per-workflow auto-approve-gates toggle on (#2349). It is the standing "yes" that carries the grow end-to-end (#2674): it bypasses the §8 filing gate, and — for the follow-ups §9.5 routes *to* the originating workflow (#2673) — when the workflow's `auto_execute` is also on the dispatcher then runs those grown members and ships the second wave hands-off. It is not a standing yes to the *destination*: a finding routed to the backlog (§9.5) lands on an unstarted epic under `--auto-approve` exactly as it does without it. Every other step (skip rules, cap, file-only boundary, the §6 reconcile: fine-key deduplicate-and-close plus coarse-key umbrella-append) is unchanged. Absent the flag, the §8 gate stays in force and the grown members land runnable but idle.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` and `gh` command.

## 1. Validate branch

```bash
git branch --show-current
```

If RUN_ID or HEAD_SHA is present, skip branch-name validation — the dispatcher resolved the diff coordinates and the working tree is no longer authoritative. Otherwise require the branch to start with `fix/<ISSUE>-`, `feat/<ISSUE>-`, `refactor/<ISSUE>-`, `chore/<ISSUE>-`, or `docs/<ISSUE>-`. If not, stop and tell the user:

> "Current branch `<branch>` does not belong to issue #<ISSUE>. Check out the correct branch first."

## 2. Apply the rule

The mandatory-reads block already loaded the authoritative `propagation-scan` rule — the source of truth on when propagation applies, when to skip, the file-only boundary, the cap, and the human gate. Evaluate the current diff against it; if the rule says "skip," emit `status: skipped` with the reason and stop.

## 3. Gather the diff

```bash
BASE="${BASE_BRANCH:-$(devwatch --repo "$REPO" branches dev)}"
if [ -n "$HEAD_SHA" ]; then
  git diff "origin/${BASE}...${HEAD_SHA}"
else
  git diff "origin/${BASE}...HEAD"
fi
```

When `--head <sha>` is supplied the diff is pinned to that SHA — deterministic across re-runs and immune to checkout state. The implement run's tip SHA is recorded on the `agent_runs` row by `/fix-issue` / `/feat-issue`; the dispatcher resolves it at trigger time and passes it here.

**Run exactly the diff command above — nothing else.** The single `git diff` invocation in this section is the *only* sanctioned way to obtain the diff. The two refs (`origin/${BASE}` and `${HEAD_SHA}` / `HEAD`) come from the dispatcher; you do not choose them.

**Fail closed on an empty or degenerate diff — never reconstruct one.** If the command above prints nothing, or the diff is purely cosmetic (formatting or import reorder), or limited to a dependency bump (`uv.lock`, `package-lock.json`, `package.json` deps), emit `status: skipped` with the reason and stop. An empty diff means *there is nothing to scan* — it does **not** mean "find the change some other way." You must **not** improvise a substitute diff by any means: not `git show <sha>`, not `git diff <sha>^ <sha>`, not diffing against a different base, not a commit's own patch, not the working tree. If the sanctioned diff is empty, the correct and only output is `status: skipped`.

This matters most under `--auto-approve` (§8), where no human reviews what you file: auto-approve is a standing "yes" to the proposed list, **never** a licence to manufacture a list from a reconstructed diff. "Nothing to scan" is a valid, expected outcome — report it and stop.

## 4. Identify candidate patterns

Read the diff. Identify each change that might propagate and classify it into one of:

- **new_helper** — a function, class, hook, arrow-const, or interface was added that might replace inline duplication elsewhere.
- **new_pattern** — a reusable shape (error handling, logging, validation, pagination) appears in two or more added sites and was not present before.
- **perf_fix** — a loop replaced by a batch / bulk / parallel call, a blocking call made async, or a comprehension replacing an appending loop.
- **bugfix_shape** — a defensive guard (None / empty / bounds), a corrected conditional, an off-by-one boundary fix.

For each candidate, note:

- A **one-line summary** of the change.
- The **originating evidence** — `file:line` in the diff.
- The **class** — which of the four kinds above.
- The **helper-or-signal** — the named primitive the change introduces (the function / class / hook / arrow-const for `new_helper`; the signal keyword or regex that names the shape for `new_pattern` / `perf_fix` / `bugfix_shape`). This is the same token you grep for in §5.
- The **search targets** — glob patterns narrowing where to scan (`**/*.py`, `server/**`, etc.).
- The **verdict** — `completion` or `pre-existing`, from the revert test below. It decides where the candidate's findings are tracked (§9.5).

If no candidate fits any of the four kinds, emit `status: skipped` with reason `"no propagation-firing patterns in diff"` and stop.

### The verdict — did the scanned diff create this? (#3798)

One question settles it:

> **Would this site still be a defect if the scanned diff were reverted?**

- **No → `completion`.** The diff created the inconsistency: the site was consistent before and is contradictory after. Bounded by the diff, so naturally one or two items, and it genuinely belongs on this branch. Example: the diff introduced the shared bound `FOLDER_SUBTREE_MAX_DEPTH` and left one site still carrying the bare literal — before that diff the literal duplicated nothing.
- **Yes → `pre-existing`.** The site was already like that and still is; the diff only revealed it. Real work, but not this workflow's work. Example: an edge write that never confirms it landed, broken long before the feature that surfaced it.

**Ambiguity resolves to `pre-existing`.** When you cannot answer the revert question cleanly, the verdict is `pre-existing`. This is not a tie-break for tidiness — it is the cheap direction. A wrong `pre-existing` call costs one line on a backlog list; a wrong `completion` call costs a branch, a pipeline run, and another scan that finds five more things (#3797). The expensive outcome is the one that has to be justified.

**The verdict is a property of the coarse key, not of one site.** What makes the sites inconsistent is the same helper-or-signal for all of them, so they share one answer to the revert question and one destination. Different keys in the same run can — and routinely do — reach different verdicts.

**Record it; never leave it implicit.** The verdict goes on the candidate here, onto every per-site issue filed for the key (§9), and into the summary as a destination (§10). A verdict nobody can read is a decision nobody can review, and a wrong call has to be visible rather than silent.

### Two keys per candidate — fine and coarse

Every candidate carries **two** dedup keys, used at different scopes:

- **Fine key — `(file, overlapping line range)`.** Identifies one candidate *site*. Scoped to a single scan target; it is what §6a uses to prevent re-filing a tracked site and to collapse exact-duplicate per-site issues. Recorded in each per-site issue's `**Candidate site:**` line.
- **Coarse key — `(class, helper-or-signal)`.** Identifies one mechanical *pattern* — e.g. `(new_helper, resolveMutationErrorMessage)`, `(bugfix_shape, none-guard-on-user-lookup)`. Every site that adopts the same helper-or-signal shares this key, regardless of which file it lives in or which scan target surfaced it. The coarse key is **stable across scan runs and across epic children** — it is the cross-run accumulator's join key. Serialise it as `<class>:<helper-or-signal>` (lower-kebab the helper-or-signal when it is a free-text signal; keep the verbatim symbol for `new_helper`).

The fine key is per-site, per-target. The coarse key is per-pattern, codebase-wide. §6b reads the coarse key to find an existing umbrella; §6a and §9 use the fine key for per-site dedup and per-site filing.

## 5. Scan the codebase

For each candidate:

1. Grep the repository for sibling sites using `rg` (ripgrep), scoped to the candidate's search targets:
   ```bash
   rg -n --glob '<search_target>' '<signal keyword or regex>'
   ```
2. Filter the hits:
   - **Drop** hits inside files the current diff already modified (the originating work).
   - **Drop** hits inside vendor / generated / lockfile paths.
   - **Drop** hits that already use the new helper / pattern (import check or call-site match).
3. **Rank** remaining hits by confidence — higher rank for matches in the same module / area, the same language, and lines whose shape matches the candidate's class.

Diff-scoped rule: **only patterns derived from the current diff.** Do not sweep the codebase for unrelated patterns.

## 6. Reconcile against existing propagation issues

`/propagation-scan` is idempotent by design: re-running it on the same target must not re-file candidates that are already tracked, and it collapses duplicate `propagation:` issues that earlier runs left behind. This step runs **always** — it is independent of the §8 human gate and of `--auto-approve`. It is the one place the skill is permitted to touch a pre-existing issue, and only to **close** an exact duplicate (§6a) or **append** to an open umbrella (§6b).

The step has three passes over two different keys:

- **§6a — fine key `(file, line range)`, scoped to this scan target.** Prevents re-filing a tracked site and closes exact per-site duplicates via `devwatch close-folded-issue` (close as not-planned + step-fold in one operation, #3616). The fine-key dedup logic is unchanged; only the close verb folds the now-orphaned step.
- **§6b — coarse key `(class, helper-or-signal)`, codebase-wide.** Discovers an **open umbrella** for the pattern and **appends** new sites to it instead of filing fresh per-site issues. This is the cross-run accumulator: state lives on GitHub (the umbrella issue), read via `gh issue list` on the request path — **no devwatch table, no cache column** (Rule 3).
- **§6c — threshold escalation, coarse key.** For a coarse key with **no** open umbrella yet, count its sites (open per-site issues already filed for the key, plus this run's candidates). When an **eligible** key crosses the threshold (>5 sites, or the key recurs across >1 scan run), **create one umbrella from scratch**, move the sites into its checklist, and close the folded per-site issues onto it (#2517). Eligible means a mechanical sweep, or a `bugfix_shape` key that has **graduated** — its rule is committed under `.claude/rules/` carrying the key's `propagation-key:` marker and specifying the per-site treatment (#155). An ungraduated `bugfix_shape` key never escalates — each site stays an individual per-site issue. That is a bar on *collapsing* the key, not on where its findings are tracked: §9.5 routes a `bugfix_shape` key like any other (#3800). This is the single place the scanner is allowed to *create* an umbrella; see **Boundary**.

### 6a. Fine-key dedup against this scan target's per-site issues

First, list the propagation issues already filed against this scan target:

```bash
gh issue list --repo "$REPO" --state all --limit 200 \
  --search "propagation: in:title" --json number,title,state,body
```

Keep only issues that are genuinely children of **this** scan target — their body carries `Surfaced by /propagation-scan on #<ISSUE>` and a `child-of` link to `#<ISSUE>`. Discard the rest (propagation issues for other targets are not yours to touch). For each kept issue, read its **candidate site** (`<file>:<line range>`, recorded in the body). That `(file, overlapping line range)` is the **dedup key**.

**Prevent — never re-file a tracked site.** Drop every current candidate whose dedup key matches an **open** existing propagation issue for this scan target. Do this *before* the cap (§7) so the cap counts only genuinely-new sites, not slots wasted on already-tracked ones. Note each dropped candidate and the issue number it duplicates, for the summary.

**Close — collapse redundant duplicates.** Group the kept propagation issues by dedup key. When a key has more than one issue, the **canonical** one is the lowest-numbered (it was filed first; if its work already shipped it is closed/merged — that is still the canonical). Close every *other* **open** issue for that key **and** reconcile its now-orphaned devwatch step in **one command**, pointing at the canonical:

```bash
devwatch --repo "$REPO" close-folded-issue <dup> \
  --comment "Duplicate of #<canonical> — same propagation candidate (\`<file>:<line range>\`). Tracked there."
```

The server closes `<dup>` as **not planned** with your comment and — only on a successful close — reconciles its workflow step to `CANCELLED` via #2516's fold. A redundant duplicate is itself a filed workflow member (§9.5 routed it to one), so closing it any other way leaves its step `pending`: the run-lifecycle fold fires only for an *active* run (#2514), so only the 10-minute reconcile tick would eventually catch up, and the dashboard shows a phantom `pending` step beside a `Closed` stage for that whole window (#3574). Folding the step into the close closes that window — the close and the fold land together or not at all.

Rules for this step — they are narrow on purpose:

- **Close, never edit.** The only mutation permitted is **closing** a redundant duplicate. Never re-title, re-label, re-parent, re-link, or reopen any issue.
- **Keep exactly one per key.** Never close the canonical (lowest-numbered) issue for a key, whether it is open or already closed. After this step each dedup key has exactly one non-duplicate issue.
- **Scope is this scan target's propagation issues only.** Never close an issue that is not a `propagation:` child of `#<ISSUE>` — not an unrelated issue, not a propagation child of a different target.
- **No creation here.** This step only drops candidates (prevention) and closes redundants (cleanup); it never files an issue.

If there are no existing propagation issues for this scan target, this step is a no-op — proceed.

### 6b. Coarse-key umbrella discovery and append

The fine key in §6a is scoped to one scan target, so it cannot stop the same mechanical pattern from re-filing on the *next* epic child — a different scan target with an empty fine-key set. The coarse key closes that gap. Its accumulator is a single **umbrella issue** on GitHub, discovered live on the read path.

**The umbrella marker.** An umbrella issue carries a `propagation-umbrella: <class>:<helper-or-signal>` line in its body — the serialised coarse key from §4. That marker line is the join key; it is what makes an umbrella discoverable by `gh issue list` and what binds it to a pattern rather than to any one scan target. (§6b only *discovers* and *appends to* an umbrella that already exists. Creating one from scratch when none exists yet — the threshold-escalation path — is §6c. Both append via comment to a marker-bearing umbrella; only §6c is permitted to mint the marker.)

For each surviving candidate (after §6a), discover an open umbrella for its coarse key:

```bash
gh issue list --repo "$REPO" --state open --limit 200 \
  --search "propagation-umbrella: in:body" --json number,title,body
```

Match an umbrella to a candidate when the umbrella body contains a line exactly equal to `propagation-umbrella: <class>:<helper-or-signal>` for that candidate's coarse key. The marker comparison is exact on the serialised key — never a substring or fuzzy match. At most one open umbrella may match a coarse key; if two open umbrellas carry the same marker that is a data error — match the lowest-numbered and note the collision for the summary, do not append to both.

For every candidate whose coarse key matches an open umbrella:

1. **Skip the per-site issue.** Do not file a fresh `propagation:` child for this candidate in §9 — the umbrella is its tracked unit of work. Remove it from the to-file list *before* the cap (§7), so cap slots go only to candidates with no umbrella.
2. **Append the site to the umbrella's checklist** — only if the site is not already listed. Read the umbrella body, check whether a checklist line already references this candidate's `<file>:<line range>` (the fine key); if it does, the site is already tracked there — skip silently. Otherwise append one unchecked checklist item. The summary is your own prose and sits beside a backticked path, so pass the body through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

```bash
BODY=$(cat <<'BODY_EOF'
Appending propagation site (surfaced by /propagation-scan on #<ISSUE>):

- [ ] `<file>:<line range>` — <one-line summary>
BODY_EOF
)

gh issue comment <umbrella> --repo "$REPO" --body "$BODY"
```

   Use an issue **comment** to append, not a body rewrite. Appending via comment is additive and auditable; rewriting the umbrella body risks clobbering checklist items other runs added and is out of this child's scope. The umbrella owner's tooling folds appended-site comments into the checklist; this skill's job ends at recording the site.

Rules for §6b — narrow on purpose:

- **Append-only, to an existing open umbrella.** §6b's only mutation is **adding a site** (via comment) to an umbrella that already carries the matching marker. §6b never *creates* an umbrella, never rewrites an umbrella body, never re-labels / re-titles / re-parents / reopens one. Creating an umbrella from scratch and folding existing per-site issues into it is §6c (#2517), the threshold-escalation path.
- **No umbrella → hand off to §6c, do not file yet.** A candidate whose coarse key has no open umbrella does **not** fall straight through to per-site filing — it is handed to §6c, which counts the key's sites and either escalates it to a fresh umbrella (an eligible key at/above threshold) or releases it to the cap (§7) and per-site filing (§9) below threshold. §6b raises no umbrella itself.
- **Idempotent.** Re-running the scan re-discovers the same umbrella and finds each site already in its checklist — so the append is a silent no-op. No duplicate checklist lines, no duplicate comments for a site already recorded.
- **GitHub is the only store.** The umbrella issue *is* the accumulator. Discovery is the `gh issue list` read above, run on every scan — there is no devwatch row, no cache column, no local file mirroring umbrella state (Rule 3).

If no candidate's coarse key matches an open umbrella, §6b is a no-op — every surviving candidate passes to §6c.

### 6c. Threshold escalation — create an umbrella for a mechanical sweep

§6b appends to an umbrella that *already exists*. But the first time a mechanical pattern floods — before any umbrella has been minted — there is nothing to append to, and §6b lets the candidates fall through to per-site filing. That is the flood (#2514): five fresh per-site issues per child step, scan after scan. §6c is the relief valve. It promotes a coarse key to a **single umbrella** the moment the key's running total crosses a threshold, folding the per-site issues already filed for the key into the new umbrella's checklist — and it is the **single place in this skill permitted to create an umbrella** (see **Boundary**).

Run §6c **only** over candidates that survived §6a and matched **no** open umbrella in §6b. A candidate already routed to an umbrella (§6b) is tracked; do not re-promote it.

**Eligible classes — mechanical sweeps, and graduated `bugfix_shape` keys.** §6c escalates a coarse key **only** when collapsing its sites onto one checklist loses nothing. That holds by construction for a mechanical sweep — `new_helper`, `new_pattern`, or `perf_fix`: one find-and-replace refactor whose natural scope is the whole codebase. It does **not** hold for a `bugfix_shape` key while its shape is still being worked out: a recurring bug-shape is semantically distinct per site (each guard may be subtly different and merits individual review), so an **ungraduated** `bugfix_shape` candidate stays an individual per-site issue and passes straight through to the cap (§7) and per-site filing (§9). The class is the candidate's coarse-key class from §4; do not infer it from site count.

**Graduation — the `bugfix_shape` exit condition (#155).** "Each guard merits individual review" is a justification that is *spent* once that review has happened and converged on one rule. Without an exit condition it never expires, and one settled shape keeps filing at ~2 new sites per closed fix, generation after generation. A `bugfix_shape` coarse key **graduates** to mechanical-sweep eligibility when **both** hold:

1. **A rule under `.claude/rules/` carries the key's marker** — a line exactly equal to `propagation-key: <class>:<helper-or-signal>`. Exact-match on the serialised coarse key, the same discipline as §6b's `propagation-umbrella:` and §9.5's `propagation-epic:` markers — never a substring, never a fuzzy match on a rule's title or prose.
2. **That rule section specifies the per-site treatment** — enough that a reader picks the right fix at a *new* site without re-deriving it. For a copy-ownership key that means which shape takes a shallow copy, which takes a deep one, and where the copy goes. A rule that only names the hazard has not graduated the key; one that tells you what to write at each site has.

Check both with one read, then open the section the marker hits and confirm (2):

```bash
rg -n 'propagation-key: <class>:<helper-or-signal>' .claude/rules/
```

A graduated key is a find-and-replace refactor by definition, so it escalates on the same thresholds as a mechanical sweep and takes the same escalation path below. Absent the marker, the default stands — file individually.

**Why graduation is yours to evaluate.** Both conditions are a file read: the marker line exists or it does not, the treatment is written down or it is not. Neither asks you to judge a class's history across waves. That judgment already happened — in the reviews that produced the rule — and the committed rule *is* its record. That is exactly what separates graduation from the human-gated exception below.

**An ungraduated key is still not yours to call settled (#3553).** For a key with no `propagation-key:` marker you never mint a `bugfix_shape` umbrella, whatever the count and however identical the sites look to you. But that constrains §6c's *automatic* minting, not a human's judgment. A human may decide such a class has **settled** — its fix is byte-identical at every site, and per-site review across prior waves has stopped surfacing per-site divergence — and hand-mint an umbrella for it. That path is sanctioned, not a violation: #3535, #3552, and #3563 each took it for the quoted-heredoc class after per-site judgment had been exercised and had converged on one fix. Committing the class's rule with its marker is the durable version of the same decision.

No machinery changes for either route. §6b matches an umbrella on its marker line alone, with no class restriction, so a hand-minted or graduation-minted `bugfix_shape` umbrella is discovered and appended to like any other — your §6b append is the *correct* behaviour when you meet one, not an exception to it. What stays yours is the per-site default: absent graduation and absent an existing umbrella, every `bugfix_shape` candidate is filed individually. Unrecorded "settledness" is a judgment about a class's history across waves, which a single scan run cannot see — that is why it is the human's call until the rule is written down.

**The gate stops here — it does not reach §9.5 (#3800).** Everything above is about *collapsing*: folding N sites onto one umbrella checklist and closing them, which asserts the class has settled. §9.5 decides something else entirely — **which workflow or epic tracks the finding**. A routed `bugfix_shape` group arrives at its destination as N open per-site issues, each still reviewed on its own; nothing is folded and nothing is closed. So §9.5 routes it by the same decision as every other class, including minting the destination *epic* when the coarse key has none yet. Minting an epic is not minting an umbrella — the `propagation-epic:` marker names a destination that accumulates *member issues*, `propagation-umbrella:` an accumulator that swallows *checklist sites* (§9.5). Do not carry this section's `never` into that one.

**Count the key's sites.** For each surviving eligible coarse key, count its sites:

```bash
gh issue list --repo "$REPO" --state open --limit 200 \
  --search "propagation: in:title" --json number,title,state,body
```

The site count for a key is the number of **open** per-site `propagation:` issues whose body's `**Class:**` and helper-or-signal serialise to this coarse key (children of any scan target — the coarse key is codebase-wide, so umbrella escalation spans every child that filed for the pattern), **plus** this run's surviving candidates for the same key not already among them. Dedup by fine key so a candidate that already has an open per-site issue is counted once, not twice.

**The threshold — escalate when either holds.** Promote the key to an umbrella when **either** condition is met:

- **>5 sites.** The key's site count (open per-site issues + this run's new candidates) exceeds 5. One umbrella holds the whole sweep; the cap (§7) never has to discard the overflow.
- **Recurs across >1 scan run.** The key already has an open per-site issue from an *earlier* scan run (a different scan target, or the same target on a prior run) **and** this run surfaces at least one fresh candidate for it. A pattern that re-appears across runs is the cross-child flood by definition — escalate it on sight, before it reaches five.

Below both thresholds the key stays per-site — it falls through to the cap (§7) and per-site filing (§9) unchanged. The threshold is a floor, not a ceiling: once crossed, **all** of the key's sites move onto the umbrella, including over-cap ones (§7).

**Escalate — create the umbrella and fold the per-site issues in.** When an eligible key crosses the threshold:

1. **Create one umbrella issue** for the coarse key, carrying the marker line so §6b discovers it on every later scan. The body is your own prose and its checklist names paths in backticks, so pass it through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

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

   The umbrella is a flat `child-of` child of the scan target exactly like a per-site issue — `create-issue` derives the epic edge (§9, #2095). The `## Sites` checklist seeds **every** site for the key: each folded per-site issue's candidate site **and** this run's surviving candidates for the key (one unchecked item each, deduped by fine key).

2. **Close the folded per-site issues onto the umbrella.** Every **open** per-site `propagation:` issue for the coarse key is now redundant — its site lives on the umbrella's checklist. Close each one **and** reconcile its now-orphaned devwatch step in **one command**, pointing at the umbrella. This generalises §6a's close-duplicate mutation from *same fine key* to *same coarse key* (#2517):

   ```bash
   devwatch --repo "$REPO" close-folded-issue <folded> \
     --comment "Folded into propagation umbrella #<umbrella> for the \`<class>:<helper-or-signal>\` mechanical sweep. The site (\`<file>:<line range>\`) is tracked on the umbrella's checklist."
   ```

   The server closes `<folded>` as **not planned** with your comment and — only on a successful close — reconciles its workflow step to `CANCELLED` via #2516's fold. Closing a per-site issue any other way leaves its step `pending`: the run-lifecycle fold fires only for an *active* run (#2514), so only the 10-minute reconcile tick would eventually catch up, and the dashboard shows a phantom `pending` step beside a `Closed` stage for that whole window (#3574). Folding the step into the close closes that window — the close and the fold land together or not at all. Close, never edit: the only mutation to a folded issue is closing it onto the umbrella. Never re-title, re-label, re-parent, re-link, or reopen one.

3. **Append this run's surviving candidates** for the key to the umbrella — they are already on the `## Sites` checklist from step 1, so no separate comment is needed. Remove every candidate for an escalated key from the to-file list (it is tracked on the umbrella), **before** the cap (§7).

Rules for §6c — narrow on purpose:

- **Eligible keys only.** Escalation fires for `new_helper` / `new_pattern` / `perf_fix` keys, and for a `bugfix_shape` key that has **graduated** — a `.claude/rules/` section carrying its `propagation-key:` marker and specifying the per-site treatment (#155). An ungraduated `bugfix_shape` key never escalates — its sites stay individual.
- **One umbrella per coarse key.** A key has at most one umbrella. If §6b already found an open umbrella for the key, §6c does not run for it (§6b appended; nothing to escalate). §6c creates an umbrella **only** when none exists and the threshold is crossed.
- **Idempotent.** A later scan re-discovers the umbrella §6c created (§6b) and appends to it; it never creates a second umbrella for the same marker. Re-running §6c when the per-site issues are already closed and folded is a no-op.
- **GitHub is the only store (Rule 3).** The umbrella issue is the accumulator; the site count is a `gh issue list` read on the request path. No devwatch table, no cache column.

Below the threshold for every surviving key, §6c is a no-op — proceed with the full surviving candidate list to §7.

## 7. Cap and present

With the coarse key in place (§6, #2515/#2517), the cap governs **distinct patterns** per run, not raw sites. Reaching here, every candidate routed to an umbrella (§6b append or §6c escalation) is already tracked — those are gone from the to-file list. What remains is the per-site candidates with no umbrella: at most one entry per fine key, grouped by coarse key.

Truncate to the top `CAP` **distinct coarse keys** (default `5`), ranked by confidence. Within a kept key, every surviving site is filed (§9) — the cap counts patterns, not sites, so a single pattern with eight below-threshold sites consumes one cap slot, not eight.

**Over-cap same-key sites append, never drop.** When more sites surface for a coarse key than this run will file individually — because the key crossed §6c's threshold, or because a kept key's sites would otherwise exceed what one issue should hold — the over-cap sites **accrue onto the umbrella's checklist** (§6c created it, or §6b found it), they are **not silently discarded**. Dropping over-cap sites was the pre-#2517 behaviour that lost mechanical-sweep coverage; with the umbrella as accumulator there is now always somewhere for an over-cap same-key site to land. Only **distinct coarse keys** beyond the cap are deferred to a later run — and those re-surface and accrue on the next scan, since the coarse key is stable across runs (§4). If the count of distinct keys exceeds the cap, record the overflow for the summary.

Print the list, one entry per opportunity. Each entry carries its §4 verdict and the destination that verdict routes it to (§9.5), so the human gate below sees *where* the findings land, not only that they exist:

```
## Propagation opportunities — 4 of 12 (cap: 5)

1. new_helper — format_currency
   Originating site: server/billing/invoice.py:142
   Candidate sites:
     - server/reports/invoice.py:88
     - dashboard/src/components/price-tag.tsx:28
   Verdict: completion — format_currency did not exist before this diff
   Destination: this workflow (root #<ISSUE>)
   Proposed action: extract call sites to use format_currency

2. bugfix_shape — None guard on user lookup
   Originating site: server/routers/admin.py:67
   Candidate sites:
     - server/routers/reports.py:98
     - server/services/user_service.py:211
   Verdict: pre-existing — both sites were unguarded long before this diff
   Destination: backlog epic for bugfix_shape:none-guard-on-user-lookup (§9.5)
   Proposed action: apply the same None-guard pattern
```

## 8. Human gate

**If `AUTO_APPROVE` is true** (the `--auto-approve` flag was passed): skip this gate entirely. Proceed to §9 with the **full** capped list — do not prompt, do not wait. The operator opted the owning workflow into auto-approval, which is a standing "approve all" for this run. Everything else (the cap, the file-only boundary, the skip rules) still applies.

Otherwise (the default), **do not** file issues until the human confirms. Present the list and wait.

- **Approves all** → proceed with the full list.
- **Approves a subset** → proceed with the selected subset only.
- **Rejects** → emit `status: skipped` with reason `"user declined proposed opportunities"` and stop. Not a failure.

Agent-to-agent confirmation is not sufficient for this skill. Filing issues has larger blast radius than writing a rule; the human gate is load-bearing — which is why the bypass is opt-in per workflow and off by default.

## 9. Execute — file issues

This step files per-site issues only for the candidates that survived §6 — candidates whose coarse key matched an open umbrella were already appended to it in §6b and removed from the to-file list, so they never reach here. A candidate with no matching umbrella is filed per-site exactly as before.

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

2. For each approved opportunity, create the issue. Pass exactly one `--parent` — the scan target from `$ARGUMENTS`. The title's `<one-line summary>` slot and the body are both your own prose and name files and symbols in backticks by convention, so pass each through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

```bash
TITLE=$(cat <<'TITLE_EOF'
propagation: <one-line summary>
TITLE_EOF
)

BODY=$(cat <<'BODY_EOF'
<markdown body>
BODY_EOF
)

devwatch --repo "$REPO" create-issue \
  --type feature \
  --title "$TITLE" \
  --body "$BODY" \
  --area <area> \
  --priority <P2-medium|P3-low> \
  --parent <ISSUE> \
  --run-id <RUN_ID> \
  --no-claim-run
```

`--no-claim-run` is mandatory here. This skill files several issues
against one `--run-id`; without the flag each `create-issue` would
overwrite the run's `github_issue`/`summary` and the run row would end
up pointing at the last child instead of the scan target. The run's
terminal status/summary is owned by the step-11 `agent-update` below —
written once, not re-stamped per child.

**Title format:** every propagation issue title is `propagation: <one-line summary>` — the `propagation:` prefix exactly once, then the candidate's one-line summary. No other prefix, no descriptive-only title.

**Parentage — pass only the scan target; `create-issue` derives the epic edge.** Every propagation issue carries `--parent <ISSUE>` — the scan target — and that is the **only** `--parent` you pass. When the scan target is itself a child of an epic, `create-issue` walks the scan target's `child-of` chain in code and writes the `child-of: <epic>` edge for you, so the issue groups beside the epic's other children instead of sitting one hop below them (#2095). You do **not** resolve the epic, and you do **not** pass it as a second `--parent` — the epic edge is a deterministic property of issue creation, not something the skill assembles per run.

Do **not**:

- pass any `--parent` other than the scan target — not the resolved epic (`create-issue` adds it), not a non-epic intermediate ancestor, not a second epic deeper in the chain,
- pass the workflow root (or any workflow the scan target is a step in) as a `--parent` — the workflow is execution metadata, not issue-tree parentage,
- omit `--parent <ISSUE>` — the scan-target link is always present.

The scan target records *why* the issue exists; its epic records *where the work groups*. Those are the only two `child-of` links a propagation issue may carry at filing time — and `create-issue` is the single thing that assembles them. §9.5 may then swap the *epic* half for the finding's routed destination; the scan-target half is never touched, so provenance survives the move.

Issue body template:

```markdown
## Propagation candidate

Surfaced by `/propagation-scan` on #<ISSUE>.

**Originating change:** `<file:line>` — `<commit SHA>`
**Candidate site:** `<file:line range>`
**Class:** `<new_helper | new_pattern | perf_fix | bugfix_shape>`
**Verdict:** `<completion | pre-existing>` — <why the revert test answered that way>
**Proposed action:** <one-line summary>

## Evidence

<three-line window of the grep match>

## Why this is tracked, not inlined

The originating PR is scoped to #<ISSUE>. This site is a candidate for the same treatment but belongs in its own branch and review. See the propagation-scan rule.
```

**Priority** defaults to `P3-low` (opportunistic, not blocking). Escalate to `P2-medium` when the originating kind is `bugfix_shape` — a bug pattern that recurs is higher-signal than a helper opportunity.

**Area** is inferred from the candidate path: `server/` → `backend`, `dashboard/` → `frontend`, `src/` (CLI roots) → `cli`, infra paths → `infrastructure`.

After each `create-issue`, note the returned issue number — you need them for the summary.

The `child-of` set is **not** something the skill assembles or self-verifies any more. `create-issue` writes the scan-target edge and — when the scan target has an epic — the epic edge, both in code (#2095). There is no per-run conditional for an agent to skip and no blind re-read for it to rubber-stamp: a missing or wrong `child-of` set is now a `create-issue` bug, caught by that command's own tests, not a propagation-scan responsibility.

## 9.5. Route each finding to its destination (#3798)

The follow-ups filed in §9 used to join the originating workflow unconditionally — *where they group is a deterministic property of the root binding*, with nothing asking whether the finding had anything to do with what that workflow was building. So every running epic became the billing address for whatever code health the scan happened to surface while it ran: `wunova-core-app` epic #1473 planned 4 children and carries 29 members, 25 of them unrelated (#3797). The destination is now the §4 verdict's, decided before anything is filed.

**The default inverts.** A finding lands in the backlog unless it is shown to belong to the running work. Attaching to the running workflow is a real route, not a formality — it is simply the one that has to be earned.

### The routing table

| Verdict (§4) | Class | Destination |
|---|---|---|
| `completion` | any | **the scanning workflow** — attaches automatically, exactly as before |
| `pre-existing` | any, key matches an open backlog epic | **that epic** — the theme is already tracked (#3799) |
| `pre-existing` | any, no match | **a fresh epic in the backlog**, unstarted |

**Class does not appear in this table (#3800).** Routing is decided by the verdict alone, for every candidate class — `new_helper`, `new_pattern`, `perf_fix` and `bugfix_shape` alike. The class still governs *collapsing* (§6c's escalation, and `/propagation-consolidate`'s fold), which is a different act; see below.

Candidates routed to an umbrella in §6b / §6c never reach here: they were removed from the to-file list before the cap and are tracked on the umbrella, not as per-site follow-ups. This step routes only what §9 filed.

### `completion` — nothing to run

§9's `create-issue --parent <ISSUE>` already wrote the `child-of` edge, and the server's auto-attach resolves the scan target's workflow root and appends the follow-up as a step (Rule 13 — the write is the server's; the skill never touches `workflows` / `issue_links`). "Epic vs standalone" is a member count, not a kind (#2666): the workflow grows in place, keeping its identity and behaviour config. There is no command here.

**Running the grown members.** When the owning workflow's `auto_execute` is on **and** this scan ran under `--auto-approve` (#2674), the grow carries hands-off: the dispatcher chains the newly-attached members and ships the second wave (one PR off the `epic/<root>-<slug>` integration branch) with no manual step. Absent either, they land as runnable members you start when ready.

### `pre-existing` — move the key's findings onto the key's own epic

One call per coarse key, naming every per-site issue §9 filed for that key. The epic is named for **the group's own theme**, not the scan target's — that is the whole point of moving it. Which epic is a lookup, not a judgment: **does an open epic already carry this key's marker?**

**The routing marker.** Every epic this step mints carries a `propagation-epic: <class>:<helper-or-signal>` line in its body — the serialised coarse key from §4. That line is the join key: it is what lets the *next* run find the epic instead of minting a second one for a theme already on file (#3799). It is deliberately **not** §6b's `propagation-umbrella:` marker, because the two accumulate different things: an umbrella issue accumulates *checklist sites* (§6b appends a line to it), a routing epic accumulates *member issues* (this step moves whole findings onto its workflow). One marker for both would leave a candidate ambiguous between "append a line" and "move the issue".

**Discover the key's epic:**

```bash
gh issue list --repo "$REPO" --state open --limit 200 \
  --search "propagation-epic: in:body" --json number,title,body
```

Match an epic to a coarse key when its body contains a line **exactly equal** to `propagation-epic: <class>:<helper-or-signal>` for that key — never a substring or a fuzzy match, and never an issue that carries only the `propagation-umbrella:` marker (that is §6b's accumulator, not a destination). At most one open epic may match a key; if two do, that is a data error — route into the lowest-numbered one and note the collision for the summary.

#### A match — route into the epic that already exists

```bash
devwatch --repo "$REPO" regroup-onto-existing-epic \
  --epic <E> \
  --member <N> \
  --member <M>
```

One server-side operation, and it **creates nothing**: the epic and its workflow are already there, so the findings are moved onto them with the same re-home primitive the mint route uses. There is no `--approve` flag because there is no epic to approve — nothing is published, and a wrong destination is undone by re-running with the right one. The destination keeps its own status and autonomy toggles; the moved findings land as pending members with no run opened.

The command refuses **before moving anything** when `--epic` is not an epic, or roots no non-terminal workflow, naming which of the two failed — so a mistyped number cannot deposit a group into an unrelated issue's workflow. A finding already sitting on the destination is reported as left in place, which makes a re-run of the same routing a no-op rather than a duplicate.

#### No match — mint the epic in the backlog

The title and the body are your own prose, so pass each through its own **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

```bash
TITLE=$(cat <<'TITLE_EOF'
<the group's own theme>
TITLE_EOF
)

BODY=$(cat <<'BODY_EOF'
propagation-epic: <class>:<helper-or-signal>

<markdown body, including the child checklist>
BODY_EOF
)

devwatch --repo "$REPO" regroup-onto-new-epic \
  --title "$TITLE" \
  --body "$BODY" \
  --member <N> \
  --member <M> \
  --area <area> \
  --priority <P2-medium|P3-low> \
  --approve
```

The `propagation-epic:` marker line is **mandatory** in the body — without it the epic is invisible to the lookup above and the next run mints a duplicate for the same theme, which is the whole defect #3799 closes.

One server-side operation: it mints the epic, roots a `draft` workflow on it (an epic issue alone is only a label — issues cannot be members of it), and moves each `--member` across with the re-home primitive, so membership and the `child-of` graph never disagree about which integration branch the issues execute on. The new epic lands **unstarted** — `draft`, every autonomy toggle off, nothing queued — so a human sees it before any of it runs.

`--approve` is the routing-table policy asserted at the command, not a human gate you skipped: the §8 gate already governed *whether* these findings exist, and routing to the backlog is the cheap, reversible direction. It carries no class condition — the table has none.

**Both routes keep provenance.** Moved issues keep every other `child-of` they carry, so the link to the scan target survives and the discovery is never dropped, only re-homed. The one edge a move drops is the source workflow's own *root* — the edge that would otherwise keep routing the finding back to the workflow it just left. When the scan target is itself that root (a standalone issue whose own workflow grew), those are the same edge and it goes: the command names it in its output, so record it in §10 rather than treating the issue as unmoved.

**Report what it left behind.** Either command refuses to move an issue that has already executed — one with a branch or a run — and names each with its reason instead of half-moving it. Carry those into §10 verbatim; they stay on the scanning workflow. A group where *nothing* can move is refused outright — the mint route creates no epic, the existing-epic route touches nothing — a clean outcome, not a failure.

**Why the finding is filed first and moved second.** `regroup-onto-new-epic` moves *members*: the issue must exist and be in a workflow before it can be re-homed. So a `pre-existing` finding is created by §9 like any other, then moved in this same step before the run ends. The window is inert — the auto-attach's own chain advance defers while a member is in flight, and this scan *is* that in-flight member, so nothing dispatches on the scanning workflow between the file and the move. The decision is still made before filing; only its last write lands after.

### Routing a `bugfix_shape` group is not collapsing it (#3800)

A `pre-existing` `bugfix_shape` key takes the route above like any other — the match branch when its `propagation-epic:` marker already exists, the mint branch when it does not, `--approve` on the mint exactly as for a mechanical sweep.

The human-minted-marker rule (`.claude/rules/critical.md`) is untouched by that, because it governs a different act:

- **Collapsing** — §6c minting an *umbrella* for the key, folding its sites onto one checklist and closing them. That asserts the class has settled — a claim about its history across waves you cannot observe from one run, and which you may act on only where a committed rule already records it (§6c's graduation clause, #155). **Forbidden for an ungraduated key** (§6c).
- **Routing** — this step, moving the key's findings onto the epic that tracks them. Every site arrives open and individually reviewable; nothing is folded, nothing is closed, nothing is asserted about the class.

Minting a routing epic is therefore not minting an umbrella, which is why the two carry different markers. Read §6c's `never` as binding only what §6c does.

The rule used to be applied here too, and the cost was concrete: every `bugfix_shape` finding stayed on whatever epic happened to be running, which is the most common class and so the majority of all findings. All 20 of `wunova-core-app` epic #1473's stray members are `bugfix_shape` — the re-homing machinery built for exactly them was forbidden from touching one (#3797).

### What this step still never does

It files nothing, edits no code, and mints no epic outside the `pre-existing` route above. It never re-parents an issue by hand — the `regroup-onto-*-epic` commands own both membership and the `child-of` graph, so the two move together or not at all.

## 10. Summary comment

Post a single summary comment on the parent `<ISSUE>`. **Every filed issue names its verdict and its destination**, not just the count filed — a routing call nobody can read is a routing call nobody can correct. Each per-issue line carries that candidate's own one-line summary — your prose — so pass the body through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

```bash
BODY=$(cat <<'BODY_EOF'
## Propagation scan — <N> issues filed

- #<N1> — <summary> — completion → this workflow
- #<N2> — <summary> — pre-existing → new backlog epic #<E>
- #<N3> — <summary> — pre-existing → existing backlog epic #<E2>
- #<N4> — <summary> — pre-existing (bugfix_shape) → new backlog epic #<E3>

Cap: <CAP> (distinct patterns). Overflow: <K> distinct key(s) deferred.
Dedupe (§6a): skipped <S> already-tracked candidate(s); closed <C> duplicate(s) — #<D1>→#<canonical>, ….
Umbrella (§6b): appended <A> site(s) to #<U1>, #<U2>; <P> already listed.
Escalation (§6c): promoted <E> mechanical-sweep key(s) to umbrella #<U3> (<G> site(s)); folded #<F1>, #<F2> → #<U3>.
Routing (§9.5): <C1> completion finding(s) attached to this workflow (root #<ISSUE>); <P1> pre-existing finding(s) moved onto new backlog epic #<E> (draft workflow <W>, not started); <P2> moved onto existing backlog epic #<E2> (workflow <W2>) — left in place: #<x> (already has a branch or a run).
See the propagation-scan rule.
BODY_EOF
)

devwatch --repo "$REPO" agent-comment \
  --issue <ISSUE> \
  --body "$BODY"
```

When nothing was deduped, drop the Dedupe line rather than printing zeros. Likewise drop the Umbrella line when no candidate matched an open umbrella, and drop the Escalation line when no key crossed §6c's threshold. Drop the Routing line when §9 filed no per-site follow-ups; keep the clauses of it that fired and drop the ones that did not — a run whose findings were all `completion` names no backlog epic, one whose findings were all `pre-existing` names no attached member, and a run that minted no epic (every key matched one that already existed) names no new epic. When the run filed nothing new but still closed duplicates, appended sites, escalated a key to a fresh umbrella, or routed follow-ups, the comment is still worth posting — it records the cleanup, the accrual, the consolidation, and the routing.

## 11. Emit the run report and record completion

First emit the run report (advisory — a failed post must never fail the step).
Write the fixed JSON skeleton, filling `notes` with the candidates filed and any
skipped (with the reason), and `stats` with the counts. Post it **before** the
`agent-update` status flip below so the report exists when completion hooks fire.

```bash
cat > /tmp/devwatch-report-<ISSUE>.json <<'JSON'
{
  "schema_version": 1,
  "notes": [
    {"category": "follow_up", "text": "Filed #<N> — <summary>"},
    {"category": "consideration", "text": "Skipped <candidate> — <reason>"}
  ],
  "stats": {"candidates_found": <N>, "filed": <M>, "skipped": <K>}
}
JSON

devwatch --repo "$REPO" agent-report \
  --run-id <RUN_ID> \
  --file /tmp/devwatch-report-<ISSUE>.json \
  || echo "  agent-report failed (advisory) — continuing"
```
Omit `--run-id` if RUN_ID is unavailable — the run is resolved from `DEVWATCH_AGENT_RUN_ID` instead.

Then update the agent-run trace (omit `--run-id` if RUN_ID is unavailable):

```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "Filed <N> propagation issues under #<ISSUE>"
```

For skipped runs (empty diff, cosmetic change, dependency bump, no firing pattern, user declined), still call `agent-update` with `--status completed` and prefix the summary with `Skipped —` so the dashboard can distinguish. The reason is your own prose, so it goes through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

```bash
SUMMARY=$(cat <<'SUMMARY_EOF'
Skipped — <reason>
SUMMARY_EOF
)

devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "$SUMMARY"
```

## Boundary

- **File-only.** Never edits code, never commits, never opens a PR.
- **Diff-scoped.** Only patterns derived from the current diff. No full-repo audits.
- **Capped.** Top `CAP` **distinct coarse keys** (default 5), not raw sites (§7, #2517). Over-cap same-key sites **append to the umbrella** (§6b/§6c), never dropped; only distinct-key overflow is counted and deferred.
- **Human-gated — once.** The user approves the opportunity list before any issue is created (§8); that single gate is load-bearing and agent-to-agent confirmation is not sufficient. §9.5's routing needs no second gate: a `completion` finding attaches by the root binding (#2673) with nothing to approve, and a `pre-existing` one lands on an epic that is **not running** — an unstarted fresh one, or an existing backlog epic whose own status and autonomy toggles the route never changes — so the human still sees it before any of it moves.
- **No cross-skill writes.** Does not write rules, docs, or code. `/issue-to-rule` handles rules; `/add-documentation` handles docs.
- **Every finding carries a recorded verdict, and the verdict picks the destination (§4, §9.5, #3798).** The revert question — *would this site still be a defect if the scanned diff were reverted?* — answers `completion` (the diff created the inconsistency) or `pre-existing` (the site pre-dates it), and **ambiguity resolves to `pre-existing`**. The verdict is written onto every filed issue and named per finding in the §10 summary; it is never implicit. `completion` findings auto-attach to the originating workflow as members by the root binding (#2592, #2673) — server-side on issue creation (Rule 13), the skill never writes `workflows` / `issue_links` directly — and then run under the workflow's own policy: hands-off with `--auto-approve` + `auto_execute` (#2674), else runnable-but-idle. `pre-existing` findings are moved onto the coarse key's own backlog epic — `devwatch regroup-onto-existing-epic` when an open epic already carries the key's `propagation-epic:` marker (#3799), `devwatch regroup-onto-new-epic` minting a fresh **unstarted** one when none does — keeping their `child-of` link to the scan target so provenance survives. **Routing carries no class condition (#3800):** a `bugfix_shape` key routes exactly like a mechanical sweep, because routing moves open issues between workflows rather than collapsing them onto an umbrella — the act the human-minted-marker rule (`.claude/rules/critical.md`) actually governs, and which §6c refuses for an ungraduated key. Candidates already grouped onto a propagation umbrella (§6b/§6c) are tracked there and never reach the routing step.
- **Never creates an epic except via §9.5's backlog route or §6c escalation. Only the second is class-gated (#3800).** The skill never passes `--epic` to `create-issue`. Two scoped paths create one: (1) §9.5's `pre-existing` route calls `devwatch regroup-onto-new-epic` for **any** coarse key **whose epic does not exist yet**, which mints the epic server-side with a `draft` workflow, every autonomy toggle off and nothing queued — and never when the key's `propagation-epic:` marker already matches an open epic, in which case `devwatch regroup-onto-existing-epic` routes into that one and creates nothing at all (#3799); and (2) §6c's threshold-escalation path creates **at most one umbrella issue** per coarse key, and **only** for an **eligible** key whose running site count crosses the threshold (>5 sites, or recurs across >1 scan run) — a `new_helper` / `new_pattern` / `perf_fix` key, or a `bugfix_shape` key that has **graduated** by having its rule committed under `.claude/rules/` with the key's `propagation-key:` marker and its per-site treatment written out (#155); never for an ungraduated `bugfix_shape` key (those stay individual per-site issues, for individual review). **§6c is the single place in this skill that mints an umbrella from an eligible key**, and it is the scoped relaxation of the pre-#2517 absolute "never creates an umbrella" boundary — owned in §6c, exactly the eligible-key-at-threshold case, nothing wider. §9.5's epic is the orthogonal routing destination: it re-homes findings the scanned diff did not cause, it is not a mechanical-sweep accumulator, and it lands unstarted — so the eligibility gate on (2) does not reach it, and a `bugfix_shape` key mints its routing epic like any other whether or not it has graduated. Outside those two, the skill files **flat** `child-of` per-site children of the scan target and creates nothing. The §6b accumulator still only **appends to an umbrella that already exists**; it does not create one.
- **Closes its own duplicates, appends to / creates an umbrella for a mechanical sweep, edits nothing else (#2354, #2515, #2517).** The skill may make exactly these mutations to a pre-existing issue: (1) **closing an exact fine-key duplicate** in §6a — a redundant `propagation:` child of the *current* scan target, collapsed onto its lowest-numbered canonical and reconciling its now-orphaned devwatch step in one operation via `devwatch close-folded-issue` (close as not-planned + #2516's fold, #3616); (2) **appending a propagation site** in §6b — an unchecked checklist comment to an **open umbrella** whose body already carries the candidate's `propagation-umbrella: <class>:<helper-or-signal>` marker; and (3) **folding per-site issues into a freshly-created umbrella** in §6c — closing each open per-site `propagation:` issue for a threshold-crossing eligible coarse key onto the umbrella §6c just created (generalising §6a's close-duplicate from same-fine-key to same-coarse-key) and reconciling its devwatch step in one operation via `devwatch close-folded-issue` (close as not-planned + #2516's fold, #3574). It never re-titles, re-labels, reopens, or rewrites the body of any issue, never closes the canonical of a fine-key group, and never touches an issue that is neither a propagation child under this scan target nor an umbrella (existing or just-created) for a candidate's coarse key. The one re-parent it may cause is §9.5's `pre-existing` route, and only over issues **this run just filed** — and it never writes that edge by hand: the `regroup-onto-*-epic` commands own the membership row and the `child-of` graph together, so they move as one. Routing into an epic that already exists (#3799) adds members to that epic's workflow; it does not edit the epic issue itself — no re-title, no re-label, no body rewrite. Apart from those moves, each run owns only the issues it creates.
