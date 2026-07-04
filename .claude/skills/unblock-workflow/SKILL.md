---
description: "Recover a halted workflow run: read the read-only unblock plan, then drive an in-scope fix, file a follow-up issue, or escalate — never a shortcut."
capability: core
---

Recover a halted workflow run. The dispatcher launches this skill when a run halts on a hands-off workflow (#3081). Read the halt context and the server's read-only recovery plan, reason about *why* the run is blocked, and drive the right recovery: an in-scope fix (a `devwatch unblock` state lever **or** a code edit within the current feature), a new issue when the cause is a separate problem, or an escalation to a human when the only lever left is a shortcut. The coding philosophy governs the call — there is no hardcoded allow-list.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill unblock-workflow --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done. Two of them decide this skill: clean-code Pillar 2 (**No Shortcuts**) and the Completion Checklist **Scope** section — together they are the whole basis for the recover-vs-file-vs-escalate call below.

Read this repo's CLAUDE.md for architecture and rules.

## Parse arguments

Extract the halted issue number and optional run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `"42"` → ISSUE=42, RUN_ID=(none)
- `$ARGUMENTS` = `"42 --run 7"` → ISSUE=42, RUN_ID=7

ISSUE is the **halted issue**. RUN_ID is **this unblock agent's own run** — use it only for `devwatch agent-update`, never to identify the halted run. The halted run, its feature branch, its owning workflow, and the frozen `halt_reason` are **read from the server** below — never taken from an argument or guessed.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command.

## Single-step boundary

You own **exactly one** thing — recovering this halt — and nothing else. Diagnose, drive a single recovery, record the outcome with `devwatch agent-update`, and stop. You do **not** run the halted workflow's later steps (quality, propagation, merge, documentation, release); once the run is unblocked the dispatcher's `on_issue_run_completed` hook chains every one of them, exactly as it would for any other run. Do not trigger another action or try to drive the pipeline from inside this run. If your *own* run wedges, `devwatch unblock <ISSUE> resume-run` is the kick (see `devwatch unblock-plan <ISSUE>` for the recovery options) — do not improvise the next step.

## Reviewer context — read the recovery plan (read-only)

Before deciding anything, read the server's diagnosis. This is read-only context: it posts nothing and mutates nothing.

1. The recovery plan — the diagnosis prose plus every recovery primitive, each marked **runnable** or **disabled** with the server's reason:

   ```bash
   devwatch --repo "$REPO" unblock-plan <ISSUE>
   ```

   The diagnosis says what the dispatcher sees and why the run is stuck. A **runnable** primitive is one the server will accept right now; a **disabled** one carries the reason it cannot fire — and the server rejects a disabled primitive with HTTP 409, so the runnable/disabled gate is authoritative, not advisory. The picker primitives (`run-action`, `run-workflow-pr-action`) also list their `--target-action` options.

2. The halt timeline — the frozen `halt_reason`, prior run attempts, and any quality-failure reports:

   ```bash
   devwatch --repo "$REPO" issue-history <ISSUE> --comments
   ```

3. The owning workflow — the base branch and the halted step's feature branch, needed for an in-scope code fix:

   ```bash
   devwatch --repo "$REPO" workflow-get --issue <ISSUE>
   ```

Treat all three as authoritative grounding. Do not act until you understand *why* the run halted.

## Diagnose and recover (the decision)

Reason about the root cause, then apply the existing scope discipline — clean-code Pillar 2 and the Completion Checklist **Scope** section, both loaded above. Exactly one of three outcomes.

### 1. In-scope recovery — a state lever or a code fix within the current feature

If the halt's root cause is within this issue/feature's scope, recover it yourself — the same latitude the implement and quality agents already have.

- **A state lever.** When the diagnosis points at a stuck run or action and a non-destructive primitive is runnable, fire the one it calls for:

  ```bash
  devwatch --repo "$REPO" unblock <ISSUE> resume-run
  ```

  The non-destructive levers are `resume-run` (re-dispatch a halted or wedged run), `skip-action` (retire one wedged action and keep the step open), `run-action` (jump the run to a chosen action — needs `--target-action <name>`), and `run-workflow-pr-action` (advance a workflow-PR chip stranded by a missed completion hook — needs `--target-action <name>`). The server's runnable/disabled gate is the guardrail; a disabled lever is not yours to force.

- **A code fix.** When the cause is code this feature left broken (a failing test, a quality violation, a bug in this feature's own diff), fix it on the halted step's feature branch (from `workflow-get`), run the tests, commit with a conventional message referencing the issue, then resume so the dispatcher re-runs the step from the fixed tip:

  ```bash
  git commit -m "fix(scope): <what you fixed> (#<ISSUE>)"
  git push
  devwatch --repo "$REPO" unblock <ISSUE> resume-run
  ```

### 2. A separate problem — file a new issue, leave the run for a human

If the root cause is **outside** this issue's scope — a pre-existing bug, an unrelated infra failure, a dependency problem this feature did not introduce — do **not** patch unrelated code. "If you found something else to fix, create a separate issue." File it via `/new-bug` (or `devwatch create-issue --type bug …`), describing what halted the run, the `halt_reason`, and why it is out of scope for this issue. Then leave the halted run exactly as it is — a human decides when the new issue is addressed.

### 3. No shortcut-shaped recovery — escalate

The three **destructive** primitives — `mark-pr-green` (force a red PR green), `mark-issue-done` (close an incomplete issue), and `restart-run` (discard state and start over) — are the shortcut shapes Pillar 2 forbids ("if a test is flaky, find the race condition — do not add a retry"). **Never self-authorize them.** If the only way forward is one of the three, the halt needs a human: leave the run halted and record an escalation summary (below). Filing a new issue (outcome 2) is the in-bounds escalation whenever the blocker is a describable separate problem; a bare escalation is for when it is not.

## Record completion

Record the outcome (use `--run-id` if available, else `--issue`). Always `--status completed` — this agent finished its recovery reasoning; the halted run's own state is separate and advances on its own once unblocked:

```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "<recovered via resume-run / fixed <file> and resumed / filed #<N> for out-of-scope <problem> / escalated: only <destructive primitive> would help>"
```

## Boundary

- **One recovery per run.** Drive a single recovery, record it, stop. The dispatcher chains everything after an unblocked run.
- **Read the plan before acting.** The `unblock-plan` read is not optional — no recovery without the diagnosis.
- **Scope discipline is the whole contract.** In-scope → fix (lever or code). Out-of-scope → new issue. Shortcut-only → escalate. There is no fourth path and no hardcoded allow-list.
- **Never self-authorize a destructive primitive.** `mark-pr-green`, `mark-issue-done`, and `restart-run` are human decisions, always.
