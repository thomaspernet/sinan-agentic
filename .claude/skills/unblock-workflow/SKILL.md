---
description: "Recover blocked workflow work — a halted run or a red ship PR: read the read-only unblock plan, then drive an in-scope fix, file a follow-up issue, or escalate — never a shortcut."
capability: core
---

Recover blocked workflow work. The dispatcher launches this skill on a hands-off workflow when either of two things blocks it (#3081, #3818):

- **A halted run** — a child issue's run stopped with a `halt_reason`. ISSUE is that child.
- **A red ship PR** — the workflow's `submit-workflow-pr` chip is `FAILED` because CI went red on the ship PR. Every member already reached `done`, so no run is halted. ISSUE is the workflow **root**.

Read the block context and the server's read-only recovery plan, reason about *why* the work is blocked, and drive the right recovery: an in-scope fix (a `devwatch unblock` state lever **or** a code edit), a new issue when the cause is a separate problem, or an escalation to a human when the only lever left is a shortcut. The coding philosophy governs the call — there is no hardcoded allow-list.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill unblock-workflow --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done. Two of them decide this skill: clean-code Pillar 2 (**No Shortcuts**) and the Completion Checklist **Scope** section — together they are the whole basis for the recover-vs-file-vs-escalate call below.

Read this repo's CLAUDE.md for architecture and rules.

## Parse arguments

Extract the blocked issue number and optional run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `"42"` → ISSUE=42, RUN_ID=(none)
- `$ARGUMENTS` = `"42 --run 7"` → ISSUE=42, RUN_ID=7

ISSUE is the **blocked issue** — a halted child, or the workflow root when the ship PR is red. RUN_ID is **this unblock agent's own run** — use it only for `devwatch agent-update`, never to identify the blocked run. Which of the two blocks you are looking at, the halted run, its feature branch, its owning workflow, and the frozen `halt_reason` are all **read from the server** below — never taken from an argument or guessed.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command.

## Single-step boundary

You own **exactly one** thing — recovering this block — and nothing else. Diagnose, drive a single recovery, record the outcome with `devwatch agent-update`, and stop. You do **not** run the workflow's later steps (quality, propagation, merge, documentation, release); once the run is unblocked the dispatcher's `on_issue_run_completed` hook chains every one of them, exactly as it would for any other run, and once the ship PR is genuinely green the poller's own green tick flips `submit-workflow-pr` to `done` and the ship cascade continues. Do not trigger another action or try to drive the pipeline from inside this run. If your *own* run wedges, `devwatch unblock <ISSUE> resume-run` is the kick (see `devwatch unblock-plan <ISSUE>` for the recovery options) — do not improvise the next step.

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

3. The owning workflow — its `base_branch_resolved` (the integration branch), the ship-chip statuses, and the halted step's feature branch, all needed for an in-scope code fix:

   ```bash
   devwatch --repo "$REPO" workflow-get --issue <ISSUE>
   ```

Treat all three as authoritative grounding. Do not act until you understand *why* the work is blocked.

**Which block is this?** The reads above answer it: a halted child run carries a `halt_reason` and ISSUE is that child; a red ship PR shows `submit-workflow-pr` at `FAILED` with every member step `done`, and ISSUE is the workflow root. If it is the red ship PR, add one more read before deciding — the failing CI job itself, which is the actual root cause and is not in any of the reads above:

```bash
gh pr checks <SHIP_PR> --repo "$REPO"               # which job(s) are red
gh run view <CI_RUN_ID> --repo "$REPO" --log-failed # why
```

Resolve `<SHIP_PR>` from the workflow's `pr_number`, and `<CI_RUN_ID>` from the failing check's run URL in the `gh pr checks` output. `<CI_RUN_ID>` is a **GitHub Actions** run id — it is not `RUN_ID`, your own devwatch agent run from Parse arguments; passing that one here reads an unrelated run or errors, and you get no log. Read the failing job before forming any opinion about the failure — a diagnosis you did not read the log for is a guess.

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

- **A code fix on a red ship PR.** Same call, one branch over: the failure is on the *integration* branch (`base_branch_resolved` from `workflow-get`), not a member's feature branch, and there is no run to resume. Fix the cause the failing job showed you, push to the integration branch, and stop — CI re-runs on the push, and when it goes green the poller flips `submit-workflow-pr` to `done` on its own and the ship cascade continues. You do not fire a chip to make that happen.

  ```bash
  git checkout <integration-branch>   # base_branch_resolved from workflow-get
  git commit -m "fix(scope): <what the failing job showed> (#<ISSUE>)"
  git push
  ```

  **Never re-run CI to see if it passes this time, and never re-fire `submit-workflow-pr`.** A re-run without a diagnosis is the retry Pillar 2 forbids, and the PR already exists — re-firing the submit chip tries to re-open it. If you read the log and concluded the failure is a flake, that is not a reason to re-run: a flaky test is a race condition to find. Fix the race if it is in scope, or file an issue against the flaky test (outcome 2 below) — the issue is the deliverable, not a green tick.

### 2. A separate problem — file a new issue, leave the run for a human

If the root cause is **outside** this issue's scope — a pre-existing bug, an unrelated infra failure, a flaky test this feature did not introduce, a dependency problem — do **not** patch unrelated code. "If you found something else to fix, create a separate issue." File it via `/new-bug` (or `devwatch create-issue --type bug …`), describing what blocked the work, the `halt_reason` or the failing CI job, and why it is out of scope for this issue. Then leave the blocked work exactly as it is — a human decides when the new issue is addressed.

### 3. No shortcut-shaped recovery — escalate

The three **destructive** primitives — `mark-pr-green` (force a red PR green), `mark-issue-done` (close an incomplete issue), and `restart-run` (discard state and start over) — are the shortcut shapes Pillar 2 forbids ("if a test is flaky, find the race condition — do not add a retry"). **Never self-authorize them.** `mark-pr-green` is the one to watch on a red ship PR: it is exactly the lever that looks like it solves the problem, and using it launders a genuine regression into a green ship. If the only way forward is one of the three, the block needs a human: leave it as it is and record an escalation summary (below). Filing a new issue (outcome 2) is the in-bounds escalation whenever the blocker is a describable separate problem; a bare escalation is for when it is not.

The dispatcher bounds how many times it will launch you for the same block. When that budget is spent it escalates the workflow to a real halt for a human, so an escalation here is a terminal, not a dead end.

## Record completion

Record the outcome (omit `--run-id` if RUN_ID is unavailable). Always `--status completed` — this agent finished its recovery reasoning; the blocked run's (or ship PR's) own state is separate and advances on its own once unblocked.

`agent-update` records against **your own** run only — `--run-id`, else the `DEVWATCH_AGENT_RUN_ID` your launcher put in this process's environment (#3761). It never resolves a run by issue: a recovery lever above may have just dispatched one, and that run is not yours to close. When neither source resolves — an uncorrelated manual invocation with no `--run 7` — the command refuses. That is the correct terminal, not something to route around: report the recovery in your reply and stop.

The summary is your own prose and names the files and problems you found, so pass it through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command:

```bash
SUMMARY=$(cat <<'SUMMARY_EOF'
<recovered via resume-run / fixed <file> and resumed / filed #<N> for out-of-scope <problem> / escalated: only <destructive primitive> would help>
SUMMARY_EOF
)

devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "$SUMMARY"
```

## Boundary

- **One recovery per run.** Drive a single recovery, record it, stop. The dispatcher chains everything after an unblocked run, and the poller settles a ship PR that goes green.
- **Read the plan before acting.** The `unblock-plan` read is not optional — no recovery without the diagnosis. On a red ship PR the failing CI log is part of that read.
- **Scope discipline is the whole contract.** In-scope → fix (lever or code). Out-of-scope → new issue. Shortcut-only → escalate. There is no fourth path and no hardcoded allow-list.
- **Never self-authorize a destructive primitive.** `mark-pr-green`, `mark-issue-done`, and `restart-run` are human decisions, always.
- **Never re-run CI on red, and never re-fire `submit-workflow-pr`.** Re-running without a diagnosis is a retry, not a recovery; the ship PR already exists, so re-firing its chip re-opens nothing. Fix the cause on the integration branch or file the issue.
