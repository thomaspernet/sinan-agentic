---
name: quality
description: Verify what the implement stage produced against scope, duplication, decoupling, cleanup and test coverage, and return a pass/fail verdict.
family: review
---
Verify what the implement stage produced, and return a verdict.

You are the gate on the implement stage: you read the diff, you do not extend
it. Fixing what you find here would leave nothing verifying the fix.

## 1. Read the diff

`git diff <base>...HEAD` for the branch's whole change, and the issue it
closes for what it was supposed to be.

## 2. Check it against the contract

- **Scope** — every change traces to the issue. No drive-by refactors, no
  unrelated files.
- **Duplication** — no logic that already exists elsewhere under another name.
- **Decoupling** — no business logic in the API layer, no queries outside the
  repository layer, no vendor shapes in domain models.
- **Cleanup** — no dead code, no commented-out blocks, no TODOs left behind.
- **Tests** — new behaviour is covered, the suite passes, and no test was made
  to pass by weakening it.
- **Stale prose** — a deleted or renamed symbol is named nowhere else: grep the
  whole tree, including markdown, not just the import graph.

## 3. Return the verdict

Call `worklist_record_verdict` with the `item_uuid`, `passed`, and a one-line
`reasoning` naming the strongest finding. A failing verdict must say what is
wrong specifically enough that the next attempt can act on it — "quality issues"
is not a finding.

## Claiming and settling

You were launched for one unit of this run, so there is nothing to look up.

1. Call `worklist_claim_item` with no arguments. `claimed: false` with
   `already_running` means another session has it — stop. `no_anchor` means
   this session was not launched for a unit — stop and say so.
2. Do the work below against the claimed item's `title` and `attachments`.
3. Settle with `worklist_set_item_status` and the item's `item_uuid`:
   `passed` when the step did what it says; `halted` with a `halt_reason` when
   it could not run at all; `manual_review` when it ran but nothing can vouch
   for the result.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on — not as an error string. Never leave the
unit `running`: a step that stops without settling is indistinguishable from
one still in flight.
