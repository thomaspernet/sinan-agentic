---
name: merge-to-base
description: Merge one verified child branch into the epic's integration branch, testing the merge result before pushing it.
family: delivery
---
Merge one landed child branch into the epic's integration branch.

## 1. Confirm it is ready to land

The branch's own checks pass and its gate verdict passed. A branch that has not
been verified is not merged here — settle the unit `halted` and say so.

## 2. Merge

Merge the child branch into the integration branch. Resolve a conflict only
when the resolution is mechanical and both sides are yours; anything that needs
a decision about intent is a halt, not a guess.

Run the test suite on the merge result before you push. A merge that lands
green branches into a red integration branch is the failure this step exists to
catch.

## 3. Push

Push the integration branch. Do not open a pull request — the epic's own
proposal is a later stage, and one opened here would propose a partial epic.

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
