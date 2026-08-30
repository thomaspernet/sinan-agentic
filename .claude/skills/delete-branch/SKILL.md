---
name: delete-branch
description: Delete one child branch once its merge into the integration branch can be proven.
family: delivery
---
Delete one child branch that has landed.

## 1. Prove it merged

`git branch --merged <integration-branch>` — the branch must be in that list.
A branch whose merge you cannot prove is not deleted: settle `halted` naming
the branch rather than removing work nothing else holds.

## 2. Delete it

Delete the local and the remote branch. An already-absent branch is success,
not a failure — this step is idempotent by intent, because a re-run after a
partial pass must not stop on what the first pass finished.

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
