---
name: implement
description: Implement one issue on its own branch — read the issue and its lineage, follow the closest existing implementation, and test before committing.
family: delivery
---
Implement one issue on its own branch.

## 1. Read the issue and the work already done

Read the issue with `gh issue view <N> --json title,body,labels`. Read its
`child-of` links too: a sub-issue of an epic inherits decisions the parent
already fixed, and re-deriving them produces a second answer to a settled
question.

Check whether the issue has been attempted before — an existing branch, a
prior review comment. A re-run is guided by that feedback, not by the original
acceptance criteria alone.

## 2. Find the reference before writing anything

Classify the work — new entity, endpoint, component, migration — then find the
closest existing implementation of that kind and read it. The patterns you
follow come from the codebase, not from memory.

## 3. Branch, implement, test

Cut the branch off the base the run is working on, implement against the
issue's acceptance criteria, and write tests for every new behaviour. Run the
suite before you commit. Commit with a conventional-commit subject naming the
issue it closes.

## 4. Nothing to implement

If the work is already in the tree — shipped by a sibling, or the criteria are
already met — do not fake a commit and do not exit silently. Settle the unit
`manual_review` and name which commit or issue already covers it.

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
