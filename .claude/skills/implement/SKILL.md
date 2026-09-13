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
prior verdict. A re-run is guided by that feedback rather than by the original
acceptance criteria alone: treat each finding as an instance of a class and
clear the class, and build on any `chore(gate):` commits already on the branch
rather than reverting or re-applying them.

## 2. Find the reference before writing anything

Classify the work — new entity, endpoint, component, migration — then find the
closest existing implementation of that kind and read it. The patterns you
follow come from the codebase, not from memory.

## 3. Branch, implement, test

Cut the branch off the base the run is working on and implement against the
issue's acceptance criteria. A bug is fixed against a test that fails before
the fix and passes after it, written first — what it proves is then the bug
rather than the fix. Run the suite before you commit. Commit with a
conventional-commit subject naming the issue it closes.

## 4. Review your own diff before settling

The gate grades this diff against the checklists this launch already primed, so
most of what it would find is reachable from here. A unit settled with a known
failing item is a round trip nobody needed.

- **Walk the checklists against the diff**, and fix what fails now.
- **Ask where else the change belongs.** Every change has a family: a sentence
  restating the old behaviour, a second copy of a literal this diff introduces,
  another instance of a shape fixed once. Search with ignore rules off —
  ignore-respecting search skips the very surfaces a stale reference survives
  in — and clear every member.
- **Walk the fan-out.** List an edited function's branches and its callers, and
  confirm the change reached each symmetric sibling: the other scope, the other
  opener, the matching config path. A fix applied to one of two twin paths is
  the most common behaviour failure, and no text search surfaces it.
- **Your own additions are review surface.** Docstrings, tests and helpers
  written this round are graded on the same terms as the code.

## 5. Record what you noticed

`report_outcome` carries `notes`, each with a `category`: a `risk` a reviewer
should watch, a `consideration` weighed and deliberately not done, a
`follow_up` this unit could not do. The gate reads them to focus its review, so
the family sweep above belongs there as a consideration — naming what was
searched and cleared lets the reviewer verify it rather than repeat it.

## 6. Nothing to implement

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
