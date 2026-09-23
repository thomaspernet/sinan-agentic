---
name: delete-branch
description: Delete one child branch once its merge into the integration branch can be proven.
family: delivery
shipped-from: d103de5087e3ef3a4e33f1466126476bbcb73f54fe5998dda6ede17a3371ca69
---
Delete one child branch that has landed.

## 1. Prove it merged on origin

The proof is read from origin, never from the local checkout: the drop deletes
the remote ref and checks no ancestry itself, and a local `--merged` list only
says what the local target holds, which lags origin and may not hold the
branch at all. `git fetch origin`, then
`git merge-base --is-ancestor origin/<branch> origin/<target>` — the claimed
item's branch against the branch it landed on: the run's integration branch,
or the development branch for a run whose shape cuts none. A branch whose
merge you cannot prove is not deleted: settle `failed` naming the branch
rather than removing work nothing else holds.

## 2. Drop it through the run

Call `worklist_drop_branch` with no arguments. The branch is the one the run
recorded on the claimed item when it cut it for the implement step, so you name
none. The
app reads the branch's tip commit, records it on the item, and only then
deletes the remote ref — that recorded tip is what this step's truth proves
the merge by once the name is gone. Never delete the branch with `git` or
`gh` yourself: a branch deleted outside the run records no tip, the merge
compare by name finds no ref, and the step halts over work that landed.

`changed: false` means the branch was already gone — a re-run after a drop
that landed — and is success, not a failure: this step is idempotent by
intent, because a re-run after a partial pass must not stop on what the
first pass finished. A refusal says why — a step that is not this run's
delete-branch step, an item that recorded no branch, a repository whose
writes have not flipped to this app, or a tip that could not be read — and
each is a halt with that reason, not a reason to delete another way.

## Claiming and settling

You were launched for one unit of this run, so there is nothing to look up.

1. Call `worklist_claim_item` with no arguments. `claimed: false` with
   `already_running` means another session has it — stop. `no_anchor` means
   this session was not launched for a unit — stop and say so.
2. Do the work below against the claimed item's `title` and `attachments`.
3. Settle with `worklist_set_item_status` and the item's `item_uuid`:
   `passed` when the step did what it says; `failed` with a `halt_reason`
   when it did not — the reason in words a person can act on; `skipped` when
   the question no longer exists.

   A step that decided its work fails settles `failed` and says why,
   never with `passed` — the status write refuses a pass over a verdict that
   judged this attempt, the engine outvotes a pass the world contradicts,
   and a failing gate verdict routes the step back by policy whatever is
   written.

A `skipped` settle also says what became of this step's work, as an `outcome`
with that outcome's evidence. A skip naming none is refused, and so is one
whose outcome has nothing behind it: you are the only one who knows, and a bare
skip leaves every reader after you guessing which of the three it was.

- `already_delivered` — the work is already done, in this repository or
  another. Give `references`, one per place it landed: a commit as
  `owner/name@sha`, a pull request or issue as `owner/name#123`. It is the one
  outcome that says something shipped, and a run reads it to know this step
  delivered even though nothing landed on its own branch.
- `left_out` — the run decided not to do this work. Give `outcome_reason`, one
  line saying why.
- `not_needed` — the question turned out not to exist. Give `outcome_reason`,
  one line saying why.

Give the settle a `note`: a few sentences in your own words for a person
reading the issue later — what you found, what you chose, and what you left.
It is stored verbatim against this attempt, so write prose, not a status
string and not a commit message. It is optional — a settle with no note is
valid — and it is not the `halt_reason`: the reason says what stopped the
unit, the note says what the work was.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on — not as an error string. Never leave the
unit `running`: a step that stops without settling is indistinguishable from
one still in flight.
