---
name: delete-epic-branch
description: Delete the epic's integration branch once its merge is proven and nothing downstream still needs its diff.
family: delivery
---
Delete the epic's integration branch.

## 1. Prove it merged

The branch is merged into the development branch. Prove it — a branch whose
merge you cannot establish is a halt, not a deletion.

## 2. Confirm nothing still needs its diff

A repository that opens a pull request carries the epic's diff on the merge
commit, so the branch can go. A repository that opens none has only the branch
to diff against, and its documentation stage runs *before* this one. If the
documentation stage has not run, this step is early — settle `halted` and say
so rather than removing the only thing left to read.

## 3. Delete it

Delete the local and the remote branch. An already-absent branch is success.

## Settling

You were launched for one stage of this run as a whole, not for one document,
so there is nothing to claim and nothing to look up. Every child of the run has
already settled by the time this stage starts; the work below acts on what they
landed.

Settle with `worklist_set_stage_status`, which takes no uuid — the run and the
stage rode in with the launch: `passed` when the stage did what it says;
`halted` with a `halt_reason` when it could not run at all; `manual_review`
when it ran but nothing can vouch for the result.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on — not as an error string. Never leave the
unit `running`: a step that stops without settling is indistinguishable from
one still in flight.
