---
name: delete-feature-branch
description: Delete the branch a standalone run delivered once its merge into the development branch is proven and nothing downstream needs its diff.
family: delivery
---
Delete the branch this run delivered.

This run has one member and no integration branch: the branch its member wrote
is the branch the whole run delivered, and it is the one the run's pull request
proposed. The launch names it in *The run you were launched for*, and the
development branch likewise. You name no branch anywhere below — the record
does.

## 1. Prove it merged on origin

The branch this run delivered is merged into the development branch. Prove it
from origin, never from the local checkout — the drop deletes the remote ref
and checks no ancestry itself: `git fetch origin`, then
`git merge-base --is-ancestor origin/<run-branch> origin/<development-branch>`.
A branch whose merge you cannot establish is a halt, not a deletion.

## 2. Confirm nothing still needs its diff

A repository that opens a pull request carries the diff on the merge commit, so
the branch can go. A repository that opens none has only the branch to diff
against, and its documentation stage runs *before* this one.

The chain to read is this run's, not the canvas's. The launch's *The run you
were launched for* block lists the stages this run takes on, in order, each
with how its record stands, and names separately the stages of the canvas this
run left out; `delivery_run_chain` answers the same for a re-read mid-work. A
canvas read lists every block whether or not this run selected it, so a stage
the run left out reads there as one still ahead — and a stage that will never
run is not a stage to wait for.

A documentation stage this run takes on, standing ahead of this one and not yet
run, makes this step early — settle `failed` and say so rather than removing
the only thing left to read. A run that left its documentation stage out, and a
chain that carries none, leave nothing waiting: prove the merge and drop it.

## 3. Drop it through the run

Call `worklist_drop_branch` with no arguments. The branch is the one this run's
member recorded when the run cut it for the implement step, so you name none.
The app reads its tip commit, records it on that member, and only then deletes
the remote ref — that recorded tip is what this stage's truth proves the merge
by once the name is gone. Never delete the branch with `git` or `gh` yourself:
a branch deleted outside the run records no tip, the merge compare by name
finds no ref, and the stage halts over work that landed.

`changed: false` means the branch was already gone — a re-run after a drop
that landed — and is success, not a failure: this stage is idempotent by
intent. A refusal says why — a stage that is not this run's cleanup stage, a
run whose members name no single branch, a repository whose writes have not
flipped to this app, or a tip that could not be read — and each is a halt with
that reason, not a reason to delete another way.

## Settling

You were launched for one stage of this run as a whole, not for one document,
so there is nothing to claim. What the run is working — its repository, its
epic and its integration branch — is stated in the launch's own *The run you
were launched for* block; read them there, never from the checkout, which can
hold several epic branches and proposals that are not this run's. Every child
of the run has already settled by the time this stage starts; the work below
acts on what they landed.

Settle with `worklist_set_stage_status`, which takes no uuid — the run and the
stage rode in with the launch: `passed` when the stage did what it says;
`failed` with a `halt_reason` when it could not — the reason in words a
person can act on.

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
