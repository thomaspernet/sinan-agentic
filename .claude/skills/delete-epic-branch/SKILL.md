---
name: delete-epic-branch
description: Delete the epic's integration branch once its merge is proven and nothing downstream still needs its diff.
family: delivery
shipped-from: bd7cae4416d1aed21cf1cf1efafa2dcb80559ac659bad20b8a2e2badf2695cf9
---
Delete the epic's integration branch.

The integration branch is the one the launch names in *The run you were
launched for*, and the development branch likewise.

## 1. Prove it merged on origin

The named integration branch is merged into the development branch. Prove it from
origin, never from the local checkout — the drop deletes the remote ref and
checks no ancestry itself: `git fetch origin`, then
`git merge-base --is-ancestor origin/<integration-branch> origin/<development-branch>`.
A branch whose merge you cannot establish is a halt, not a deletion.

## 2. Confirm nothing still needs its diff

Nothing this run still does reads the branch by name, so a proven merge is all
this step waits for. The documentation pass, the propagation scan and the rule
pass are the run's tracks: each starts beside the pull request in a copy of the
code of its own, pinned to the commits where every member landed, and may still
be running or already settled when this step opens. A pinned copy holds
commits rather than a branch, so dropping the branch takes nothing from a track
still running. The documentation pass reads the run's story off
the merge commit the merge stage wrote, which carries the epic's diff between
its two parents, and falls back to the tip this drop records once the name is
gone.

So do not hold this step back for a documentation stage, whether it is running,
settled or never taken on: a track that has not settled is no reason to settle
`failed` here. Prove the merge and drop it.

## 3. Drop it through the run

Call `worklist_drop_branch` with no arguments. The branch is the integration
branch the run recorded cutting at its start, so you name none. The app reads
its tip commit, records it on the run, and only then deletes the remote ref —
that recorded tip is what this stage's truth proves the merge by once the name
is gone. Never delete the branch with `git` or `gh` yourself: a branch deleted
outside the run records no tip, the merge compare by name finds no ref, and
the stage halts over work that landed.

`changed: false` means the branch was already gone — a re-run after a drop
that landed — and is success, not a failure: this stage is idempotent by
intent. A refusal says why — a stage that is not this run's cleanup stage, a
run that recorded no integration branch, a repository whose writes have not
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
