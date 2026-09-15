---
name: merge-epic
description: Merge the epic — through its pull request once the checks and the gate have passed, or its integration branch directly when it has none.
family: delivery
shipped-from: dadbf9c83b81df3227d18d59139dd23100c1e4abe06defb53665abc1a2ad7132
---
Merge the epic — its pull request when it has one, its integration branch when
it does not.

The epic and its integration branch are the ones the launch names in *The run
you were launched for*. A proposal for any other branch is not this run's,
however finished it looks; a run whose named branch has no proposal and no
merge to confirm is a halt naming that branch, never a pass on a sibling's.

## 1. Confirm the gate

A repository that opens a pull request merges through it, and only once its
checks and its acceptance gate have passed. `gh pr checks` — a pending or
failing check is a halt with the check named, never a merge with a note.

A repository that opens no pull request merges directly, with the same test
run this step would demand of any merge: the epic's integration branch into
the development branch. Nothing ahead of this stage has landed it — the
per-member merges put each child on the integration branch, and this step is
what puts the integration branch on the development branch.

A member the launch lists as settled `skipped` was deliberately left out of
the epic and is no reason to hold the merge: its absence from the integration
branch is the decision, not a gap. A member settled `failed` is one — settle
`failed` naming it rather than merging an epic missing work it meant to ship.

## 2. Merge

Merge through the pull request with a merge commit — `gh pr merge --merge`.
GitHub allows squash and rebase too, and both write a new commit while leaving
the branch they merged exactly where it stood, so the cleanup stage that
follows, which proves its own work by that branch's ancestry, can no longer
see the merge. This stage reads a merged proposal by the commit its merge
wrote and so passes on any of the three — a squash already made is not a halt
— but the method is named here so the rest of the chain has one proof to read.

Merge, and confirm the merge landed by reading the target branch back rather
than by trusting the command's own exit. Do not delete the branch here — the
cleanup stage owns that, and deleting it early takes the diff a later stage
still has to read.

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
