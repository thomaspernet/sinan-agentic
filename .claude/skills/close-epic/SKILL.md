---
name: close-epic
description: Close the epic and the member issues it delivered, once its work has merged.
family: delivery
shipped-from: e485226fca74fd65d2b4090132654dd818d5f3144d670cfce7e464bafb03b0c4
---
Close the epic and the members it delivered.

Both are named by the launch in *The run you were launched for* — the epic,
whose number is the `<N>` below, and every member under it.

## 1. Confirm it merged

The named epic's own branch is merged into the development branch. A branch
that has not landed is a halt rather than a close — settle `failed` saying so.

## 2. Close the members

Closing them is this step's work, not GitHub's. The run merges into the
development branch, which is not the repository's default branch, so a
`closes #NNNN` trailer on a merged commit never fires there, and no other step
of the chain closes a member: left to itself, work that shipped keeps reading
as outstanding.

`gh issue close <M> --comment "..."` each member the launch names that shipped,
the comment saying what that member shipped and naming the proposal that
carried it. A member the launch names in another repository is closed there,
with `--repo <owner/name>`. A member already closed is success: this step is
idempotent, so a re-run does not stop on what a prior pass finished.

A member the launch lists as settled `skipped` was deliberately left out of the
epic, and its work did not ship: leave its issue open, and comment on it with
`gh issue comment <M> --body "..."` naming the epic it was deferred from and
the reason its settle note gives, so it reads as work still to do rather than
work the epic forgot. One already carrying that comment is success. Only a skip
is read that way: a member settled `failed` is neither closed nor passed over,
and is a halt naming it.

## 3. Close the epic

`gh issue close <N>` with a comment saying what shipped, in one or two
sentences a reader outside this run can follow. An already-closed epic is
success on the same terms.

A member that shipped and is still open is a failed close even when the epic
closed — settle `failed` naming the members that are open, never `passed`. A
member left out and left open is not one of them.

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
