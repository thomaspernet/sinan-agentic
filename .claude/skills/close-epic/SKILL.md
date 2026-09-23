---
name: close-epic
description: Close the issues this run delivered, each by what the run recorded of it, once its work has merged.
family: delivery
shipped-from: 93edf679bb9365433a04562e199b928065715c61c471bcca4ff48ad9ba768295
---
Close the issues this run delivered, each by what the run recorded of it.

They are named by the launch in *The run you were launched for*: the epic,
whose number is the `<N>` below, and every member under it. A run that
delivers one issue on its own has no epic above the work — that issue is its
one member, and the members below are the whole of this stage's work.

## 1. Confirm it merged

The branch this run delivered is merged into the development branch. A branch
that has not landed is a halt rather than a close — settle `failed` saying so.

A run that put nothing on its branch takes on neither its proposal nor its
merge — the launch lists them as left out, *the branch carries no work*.
There is nothing here to confirm: whatever its members shipped, they shipped
elsewhere or there was nothing to ship, and the close goes on to the issues.

## 2. Close each issue by what its line says

Closing them is this step's work, not GitHub's. The run merges into the
development branch, which is not the repository's default branch, so a
`closes #NNNN` trailer on a merged commit never fires there, and no other step
of the chain closes a member: left to itself, work that shipped keeps reading
as outstanding.

Every member's line ends in what its issue is owed, read off the outcome that
member settled with. Do what the line says and nothing else — this stage's
verdict is taken over that same reading, so an issue closed against your own
judgement fails the close that performed it.

- **close it as completed** — `gh issue close <M> --comment "..."`. The
  comment says what shipped and where: the proposal that carried it for work
  this run landed, and the commits, pull requests or issues the member's line
  names for work that was already in a tree.
- **close it as not planned** —
  `gh issue close <M> --reason "not planned" --comment "..."`, the comment
  carrying the reason the member's line gives. The question turned out not to
  exist, and an issue left open for it is offered back to the backlog forever.
- **leave it open** — `gh issue comment <M> --body "..."` naming this run and
  the reason the member's line gives, so it reads as work still to do rather
  than work the run forgot.

A member the launch names in another repository is acted on there, with
`--repo <owner/name>`.

This step is idempotent, so a re-run finishes what a prior pass left and
changes nothing else. Read each issue before you act on it —
`gh issue view <M> --json state,stateReason,comments` — and leave alone the
one already standing where its line says it belongs and already carrying this
run's comment. Neither is a halt: the close writes no second comment and
reopens nothing.

Only a skip's outcome is read this way. A member settled `failed` is neither
closed nor passed over, and is a halt naming it.

## 3. Close the epic

An epic-rooted run closes its epic last: `gh issue close <N>` with a comment
saying what shipped, in one or two sentences a reader outside this run can
follow. An already-closed epic is success on the same terms. A run that
delivers one issue on its own has no epic to close — step 2 closed its issue.

A member that shipped and is still open is a failed close even when the epic
closed — settle `failed` naming the members that are open, never `passed`. One
whose line said to leave it open is not one of them.

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
