---
name: submit-pr
description: Open the pull request a standalone run proposes, from the branch its work landed on into the development branch.
family: delivery
---
Open the pull request a standalone run proposes into the development branch.

This run delivers one issue rather than an epic, so there are no children to
confirm have landed — what the run's members wrote is the whole proposal. The
repository and the development branch are the ones the launch names in *The
run you were launched for*.

## 1. Find the branch the work landed on

The run's own branch: the one its member implemented on, or the integration
branch its member merged into where the run has one — the launch names it
where the run has one. Read it rather than assume it — a proposal opened from
the wrong branch proposes someone else's work.

## 2. Check for a proposal already open

`gh pr list --head <branch> --state open`. One open pull request per branch: a
second proposal for one branch splits review across two threads. If one is
open, this step is already done.

## 3. Open it

Open the pull request from that branch into the development branch, naming the
issue it closes. The body says what changed and how a reviewer convinces
themselves it works.

Write nothing a reader outside this run cannot understand: no run identifiers,
no phase names, no first-person agent voice.

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
