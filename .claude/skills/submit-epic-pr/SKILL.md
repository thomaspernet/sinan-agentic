---
name: submit-epic-pr
description: Open the epic's pull request into the development branch, once every child has landed and no proposal is already open for the branch.
family: delivery
---
Open the epic's pull request into the development branch.

The epic, the integration branch and the development branch are the ones the
launch names in *The run you were launched for*. A checkout can hold other
epic branches and other proposals; none of them is this run's, and a proposal
found for one of them does not make this step done.

## 1. Confirm every child has landed

The launch lists the run's members with how the run settled each. A member
settled `skipped` is neither waited for nor named as a blocker, whatever its
issue or its branch reads: its line says which of the three it was — work
already delivered, work left out of this epic, or work that turned out not to
be needed — and none of the three is a child still to land. Every other member
is a child that must have landed. One the run has not settled, one
settled `failed`, or one whose branch has not merged into the integration
branch means the epic is not ready to propose — settle `failed` naming the
child. An integration branch that is not on origin, or an epic that is not this
repository's, is a halt naming what was expected, never a proposal for what the
checkout holds.

## 2. Check for a proposal already open

`gh pr list --head <integration-branch> --state open`, with the named branch.
One open pull request per branch: a second proposal for one branch splits
review across two threads. If one is open for the named branch, this step is
already done.

## 3. Open it

Open the pull request from the named integration branch into the named
development branch.
The body summarises what the epic changed and how a reviewer convinces
themselves it works — the children's own titles, not a restatement of every
commit. A member whose work was already delivered is listed with the
references its line names, so a reviewer knows where to read it. A member the
run left out, and one whose work turned out not to be needed, is listed under
what the epic does not ship, with the reason its line gives, so a reviewer
reads its absence as a decision rather than an oversight.

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
