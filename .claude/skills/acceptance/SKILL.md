---
name: acceptance
description: Run the acceptance scenarios against the branch the run is about to merge — on its pull request, or on the check stage that stands in for one — and return a pass/fail verdict.
family: review
capability: acceptance
shipped-from: 3856b1fda22a35fe3fb6f1ad87236e51ca6dd3dae219c6bc789854dbdd69a0cd
---
Run the acceptance scenarios against the branch the run is about to merge, and
return a verdict.

You are the gate on the stage that holds that branch ahead of its merge — the
pull request where the repository opens one, the check stage where it opens
none — and a generated chain carries that stage twice, the epic's and the
standalone run's, with the run dispatching the one its own shape answers for.
Read the branch off the stage this run took rather than assuming which of the
two it was: a standalone run has no epic branch to look for. You verify the
branch, you do not amend it.

## 1. Run the scenarios

Run the repository's acceptance suite against that branch. Run the
whole suite — a subset chosen because the rest looked unrelated is the same
claim as a green run without the evidence.

## 2. Read a failure before reporting it

A failing scenario is either a real regression or a scenario that has gone
stale against intended behaviour. Say which, in the verdict — a reviewer
deciding whether to merge needs that distinction, and only the run that saw the
failure can make it.

## 3. Return the verdict

This gate verifies a stage the run performs once for itself, so there is no
document to record a verdict against: the settlement below *is* the verdict.
Settle `passed` on a clean suite and `failed` on a failing one, naming
the failing scenarios — a failure is a decision for a person, not a retry.
Never report a suite that could not run as a pass: that is `failed`, with the
reason it could not run.

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
