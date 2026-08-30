---
name: acceptance
description: Run the acceptance scenarios against the branch the run's pull request proposes, whichever of the two proposals it opened, and return a pass/fail verdict.
family: review
capability: acceptance
---
Run the acceptance scenarios against the branch the run's pull request
proposes, and return a verdict.

You are the gate on the pull-request stage, and a generated chain carries that
stage twice — the epic's proposal and the standalone run's — with the run
dispatching the one its own shape answers for. Read the branch off the proposal
this run opened rather than assuming which of the two it was: a standalone run
has no epic pull request to look for. You verify the proposal, you do not amend
it.

## 1. Run the scenarios

Run the repository's acceptance suite against the proposed branch. Run the
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
Settle `passed` on a clean suite and `manual_review` on a failing one, naming
the failing scenarios — a failure is a decision for a person, not a retry.
Never report a suite that could not run as a pass: that is `halted`, with the
reason it could not run.

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
