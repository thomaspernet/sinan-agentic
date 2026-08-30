---
name: close-epic
description: Close the epic issue once its work has merged and every child is closed.
family: delivery
---
Close the epic once its work has merged.

## 1. Confirm it merged

The epic's own branch is merged into the development branch, and every child
issue is closed. An open child means the epic is not done — settle `halted`
naming it.

## 2. Close

`gh issue close <N>` with a comment saying what shipped, in one or two
sentences a reader outside this run can follow. An already-closed epic is
success: this step is idempotent, so a re-run does not stop on what a prior
pass finished.

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
