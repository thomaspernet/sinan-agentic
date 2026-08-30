---
name: merge-epic
description: Merge the epic — through its pull request once the checks and the gate have passed, or its integration branch directly when it has none.
family: delivery
---
Merge the epic — its pull request when it has one, its integration branch when
it does not.

## 1. Confirm the gate

A repository that opens a pull request merges through it, and only once its
checks and its acceptance gate have passed. `gh pr checks` — a pending or
failing check is a halt with the check named, never a merge with a note.

A repository that opens no pull request merges its integration branch into the
development branch directly, with the same test run this step would demand of
any merge.

## 2. Merge

Merge, and confirm the merge landed by reading the target branch back rather
than by trusting the command's own exit. Do not delete the branch here — the
cleanup stage owns that, and deleting it early takes the diff a later stage
still has to read.

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
