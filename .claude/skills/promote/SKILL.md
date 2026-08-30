---
name: promote
description: Promote the merged work onto the next branch of the cascade, from a commit whose checks are passing.
family: delivery
---
Promote the merged work onto the next branch of the cascade.

## 1. Confirm the source is green

The branch being promoted has its checks passing at the commit being promoted —
not at some earlier commit that happened to be green.

## 2. Promote

Merge the source branch into the target tier's branch and push. Promotion moves
what is already there; it never rewrites history and never force-pushes.

A conflict at a promotion boundary means the tiers have diverged, which is a
decision about intent — settle `halted` with what diverged, rather than
resolving it here.

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
