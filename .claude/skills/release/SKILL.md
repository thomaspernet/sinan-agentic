---
name: release
description: Cut the release the promoted work ships, against the published history read live.
family: delivery
capability: releasable
---
Cut the release the promoted work ships.

## 1. Read what is already published

`gh release list` — read it live rather than from anything cached. A stale
history mints a version the repository already carries, which is a write that
cannot be taken back.

## 2. Pick the version

Derive the next version from the published history and the nature of the work
that landed. A version that already exists is a halt, not an overwrite.

## 3. Cut it

Create the release against the promoted branch, with notes naming what changed
for someone who did not follow the work. Confirm it published by reading it
back.

Do not cut a release for a repository whose writes have not flipped to this
app — that refusal belongs to the service, and a step that works around it is
writing on behalf of a system that has not stood down.

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
