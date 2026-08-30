---
name: tag-staging
description: Cut the staging tag the artifact is named by, against the staging branch's current commit.
family: delivery
capability: tag_staging
---
Cut the staging tag the artifact is named by.

Only a repository whose cascade has a staging tier *and* whose build produces
an artifact has anything to tag — that pairing is the capability this skill
declares, so a unit that reaches here has already been gated on it.

## 1. Read the existing tags

Read the tags already published. A tag is immutable by convention: re-cutting
one renames an artifact someone may already have downloaded.

## 2. Cut it

Cut the tag against the staging branch's current commit and push it. Confirm it
resolves to the commit you intended by reading it back — a tag on the wrong
commit names the wrong artifact, and nothing downstream can tell.

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
