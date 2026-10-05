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

## 2. Cut it through the run

Call `worklist_cut_release` with `tier: staging` and the bump the work calls
for — `staging` is what the stage you were launched for cuts for, and any
other tier is refused naming both (#2642). The app derives the staging
version from the published history read live, cuts the pre-release against
the staging branch's current commit, and records the tag on this stage in the
same call — the record the stage's truth is read from, by probing the tag on
origin. Do not cut with `git` yourself: a tag cut that way records nothing —
the app's own Ship view is the one exception, since a cut there for the tier
this stage stands at records on it too (#2641). `changed: false` means the
tag was already published — a re-run after a cut that landed, which repeats
that tag rather than cutting a second one beside it (#2643) — and the record
names it: a settled success, not a halt.

## 3. Confirm it

Read the tag back and confirm it resolves to the commit you intended — a tag
on the wrong commit names the wrong artifact, and nothing downstream can tell.

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
