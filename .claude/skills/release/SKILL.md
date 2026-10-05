---
name: release
description: Cut the release the promoted work ships, against the published history read live.
family: delivery
capability: releasable
---
Cut the release the promoted work ships.

## 1. Confirm the checks result

The checks ran once, before the work merged, and this stage reuses that result
rather than running them again.

Where this repository's checks run is the launch's *Checks* line: `none`,
`local_script` with the script it runs, or `gh_workflow` with the workflow
file. Read it there, and never from whether the repository opens a pull
request — a repository can open one that nothing verifies, and one that opens
none can still declare a script.

The launch also states *Checks result recorded at the merge* — the result that
verified the work this stage carries.

`passing` is the go. A *Checks* line reading `none` means the repository
declared nothing to verify, so there is no result to wait for. Anything else
while checks are declared — `pending`, `failing`, or no result recorded — is a
halt naming it. Do not run the check script, rerun a workflow, or read another
commit's checks to stand in for the recorded result.

## 2. Read what is already published

`gh release list` — read it live rather than from anything cached, and decide
from it and from the nature of the work that landed which part of the version
advances: `major`, `minor` or `patch`.

## 3. Cut it through the run

Call `worklist_cut_release` with the tier this stage releases to and the bump
you decided. The tier is not yours to choose: the stage you were launched for
cuts for exactly one, and any other is refused naming both (#2642). The app
derives the next version from the published history read live, cuts the
release against the tier's branch, and records the tag on this stage in the
same call — that record is what the stage's truth is read from, by probing
the tag on origin. Do not cut with `gh` yourself: a tag cut that way records
nothing. The app's own Ship view is the one exception
— a cut there for the tier this stage stands at records on it too (#2641) —
and it is no substitute for cutting here: it names no run.

`changed: false` means the tag was already published — a re-run after a cut
that landed — and the record names it: a settled success, not a halt.

## 4. Confirm it

Read the release back by the tag the call returned and confirm it points at
the promoted branch's commit.

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
