---
name: promote
description: Promote the merged work onto the next branch of the cascade, reusing the checks result its merge recorded.
family: delivery
shipped-from: 79c00e838ade4aace170dd79ac4085470f8699b1bd1a311f5e8841bd32ea75bf
---
Promote the merged work onto the next branch of the cascade.

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

## 2. Promote through the run

Call `worklist_promote`. It takes no arguments: the stage you were launched for
names the tier it promotes onto. The app reads where the tier below stands,
merges it onto the tier above, and records that commit on this stage in the
same call — that record is what the stage's truth is read from, by asking
whether the commit is contained in the tier. Do not promote with `git`, `gh` or
a pull request yourself: a promotion made that way records nothing, and a stage
whose record names no commit classifies as nothing promoted.

Promotion moves what is already there; it never rewrites history and never
force-pushes. A promotion that finds nothing to move still records the commit
it read — a re-run after a promotion that landed is a settled success, not a
halt.

A conflict at a promotion boundary means the tiers have diverged, which is a
decision about intent — settle `failed` with what the call answered, rather
than resolving it here.

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
