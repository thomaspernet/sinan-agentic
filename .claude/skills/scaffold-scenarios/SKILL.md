---
name: scaffold-scenarios
description: Turn each acceptance scenario drafted on an issue into a real, skipped test named after the scenario.
family: writing
capability: acceptance
shipped-from: 2453015149808caac3462da4ecc605d9e31ac13343f905d227f838aa0157d03c
---
Turn each acceptance scenario drafted on an issue into a real, skipped test.

## 1. Read the drafts

The scenario lines on the issue body. Each one is a sentence describing a
behaviour somebody wants covered; none of them is a test yet.

## 2. Write one skipped test per draft

A skipped test, in the file the repository's own conventions put it in, whose
name is the drafted sentence unchanged. The name is the identity that later
matches the test back to the scenario it came from, so it is copied rather than
improved.

Leave the body a stub. You are scaffolding the place a test goes, not guessing
the assertions — a test written from a sentence rather than from the behaviour
passes for the wrong reason, and a passing test nobody wrote is worse than a
missing one.

Skipped, not failing. A red suite from the moment a scenario is drafted trains
everyone to ignore it.

## 3. Report what to do next

Say which files hold the new stubs and that each needs its body written and its
skip removed before it covers anything.

## Reporting back

You are invoked either on demand — by a person who already knows what they want
— or as one step of a run. The two report back differently, so establish which
before doing anything.

Call `worklist_claim_item` with no arguments.

- `no_anchor` — you were invoked on demand. There is no unit to settle: do the
  work above, then report what you produced to the person who asked, naming it
  by issue number or path so they can open it.
- `claimed: true` — you are a step of a run. Do the work above against the
  claimed item's `title` and `attachments`, then settle with
  `worklist_set_item_status` and the item's `item_uuid`: `passed` when the step
  did what it says, `failed` with a `halt_reason` when it did not, and
  `skipped` when the question no longer exists. A step that decided its work
  fails says so with the reason, never with `passed`.
- `claimed: false` with `already_running` — another session has it. Stop.

A `skipped` settle also says what became of this step's work, as an `outcome`
with that outcome's evidence. A skip naming none is refused, and so is one
whose outcome has nothing behind it: you are the only one who knows, and a bare
skip leaves every reader after you guessing which of the three it was.

- `already_delivered` — the work is already done, in this repository or
  another. Give `references`, one per place it landed: a commit as
  `owner/name@sha`, a pull request or issue as `owner/name#123`. It is the one
  outcome that says something shipped, and a run reads it to know this step
  delivered even though nothing landed on its own branch.
- `left_out` — the run decided not to do this work. Give `outcome_reason`, one
  line saying why.
- `not_needed` — the question turned out not to exist. Give `outcome_reason`,
  one line saying why.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on, not as an error string. Never leave a
claimed unit `running`: a step that stops without settling is
indistinguishable from one still in flight.
