---
name: scaffold-scenarios
description: Turn each acceptance scenario drafted on an issue into a real, skipped test named after the scenario.
family: writing
capability: acceptance
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
  did what it says, `halted` with a `halt_reason` when it could not run at all,
  `manual_review` when it ran but nothing can vouch for the result.
- `claimed: false` with `already_running` — another session has it. Stop.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on, not as an error string. Never leave a
claimed unit `running`: a step that stops without settling is
indistinguishable from one still in flight.
