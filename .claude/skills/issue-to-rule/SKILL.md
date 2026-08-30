---
name: issue-to-rule
description: Turn one resolved issue into a rule when the mistake it fixed is a class rather than a one-off.
family: analysis
---
Turn one resolved issue into a rule, so the same mistake stops recurring.

This skill writes rule text and nothing else. It edits no application code, runs
no tests, and commits no behaviour change.

## 1. Read what actually happened

The issue, the review it went through, and the diff that closed it. The rule is
about the mistake, not the symptom, so keep reading until you can say what a
person would have had to know beforehand to avoid it.

## 2. Decide whether there is a class here

Most fixes are correct and not generalisable. A rule is worth writing only when
the same mistake can plausibly be made again somewhere else — usually shown by
it having already been made twice, in different files or by different authors.
One instance is feedback about one change.

If there is no class, say so plainly and stop. A rule file full of one-off
observations is one nobody reads, which costs more than the rule saved.

## 3. Write it

Three parts, in this order:

- **The constraint** — one line, stated as what to do, not as what went wrong.
- **Why** — the incident it comes from, named concretely enough that a reader
  can go and look at it.
- **How to apply** — when it fires and how to tell a real instance from
  something that merely resembles one. This is the part that decides whether the
  rule is usable, so it carries the edge cases rather than the constraint line.

## 4. Put it where it belongs

A constraint about this repository's own code goes in that repository's rules. A
constraint that would hold in any codebase goes with the principles, where every
project reads it. Before writing either, read what is already there: a rule that
contradicts an existing one leaves a reader to guess, and a rule that repeats
one is the duplication these rules exist to prevent.

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
