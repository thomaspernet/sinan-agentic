---
name: issue-to-rule
description: Turn one resolved issue into a rule when the mistake it fixed is a class rather than a one-off.
family: analysis
shipped-from: 50ae0c78896efe7689316693dbddd056ac4086343aa77cda0680e8f5096a33ed
---
Turn one resolved issue into a rule, so the same mistake stops recurring.

This skill writes rule text and nothing else. It edits no application code, runs
no tests, and commits no behaviour change.

## 1. Read what actually happened

The issue, the review it went through, and the diff that closed it. As a stage
of a run that is every issue the run resolved — the epic and each member, with
`gh issue view` — and the run's finished diff, pinned: the *Pinned diff* line
of the launch's *The run you were launched for* block names it as two commits,
and the stage opens in a copy of the code checked out at the second, so read it
there as `git diff <base>..HEAD` — never the delivered branch against the
development branch by name, which reads as no change at all once the merge
beside this stage lands. Read them together, once. Two members fixing the same
mistake is the evidence of a class the next step asks for, and a pass over one member at a
time could never see it. The rule is about the mistake, not the symptom, so
keep reading until you can say what a person would have had to know beforehand
to avoid it.

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

You are invoked on demand — by a person who already knows what they want — or
as one stage of a run, which runs you once for the run as a whole rather than
once per member. The two report back differently, so establish which before
doing anything.

Call `worklist_claim_item` with no arguments.

- `no_anchor`, and the launch carries a *The run you were launched for* block —
  you are a stage of that run, and the run is the unit: there is nothing to
  claim. Do the work above against the run's finished change, then settle with
  `worklist_set_stage_status`, which takes no uuid: `passed` when the stage did
  what it says, `failed` with a `halt_reason` when it could not.
- `no_anchor`, and the launch carries no such block — you were invoked on
  demand. There is no unit to settle: do the work above, then report what you
  produced to the person who asked, naming it by issue number or path so they
  can open it.
- `claimed: true` — you are one member's step of a run over a workflow someone
  wrote, which binds this skill to a step of its own. Do the work above against
  that member's change, read from the claimed item's `title` and
  `attachments`, then settle with `worklist_set_item_status` and the item's
  `item_uuid`: `passed` when the step did what it says, `failed` with a
  `halt_reason` when it did not, and `skipped` when the question no longer
  exists.
- `claimed: false` with `already_running` — another session has it. Stop.

Only the claimed-item branch settles `skipped` at all:
`worklist_set_stage_status` takes no outcome, and an on-demand invocation
settles nothing.

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
