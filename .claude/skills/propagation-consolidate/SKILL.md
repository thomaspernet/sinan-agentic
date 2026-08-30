---
name: propagation-consolidate
description: Gather the per-site issues one scan filed into a single umbrella per pattern, leaving recurring bug shapes individual.
family: planning
---
Gather the per-site issues one scan filed into a single umbrella per pattern.

You are not a scanner. Read no diff, search no code, and file no per-site issue
— everything you work on is already on GitHub, and finding more of it is the
scan's job, not yours.

## 1. Collect what was filed

`gh issue list --state open` for the per-site propagation issues this issue and
its siblings filed, with the body among the fields — the pattern each one names
and the site it points at both live there, and a listing without it says only
that the issues exist. Leave existing umbrellas out of the collection — an
umbrella is what sites are gathered *onto*, never one of the things gathered.

## 2. Group by pattern

Two issues belong together when they name the same contract, not when they touch
the same file. Group them by that, and treat the site each names as the identity
within its group, so a site filed twice is gathered once.

## 3. Decide what collapses

A mechanical sweep — one shared definition adopted at many sites — collapses.
Every site becomes a checklist line on one umbrella, and the work is done as one
pass.

A recurring bug shape does not. Each of its sites needs its own reading, because
what looks like one bug repeated is often several bugs that resemble each other,
and folding them hides the differences the individual reviews would have found.
Leave those issues open, individual, and untouched — unless a person has already
minted an umbrella for that exact pattern, in which case the decision has been
made and adding to it is bookkeeping rather than judgement.

## 4. Gather

One umbrella per pattern. `gh issue edit` to append to the one that exists
rather than creating a second; `gh issue create` only where none does, seeding
its checklist with every site in the group. Then `gh issue close --comment` each
gathered per-site issue, the comment naming the umbrella and the site's line on
it, so a reader landing on the closed issue can follow it.

Close, and nothing else. Never re-title, re-label, re-link or reopen a gathered
issue, and never close an umbrella.

## Writing for GitHub

Anything written onto an issue is public, permanent, and read months later by
someone with no knowledge of the run that produced it. Write for that reader:
third person, present tense, naming the change rather than the process that
produced it. No run identifiers, no internal phase names, no first-person
agent voice, no real names or addresses — a role (`the reporter`, `the
reviewer`) says everything the reader needs.

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
