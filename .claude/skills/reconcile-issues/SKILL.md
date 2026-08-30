---
name: reconcile-issues
description: Give an orphan issue a parent, on approval — and report, without touching, the issues whose link is already right.
family: planning
---
Give an orphan issue a parent, so it stops being the only member of its own
group.

## 1. Find the orphans

An orphan is an open issue whose body carries no `child-of` link. `gh issue
list --state open` with the body among the fields — the link lives there, so a
listing without it cannot tell an orphan from a child. Separate three shapes,
because only the first is yours:

- **No link at all** — the orphan. It has nowhere to belong until one is
  written. Continue with these.
- **Linked, but the parent is a label-only epic that roots no work** — the link
  is already right; what is missing is a workflow on the parent. Writing a
  second link fixes nothing. Report it and leave it.
- **Linked, and the link disagrees with where the work is actually running** —
  again the link is right and the membership is wrong. Report it and leave it.

## 2. Propose a parent for each orphan

For every orphan, name the epic it most plausibly belongs under and say why in
one line. An orphan with no plausible parent stays an orphan — inventing an epic
to hold one issue is how a tree becomes noise.

## 3. Ask, then write

Present the proposals and wait. On approval, `gh issue edit` each approved
orphan with the `child-of` line added to the body it already carries, and leave
every other issue untouched. An orphan the person skips is written nothing.

Only ever the link. Never re-title, re-label, close, or reopen an issue here —
this skill answers one question about an issue and touches nothing else about
it.

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