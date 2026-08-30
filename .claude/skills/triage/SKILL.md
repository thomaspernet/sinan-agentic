---
name: triage
description: Read the open issues and say which deserve attention and why, deciding nothing.
family: analysis
---
Read what is open and say what deserves attention.

## 1. Read the open set

`gh issue list --state open` with the fields a decision needs — number, title,
labels, age. Read the whole set rather than the first page: the issue that has
been open longest is exactly the one a page limit hides.

## 2. Say what stands out

Present it as one table, ordered so the top row is the one to act on. What earns
a place at the top:

- An issue carrying no priority — nobody has decided about it yet, which is a
  different state from having decided it can wait.
- A high-priority issue with no branch and no recent activity.
- An issue filed automatically that no person has read.
- A cluster of issues describing one underlying cause, which is one epic
  wearing several numbers.

## 3. Stop there

Triage decides nothing. Say what you would pick up first and why, then ask —
the answer is the person's, and an agent that starts work here has taken it.

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
