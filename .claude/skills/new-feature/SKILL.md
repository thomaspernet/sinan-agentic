---
name: new-feature
description: File one feature as a GitHub issue, with acceptance criteria as the contract the work is reviewed against.
family: writing
---
File one feature as a GitHub issue.

## 1. Establish what is being asked for

The ask is the outcome, not the implementation someone has in mind for it. Write
down what would be true once it shipped; if that cannot be stated, the request is
not yet an issue and the honest answer is to say so rather than to file a vague
one.

## 2. Write it

The title states the ask in one line. The body carries what is missing and why
it matters, then acceptance criteria — the contract, and the part worth the most
care. Each one is something a reviewer can tick off by reading the diff or
running a command. Name the areas the work touches and why each is affected.

## 3. Place it

Read the description for a stated parent — "part of epic #N", "extends #N".
Confirm it once with the person before writing a `child-of` link; a passing
mention is not a parent and is not linked.

Treat it as an epic when the acceptance criteria genuinely split into three or
more workstreams that each deserve their own branch — and when they do, list
that split in the body, because those become the children. A long single-area
feature is not an epic.

## 4. File it

`gh issue create` with the title, the body, and the labels for its area and
priority. Report the number.

Filing is the whole job. Do not implement the feature here.

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
