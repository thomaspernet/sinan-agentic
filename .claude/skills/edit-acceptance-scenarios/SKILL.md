---
name: edit-acceptance-scenarios
description: Add or remove the acceptance scenarios an issue tracks, changing nothing else about its body.
family: writing
capability: acceptance
---
Add or remove the acceptance scenarios an issue tracks.

## 1. Read the issue as it stands

`gh issue view` for the body, and read the scenario lines it already carries. An
edit computed against a body you did not read overwrites whatever changed since.

## 2. Compute the new body

Add a scenario by appending its line to the block the body already keeps them
in; remove one by deleting exactly that line. Adding one already there, or
removing one not there, is a no-op rather than an error — this is edited from
more than one surface, and both must be safe to repeat.

Change nothing else. Every other line of the body stays byte for byte as it was:
the edit rewrites the whole body, so an accidental reflow is a silent rewrite of
somebody's issue.

## 3. Write it back

`gh issue edit` with the new body, then read it back and confirm the lines you
intended are the lines that are there.

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
