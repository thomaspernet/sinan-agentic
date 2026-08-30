---
name: new-bug
description: File one bug as a GitHub issue — the observed behaviour, the steps that reproduce it, and criteria a reviewer can tick off.
family: writing
---
File one bug as a GitHub issue.

## 1. Establish what is broken

Separate what was observed from what it is assumed to mean. A report of "search
is broken" is a symptom; the issue needs the input, the observed result, and the
result that was expected instead. Reproduce it if you can reach it — a bug you
have seen is worth more to whoever fixes it than one you have only been told
about.

## 2. Write it

The title states the problem, never the fix, and stays under about eighty
characters. The body carries what is broken and its observable impact, the steps
that reproduce it, and acceptance criteria a reviewer can tick off by running
something. "Works properly" is not a criterion; "returns 400 with an error body
on an empty payload" is.

## 3. Place it

Read the description for a stated parent — "regression of #N", "found while
building #N". A stated parent is confirmed once with the person and then written
as a `child-of` link; a passing mention ("see #N") is not one, and is not
linked. Nothing is linked without asking.

Most bugs are one issue. Treat it as an epic only when the cause genuinely
splits into three or more independent fixes that cannot share a branch — a long
reproduction is not the same thing as a wide one.

## 4. File it

`gh issue create` with the title, the body, and the labels for its area and
priority. Report the number.

Filing is the whole job. Do not fix the bug here — an issue and its fix reviewed
together is an issue nothing reviewed.

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
