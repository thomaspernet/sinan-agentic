---
name: new-bug
description: File one bug as a GitHub issue — the observed behaviour, the steps that reproduce it, and criteria a reviewer can tick off.
family: writing
shipped-from: ce23adadc0258be3204d682fdb3796a3a7be0a88708a061f00a1412be6a98e7a
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
building #N". A stated parent is confirmed once with the person and then
written as a `child-of` link; a passing mention ("see #N") is not one, and is
not linked. Nothing is linked without asking.

A link is a trailing block, not a sentence — the parser that turns it into a
graph edge reads only a line shaped exactly `child-of: #N` under its own
`Links:` header, both required, and finds nothing from any other phrasing:

```
Links:
- child-of: #N
```

A body that already carries a block — a `blocks` line naming other work —
keeps that one block, and the `child-of` line is written as the block's first
line, above whatever is already there. The edge is what the block is read for,
so it is what a person opening the issue meets first.

Most bugs are one issue. Treat it as an epic only when the cause genuinely
splits into three or more independent fixes that cannot share a branch — a long
reproduction is not the same thing as a wide one.

## 4. File it

`gh issue create` with the title, the body, and the labels for its area and
priority. Report the number.

When the work came out of a brainstorming session, link each issue filed here
to it once the issue exists: `link_brainstorm_work` with the session's uuid and
the issue as `owner/name#N`. The link is what the issue and the session both
read to say where the work came from, so it is written with the tool and never
as a `brainstorm` line in the body. An issue filed a moment ago may not have
reached the mirror yet, and the tool says so: call it again once the issue has
arrived rather than straight away, and if it still has not, name the issue to
the person to link from the session.

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
