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

Treat it as an epic when the acceptance criteria genuinely split into three or
more workstreams that each deserve their own branch — and when they do, list
that split in the body, because those become the children. A long single-area
feature is not an epic.

## 4. File it

`file_issue` with the title, the body, and the labels for its area and
priority. Report the number it answers with.

When the work came out of a brainstorming session, link each issue filed here
to it once the issue exists: `link_brainstorm_work` with the session's uuid and
the issue as `owner/name#N`. The link is what the issue and the session both
read to say where the work came from, so it is written with the tool and never
as a `brainstorm` line in the body. Call it straight after filing: an issue
filed with `file_issue` is in the mirror by the time that tool returns, and one
filed with `gh` has its repository read afresh on the call. If the tool still
says the mirror holds no such issue, check the coordinate, and if it is right,
name the issue to the person to link from the session.

Filing is the whole job. Do not implement the feature here.

## Writing for GitHub

Anything written onto an issue is public, permanent, and read months later by
someone with no knowledge of the run that produced it. Write for that reader:
third person, present tense, naming the change rather than the process that
produced it. No run identifiers, no internal phase names, no first-person
agent voice, no real names or addresses — a role (`the reporter`, `the
reviewer`) says everything the reader needs.

Every issue write here goes through the app's issue tools — `file_issue`,
`edit_issue` and `delete_issue` — which put the write in the Backlog before
they return. In a session where those tools are not loaded, make the same
write with `gh` instead:
`gh issue create`, `gh issue edit`, `gh issue close` or `gh issue comment`.
A write made that way reaches the Backlog only on the repository's next
refresh, so say so when reporting it rather than reading its absence there as
a failure.

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

A claimed step whose work stops for the person's decision — a proposal they
must approve, a choice only they can make — does not settle at the pause. Call
`worklist_set_item_status` with the `item_uuid`, `status: "waiting"` and a
`question`: one line saying what the person must decide. That settles nothing:
the step stays open and yours, and the run shows them the question. Once they
have answered, do what the answer asks, then settle — never `passed` at the
pause, which reads the step done before they have decided anything.

A `skipped` settle also says what became of this step's work, as an `outcome`
with that outcome's evidence. A skip naming none is refused, and so is one
whose outcome has nothing behind it: you are the only one who knows, and a bare
skip leaves every reader after you guessing which of the three it was.

- `already_delivered` — the work is already done, in this repository or
  another. Give `references`, one per place it landed: a commit as
  `owner/name@sha`, a pull request or issue as `owner/name#123`. Each must
  already be on its repository's development branch, or, in this run's own
  repository, on the run's integration branch when it has one: a commit
  reachable from it, a pull request merged into it, an issue closed by a change
  merged there. Work that sits on an unmerged branch is not delivered, and a
  reference to it is refused by name. It is the one outcome that says
  something shipped, and a run reads it to know this step delivered even
  though nothing landed on its own branch.
- `left_out` — the run decided not to do this work. Give `outcome_reason`, one
  line saying why.
- `not_needed` — the question turned out not to exist. Give `outcome_reason`,
  one line saying why.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on, not as an error string. Never leave a
claimed unit `running`: a step that stops without settling is
indistinguishable from one still in flight.
