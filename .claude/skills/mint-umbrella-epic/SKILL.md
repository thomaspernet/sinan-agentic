---
name: mint-umbrella-epic
description: Draft an umbrella epic grouping related work, and create it once a person approves the name, the body and the members — or at once when a run is the approval.
family: planning
shipped-from: ce683e85d4028a46b68c604eb556ca4c2c88e777e9e7a3d02ab404255d034dc4
---
Draft an umbrella epic that groups related work, and create it once a person
approves — or at once, when a run is the approval.

## 1. Read what would go under it

The issues named, or the follow-ups a scan produced. Read each one rather than
its title: an umbrella whose members turn out to be two unrelated patterns is
worse than the loose issues it replaced.

## 2. Draft it

A name that says what the group is, and a body carrying what the pattern is, why
it is worth one container, and a checklist with one line per member. The
checklist is the epic's whole substance — a member with no line on it is not in
the epic. An umbrella a scan mints for what it filed also names the change
that surfaced it, so the lineage survives its members linking here instead.

## 3. Present it, unless a run is the approval

Invoked on demand, show the draft and wait. Creating an epic is a public,
outward-facing act that reorganises other people's work, and nothing here is
written to GitHub until the person says so. Present the name, the body, and the
members it would claim.

Invoked from a run — the launch carries a *The run you were launched for*
block, or a propagation scan running as a stage of one is minting the umbrella
for what it filed — nobody is there to answer, and the run stands as the
approval: create the epic and link its members without presenting a draft or
waiting. Minted from inside a scan, the scan owns the report and the settle,
so hand it the epic's number and settle nothing here.

## 4. Create it

`file_issue` with the `epic` label and every label its members share, so the
backlog lists it as work that can be run, then link each member to it with
`edit_issue`, sending back the member's whole body with the line added — the
`body` an edit sends replaces the one the issue holds. Read the member's current
body first, with `gh issue view`: the parser that turns a `child-of` line into a
graph edge reads only the body's *last* `Links:` block, so a member already
carrying one — a `blocks` line from other work — keeps that block and never
gets a second header below it, which would silently drop what the first one
held. The new line is written as the block's first line, above whatever is
already there: the edge is what the block is read for, so it is what a person
opening the member meets first.

```
Links:
- child-of: #N
- blocks: #M
```

A member with no block yet gets one carrying that single line.

When the work came out of a brainstorming session, link each issue filed here
to it once the issue exists: `link_brainstorm_work` with the session's uuid and
the issue as `owner/name#N`. The link is what the issue and the session both
read to say where the work came from, so it is written with the tool and never
as a `brainstorm` line in the body. Call it straight after filing: an issue
filed with `file_issue` is in the mirror by the time that tool returns, and one
filed with `gh` has its repository read afresh on the call. If the tool still
says the mirror holds no such issue, check the coordinate, and if it is right,
name the issue to the person to link from the session.

A member already claimed by another epic is left where it is and named in the
report — moving it is a decision the approval did not cover.

Always a fresh epic. Never promote a working issue into the container for its
own siblings: an issue that is both the work and the group around it can never
be closed, because closing it would close them.

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
