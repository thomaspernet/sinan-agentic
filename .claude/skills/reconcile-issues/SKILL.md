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
listing without it cannot tell an orphan from a child. Separate them by shape,
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

Present the proposals and wait. No run stands as the approval here, as one
does for a scan or an umbrella: every link is the person's to approve. As a
claimed step of a run, the wait is a call — `worklist_set_item_status` with the
item's `item_uuid`, `status: "waiting"` and a `question` naming each orphan and
the parent proposed for it — and never `passed`, which reads the step done
before any parent is approved.

On approval, `edit_issue` each approved orphan with its whole body, read first
with `gh issue view` — the `body` an edit sends replaces the one the issue
holds — and a `child-of` line added to it: into the body's existing trailing
`Links:` block if it already carries one, written as the block's first line
above whatever is already there, since the parser reads only the last such
block and a second header below it would silently drop what the first held; a
fresh

```
Links:
- child-of: #N
```

block otherwise. Leave every other issue untouched. An orphan the person skips
is written nothing. As a claimed step, settle once the approved links are
written.

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
