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

One umbrella per pattern. `edit_issue` to append to the one that exists rather
than creating a second, sending its whole body back with the new checklist
lines — the `body` an edit sends replaces the one the issue holds; `file_issue`
only where none does, seeding its checklist with every site in the group. Then
close each gathered per-site issue with `edit_issue`, `state: closed` and a
`comment` naming the umbrella and the site's line on it, so a reader landing on
the closed issue can follow it.

Close, and nothing else. Never re-title, re-label, re-link or reopen a gathered
issue, and never close an umbrella.

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
