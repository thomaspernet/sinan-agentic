---
name: implement
description: Implement one issue on its own branch — read the issue and its lineage, follow the closest existing implementation, and test before committing.
family: delivery
---
Implement one issue on its own branch.

## 1. Read the issue and the work already done

Read the issue with `gh issue view <N> --json title,body,labels`. Read its
`child-of` links too: a sub-issue of an epic inherits decisions the parent
already fixed, and re-deriving them produces a second answer to a settled
question.

Check whether the issue has been attempted before — an existing branch, a
prior verdict. A re-run is guided by that feedback rather than by the original
acceptance criteria alone: treat each finding as an instance of a class and
clear the class, and build on any `chore(gate):` commits already on the branch
rather than reverting or re-applying them.

Then read the base branch before you write anything against the criteria. An
issue is filed when it is filed, and a sibling can ship its fix while it sits
in the backlog — #3167 asked for a fix that had already landed as #3172, and
three runs in a row settled it bare with that commit sitting in the history
none of them read. Read the files the issue names as they stand on the branch
you were cut from, and search that branch's history for the issue's subject:
`git log --oneline <base> --grep=<the words the issue uses>`, then read what it
returns. A criterion the tree already meets is met however recently it was met.

What that read finds decides the step. All of it already there settles
`already_delivered` against the commit that put it there — step 7, before a
line is written. Part of it there implements the rest as usual and says in the
debrief which part it found. None of it there is the ordinary case below.

## 2. Find the reference before writing anything

Classify the work — new entity, endpoint, component, migration — then find the
closest existing implementation of that kind and read it. The patterns you
follow come from the codebase, not from memory.

## 3. Branch, implement, test

The run cut your branch and recorded its name before this session opened, and
the launch names it: work on that one. Check it out if you are not on it, and
cut a branch of your own only when the launch named none — the recorded name is
what every later read of this work uses, so work anywhere else reads as work
that never happened.

Implement against the issue's acceptance criteria. A bug is fixed against a test
that fails before the fix and passes after it, written first — what it proves is
then the bug rather than the fix.

Before you commit, run the checks this repository declares — the same ones
the quality gate runs against this diff.

Where this repository's checks run is the launch's *Checks* line: `none`,
`local_script` with the script it runs, or `gh_workflow` with the workflow
file. Read it there, and never from whether the repository opens a pull
request — a repository can open one that nothing verifies, and one that opens
none can still declare a script.

- `local_script` — run the script the *Checks* line names from the repository
  root.
- `gh_workflow` — run the command the workflow file the *Checks* line names
  runs. In the core app that command is
  `cd digital_brain_back && uv run scripts/check.sh`: lint, format, complexity
  and the test suite, stopping at the first failure. Run it as written, both
  halves; the checks belong to that package and reach their tools through
  `uv run`, so the bare script path from the repository root fails on tool
  lookup rather than on the branch — a failure that reads exactly like the
  branch's own. The suite is the last of its four steps, so a green pytest run
  on its own leaves lint, format and complexity unread, and those are what a
  round fails at the gate for.
- `none` — the repository declared nothing to verify, so there is nothing to
  run. Say so in the settle note rather than assembling a substitute of your
  own.

A non-zero exit means the commit would fail the same check one stage later, so
do not push it — fix what the command printed and run it again.

Commit with a conventional-commit subject naming the issue it closes.

Then push the branch to origin. The step's verdict is read there and nowhere
else — the recorded branch, ahead of the base it was cut from — so a commit
that stays on this machine reads exactly like a branch nothing was committed
on, and the pass is outvoted however green the suite was. The push is part of
the step rather than tidying up after it: it happens before you settle,
because the reading is taken the moment you do.

## 4. Review your own diff before settling

The gate grades this diff against the checklists this launch already primed, so
most of what it would find is reachable from here. A unit settled with a known
failing item is a round trip nobody needed.

- **Walk the checklists against the diff**, and fix what fails now.
- **Ask where else the change belongs.** Every change has a family: a sentence
  restating the old behaviour, a second copy of a literal this diff introduces,
  another instance of a shape fixed once. Search with ignore rules off —
  ignore-respecting search skips the very surfaces a stale reference survives
  in — and clear every member.
- **Walk the fan-out.** List an edited function's branches and its callers, and
  confirm the change reached each symmetric sibling: the other scope, the other
  opener, the matching config path. A fix applied to one of two twin paths is
  the most common behaviour failure, and no text search surfaces it.
- **Your own additions are review surface.** Docstrings, tests and helpers
  written this round are graded on the same terms as the code.

## 5. Write the debrief

The commit and the branch are this item's deliverables; the debrief is its
account of itself, for the people who were not there — the quality gate, the
documentation pass at the end of the run, and a person reading the issue. A
`passed` settle is refused with `missing_debrief` until the item carries one,
and no other output stands in for it.

Write it with `create_page`, `type_name: "debrief"`, in three sections:

- **What a user of the app would notice** — the behaviour, in plain words,
  before any file is named.
- **What changed and why** — the surfaces, endpoints or entities touched, and
  the reasoning for the shape chosen.
- **What was deliberately left alone** — considerations weighed and not done,
  follow-ups the unit could not take.

Not a file-by-file diff: the commit carries that. The page is filed against
the item, run and step this session was launched for, so there is nothing else
to pass. Record it with `worklist_add_output` and the item's `item_uuid`, then
settle.

## 6. Record what you noticed

The gate reads the debrief before it grades, and its third section aims the
review, so what you noticed belongs there: a risk a reviewer should watch, a
consideration weighed and deliberately not done, a follow-up this unit could
not do. The family sweep above belongs there as a consideration — naming what
was searched and cleared lets the reviewer verify it rather than repeat it.

## 7. Nothing to implement

If the work is already in the tree — shipped by a sibling, or the criteria are
already met — do not fake a commit and do not exit silently. Settle the unit
`skipped` with the `already_delivered` outcome below, referencing where the
work is; this repository is as nameable as any other. If it belongs in other
repositories rather than this one, it counts as delivered only once it has
reached that repository's development branch, through that repository's own
delivery — a commit you push to a branch there is not delivered yet, and a
reference to it is refused. Where it has reached it, settle the same way, with
a reference into each repository it landed in. Where it has not, do not cite
it: settle `left_out` with an `outcome_reason` naming where that repository now
tracks the work — its issue, or its open pull request — so this issue stays
open until the work lands there.

Write the debrief of step 5 for that skip too. The references say where the
work landed; the debrief says why it answers *this* issue — each acceptance
criterion against the test or the code that satisfies it. A hash on its own
leaves the next reader to work the match out again, which is the re-derivation
a recorded outcome replaced.

That is what tells the run this skip shipped. The run's close stage closes
this issue by the outcome recorded here, whichever shape the run is — an
epic's member with the epic, a standalone run's one issue on its own — so the
references are what that close writes into the comment it leaves. A run whose
members all landed elsewhere has nothing on its own branch to propose or
merge, and takes on neither stage.

A skip that is not delivery says which of the other two it is: `left_out` for
work this run decided not to do, whose issue stays open and returns to the
backlog, or `not_needed` for work that turned out not to exist, whose issue
closes as not planned.

## Claiming and settling

You were launched for one unit of this run, so there is nothing to look up.

1. Call `worklist_claim_item` with no arguments. `claimed: false` with
   `already_running` means another session has it — stop. `no_anchor` means
   this session was not launched for a unit — stop and say so.
2. Do the work below against the claimed item's `title` and `attachments`.
3. Settle with `worklist_set_item_status` and the item's `item_uuid`:
   `passed` when the step did what it says; `failed` with a `halt_reason`
   when it did not — the reason in words a person can act on; `skipped` when
   the question no longer exists.

   A step that decided its work fails settles `failed` and says why,
   never with `passed` — the status write refuses a pass over a verdict that
   judged this attempt, the engine outvotes a pass the world contradicts,
   and a failing gate verdict routes the step back by policy whatever is
   written.

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

Give the settle a `note`: a few sentences in your own words for a person
reading the issue later — what you found, what you chose, and what you left.
It is stored verbatim against this attempt, so write prose, not a status
string and not a commit message. It is optional — a settle with no note is
valid — and it is not the `halt_reason`: the reason says what stopped the
unit, the note says what the work was.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on — not as an error string. Never leave the
unit `running`: a step that stops without settling is indistinguishable from
one still in flight.
