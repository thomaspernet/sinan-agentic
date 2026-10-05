---
name: merge-to-base
description: Merge one verified child branch into the epic's integration branch, testing the merge result before pushing it.
family: delivery
---
Land one branch — onto the branch the launch names it lands on: the epic's
integration branch where this run's shape cut one, the development branch where
it did not.

## 1. Confirm it is ready to land

The branch's gate verdict passed. A branch that has not been verified is not
merged here — settle the unit `failed` and say so.

## 2. Read which shape this is

The launch's *The run you were launched for* block states it: the epic this run
delivers, the integration branch its shape cut, and the branch your work lands
on. Read the shape and the target there, and read neither off the issue you
are landing. A member issue carries no epic label — that label belongs to the
epic — so a step that decided its shape from its own issue answered standalone
for every member of every epic, and left the work of a run that cut an
integration branch sitting on a branch nothing merges.

Where the block names an integration branch, that branch is what this work
lands on and the standalone path below is not open to this step at all. The
run's own proposal is the epic's, later, so this step opens none.

Where the block names no epic and no integration branch, this run's shape cut
none and never will, and the branch your work lands on is the development
branch.

Whether this repository lands work through a pull request is the launch's
*Pull request on GitHub* line, beside the *Checks* line: `on` or `off`. Read it
there, and never from whether a proposal happens to exist for the branch — a
repository that is `on` and shows none has a proposal stage that has not done
its work, and one that is `off` lands directly however many proposals someone
opened by hand.

Where it reads `off`, this step lands it: merge into that branch, running the
checks step 3 names. Where it reads `on`, the proposal stage lands it instead —
settle the unit and leave it to that stage rather than halting on work that is
finished.

Never name a stage the repository's own settings exclude. A halt that recommends a
pull request to a repository that opens none contradicts its own configuration
and leaves the work on the machine that produced it.

## 3. Merge

Merge into whichever branch step 2 named. Resolve a conflict only when the
resolution is mechanical and both sides are yours; anything that needs a
decision about intent is a halt, not a guess.

Run the test suite on the merge result before you push. A merge that lands
green branches into a red target is the failure this step exists to catch.

On the epic path that is all: the checks the repository declares run once, on
the epic's own head — its proposal, or its integration branch where the
repository opens none — and the epic's merge waits on them. On the standalone
path with no pull request this merge is the last one, so the declared checks
run here.

Where this repository's checks run is the launch's *Checks* line: `none`,
`local_script` with the script it runs, or `gh_workflow` with the workflow
file. Read it there, and never from whether the repository opens a pull
request — a repository can open one that nothing verifies, and one that opens
none can still declare a script.

Nothing has verified this work yet and nothing after this step will, so run
the declared checks on the merge result before you push it:

- `local_script` — run the script the *Checks* line names from the repository
  root, on the merge result. A non-zero exit is a halt naming the script and
  what it printed, and nothing is pushed.
- `none` — the repository declared nothing to verify; there is nothing to run.
- `gh_workflow` — a workflow verifies pull requests and this repository opens
  none, so nothing would ever run it. Halt naming that mismatch — the
  repository's Checks setting names a workflow no pull request ever triggers —
  rather than merging unverified.

## 4. Push

Push the branch you merged into, and confirm the merge landed by reading that
branch back rather than by trusting the command's own exit.

Open no pull request here in either shape: on the epic path the epic's own
proposal is a later stage and one opened here would propose a partial epic, and
on the standalone path a repository that wants one has a stage for it.

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
