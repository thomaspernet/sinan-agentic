---
name: repair-checks
description: Repair the failing checks on the pull request a run is waiting to merge, on the branch that proposal was opened from.
family: delivery
---
Repair the failing checks on the pull request this run is waiting to merge, on
the branch that proposal was opened from.

The run's merge stage read this proposal's checks as failing. Nothing here
merges, closes or reopens anything: the one act is landing a fix on the head
branch and pushing it, after which the engine reads the checks again on its own
and merges when they pass.

The branch is the one the launch names in *The run you were launched for* — the
integration branch an epic-rooted run cut, or the single member's branch a
standalone run delivered. A proposal for any other branch is not this run's,
however red it looks. Check the branch out if the checkout is not already on it,
and settle `failed` naming the branch rather than working somewhere else if you
cannot.

## 1. Read what actually failed

Find the proposal from the branch — `gh pr list --head <branch>` — then read the
failing run rather than guessing from the summary: `gh pr checks` names the
checks, `gh run view <run-id> --log-failed` gives the failing job's own output.
The failing tool's output is what the fix is written against, so read it before
touching a file. A check that failed for a reason no diff can repair — a missing
secret, a runner that never started, a rate limit — is a halt naming that check,
not a fix.

## 2. Fix the class, not the first instance

Name what failed as a class — a lint rule, a formatting drift, a complexity
grade, a test whose behaviour moved — then clear every instance of it, not the
one line the log happened to print first. A push that fixes the first failure
and leaves the second re-runs the whole check for nothing and puts this session
back where it started.

The fix goes on the head branch as an ordinary commit with a conventional-commit
subject. Do not rebase, force-push, or rewrite what the branch already carries:
the children that merged into it read their own merges by that ancestry.

## 3. Run the check command before you push

Run the repository's check command — the one command its pull request check
runs. In the core app that command is `cd digital_brain_back && uv run
scripts/check.sh`: lint, format, complexity and the test suite, stopping at the
first failure. Run it as written, both halves; the checks belong to that package
and reach their tools through `uv run`, so the bare script path from the
repository root fails on tool lookup rather than on the branch.

A non-zero exit means the push would fail the same check the run is already red
on, so do not make it — fix what the command printed and run it again. Where a
repository ships no such command, say so in the settle note rather than
assembling a substitute of your own, and push having verified what you could.

Then push the branch. The checks are re-read on the pushed head and nowhere
else, so a commit that stays on this machine leaves the proposal exactly as red
as it was.

## 4. Settle

`passed` once the fix is committed and pushed — the checks have not been re-run
yet and nothing here waits for them; the engine holds the merge on its own until
they read passing, and opens this repair again if they fail once more.

`failed`, with the check and what it said, when the failure is not one a diff on
this branch repairs. That is the halt a person answers, and it is the only thing
that stops the run asking again.

## Settling

You were launched for one stage of this run as a whole, not for one document,
so there is nothing to claim. What the run is working — its repository, its
epic and its integration branch — is stated in the launch's own *The run you
were launched for* block; read them there, never from the checkout, which can
hold several epic branches and proposals that are not this run's. Every child
of the run has already settled by the time this stage starts; the work below
acts on what they landed.

Settle with `worklist_set_stage_status`, which takes no uuid — the run and the
stage rode in with the launch: `passed` when the stage did what it says;
`failed` with a `halt_reason` when it could not — the reason in words a
person can act on.

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
