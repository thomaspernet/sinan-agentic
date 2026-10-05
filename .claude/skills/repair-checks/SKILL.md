---
name: repair-checks
description: Repair the failing checks on the branch a run is waiting to merge — on its pull request where the repository opens one, on the branch head itself where it opens none.
family: delivery
---
Repair the failing checks on the branch this run is waiting to merge — the
checks on its pull request where the repository opens one, on the branch head
itself where it opens none.

The run read those checks as failing while it waits to merge. Nothing here
merges, closes or reopens anything: the one act is landing a fix on the branch
and pushing it — or, for a failure that looks like a flake, rerunning the
failed jobs once — after which the engine reads the checks again on its own and
merges when they pass.

The branch is the one the launch names in *The run you were launched for* — the
integration branch an epic-rooted run cut, or the single member's branch a
standalone run delivered. Checks failing on any other branch are not this
run's, however red they look. Check the branch out if the checkout is not
already on it, and settle `failed` naming the branch rather than working
somewhere else if you cannot.

## 1. Read what actually failed

Where the failure is read depends on where this repository runs its checks, and
the launch says which.

- **Local checks** — the launch carries *The local checks this repair reads*:
  the script the repository declares, the head commit it ran on, the
  development branch commit merged into it, its exit code and the tail of what
  it printed. That record is the failure; GitHub holds no check run for it, so
  do not look for one there. A record whose head is not the branch's current
  head, or whose status is not `failed`, means the checks moved on since this
  repair opened: push nothing and settle `passed`, and the engine reads the
  checks again on its own.
- **GitHub checks** — the launch carries no such section, and the checks ran on
  the run's pull request: only a repository that opens one runs them on GitHub.
  Find it from the branch — `gh pr list --head <branch>` — then read the
  failing run rather than guessing from the summary: `gh pr checks` names the
  checks, `gh run view <run-id> --log-failed` gives the failing job's own
  output.

The failing tool's output is what the fix is written against, so read it before
touching a file. A check that failed for a reason no diff can repair — a missing
secret, a runner that never started, a rate limit, a script the machine could
not start for a reason outside the branch — is a halt naming that check, not a
fix, and not a rerun either.

## 2. Rerun a likely flake once

Some failures are no diff's to fix: a test that races, in code the branch never
touched, fails now and then on any branch. Decide whether this is one before
writing a fix, on two readings together:

- the failing tests sit in paths the branch's diff (`git diff <base>...HEAD`,
  against the development branch it merges into) does not touch;
- the failure is timing or concurrency shaped — a race, a timeout, an ordering
  that differs between runs — as opposed to failing the same way every time.

When both hold, the failure is a likely flake. Local checks cannot be rerun from
here — the backend runs the script once per head commit — so on a local-script
repository a likely flake settles `failed`, the halt reason naming the test and
saying it looks timing shaped. On GitHub checks, read how many times the failing
run has already been attempted: `gh run view <run-id> --json attempt,headSha`.
The cap is one rerun per head commit, because the engine opens this repair
again on every red reading and would otherwise rerun the same flake forever.

- Attempt `1` on the branch's current head: rerun only the failed jobs,
  `gh run rerun <run-id> --failed`, push nothing, and settle `passed`. The
  engine goes back to waiting on the checks and reads the rerun when it ends.
- Attempt above `1`: the rerun was already tried on this head, and the failure
  came back. Settle `failed`; the halt reason names the test, says it failed on
  the rerun too, and points to the bug filed for the flake if one exists.

A failure in code the branch touches, or one that fails the same way every
time, is not a flake: fix it as below. A failure no rerun can repair — a
missing secret, a runner that never started, a rate limit — stays the halt
step 1 describes.

## 3. Fix the class, not the first instance

Name what failed as a class — a lint rule, a formatting drift, a complexity
grade, a test whose behaviour moved — then clear every instance of it, not the
one line the log happened to print first. A push that fixes the first failure
and leaves the second re-runs the whole check for nothing and puts this session
back where it started.

A failure in code the branch never touched still gets fixed here, on this
branch. The development branch can carry something already broken — local
checks run with it merged in, so its faults show up as this branch's — and
this repair is the only thing that clears it before the merge. Say in the
settle note that the fault came from earlier work on the development branch,
naming what it was, so a person can trace where it came from.

The fix goes on the branch as an ordinary commit with a conventional-commit
subject. Do not rebase, force-push, or rewrite what the branch already carries:
the children that merged into it read their own merges by that ancestry.

## 4. Run the check command before you push

Run the repository's check command — the one its checks run.

- **Local checks** — the script the launch names, run from the repository root
  the way the backend runs it. The backend runs it with the development branch
  merged in, so a pass on the branch alone can still fail there. Commit the fix
  first, then merge the development branch in without committing
  (`git merge --no-commit --no-ff origin/<development branch>`), run the
  script, and `git merge --abort` before touching anything else — the abort
  discards every uncommitted change, which is why the fix is committed first.
- **GitHub checks** — the command the workflow the launch's *Checks* line
  names runs. In the core app that command is
  `cd digital_brain_back && uv run scripts/check.sh`: lint, format, complexity
  and the test suite, stopping at the first failure. Run it as written, both
  halves; the checks belong to that package and reach their tools through
  `uv run`, so the bare script path from the repository root fails on tool
  lookup rather than on the branch.

A non-zero exit means the push would fail the same check the run is already red
on, so do not make it — fix what the command printed and run it again. Where a
repository ships no such command, say so in the settle note rather than
assembling a substitute of your own, and push having verified what you could.

Then push the branch. The checks are re-read on the pushed head and nowhere
else — a local script runs again only for a new head — so a commit that stays
on this machine leaves the checks exactly as red as they were.

## 5. Settle

`passed` once the fix is committed and pushed, once a likely flake's failed
jobs are rerun, or once the local record shows the checks moved on — the checks have not been read again yet and nothing here waits
for them; the engine holds the merge on its own until they read passing, and
opens this repair again if they fail once more.

`failed`, with the check and what it said, when the failure is not one a diff on
this branch repairs and not a flake's first rerun — including a flake that
failed on the rerun too. That is the halt a person answers, and it is the only thing
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
