---
name: merge-feature
description: Merge the branch a standalone run delivered — through its pull request once the checks and the gate have passed, or into the development branch directly where the repository opens none.
family: delivery
shipped-from: 19ca156763bbcfec6cda75929adc8f7cc67ffd4f62b90f7d0130dae4dc994db3
---
Merge the branch this run delivered — through its pull request where the
repository opens one, the branch itself directly where it opens none.

This run has one member and no integration branch: the branch its member wrote
is the branch the whole run delivered, and it is the one the run's pull request
proposed where the repository opens one. The launch names it in *The run you
were launched for*, and the development branch likewise. A proposal for any
other branch is not this run's, however finished it looks; a run whose named
branch has no proposal and no merge you can make is a halt naming that branch,
never a pass on a sibling's.

## 1. Read how this repository lands work

Whether this repository lands work through a pull request is the launch's
*Pull request on GitHub* line, beside the *Checks* line: `on` or `off`. Read it
there, and never from whether a proposal happens to exist for the branch — a
repository that is `on` and shows none has a proposal stage that has not done
its work, and one that is `off` lands directly however many proposals someone
opened by hand.

Where this repository's checks run is the launch's *Checks* line: `none`,
`local_script` with the script it runs, or `gh_workflow` with the workflow
file. Read it there, and never from whether the repository opens a pull
request — a repository can open one that nothing verifies, and one that opens
none can still declare a script.

## With the switch `on`: merge through the pull request

Where the run opened a pull request, the checks already ran on it — the
backend runs a local script against the proposal's head, GitHub runs a
workflow — and the engine launched this stage only once that reading let it.
The launch states the reading as *Checks result recorded at the merge*. Merge
when it reads `passing`, or when the *Checks* line reads `none`. Anything else
while checks are declared — `pending`, `failing`, or no result recorded — is a
halt naming the reading, never a merge with a note. Do not run the checks
yourself to make up for it: a result produced in this session is one the
engine never reads. The acceptance gate must have passed too.

Merge through the pull request with a merge commit — `gh pr merge --merge`.
GitHub allows squash and rebase too, and both write a new commit while leaving
the branch they merged exactly where it stood, so the cleanup stage that
follows, which proves its own work by that branch's ancestry, can no longer
see the merge. This stage reads a merged proposal by the commit its merge
wrote and so passes on any of the three — a squash already made is not a halt
— but the method is named here so the rest of the chain has one proof to read.

Confirm the merge landed by reading the development branch back rather than by
trusting the command's own exit.

## With the switch `off`: merge directly

The delivered branch here is the branch this run delivered.

Nothing proposes it and nothing after this stage verifies it. The engine ran
the declared checks at its head and launched this stage only once they let it;
the launch states that reading as *Checks result recorded at the merge*, with
the commit it was read at and, for a local script, the development branch
commit merged into it before it ran. Do not run the checks yourself: a result
produced in this session is one the engine never reads.

1. **Read the head.** `git fetch origin`, then read the commit the delivered
   branch stands at on origin. It must be the commit the recorded result
   names. A head that moved since — a push after the checks ran — has no
   result of its own yet: halt naming both commits.
2. **Require green at that head.** The recorded result must read `passing`.
   `pending`, `failing`, or no result recorded is a halt naming the reading.
   Where the *Checks* line reads `none` the repository declared nothing to
   verify, so there is no result to require — skip this step and step 1's
   comparison with it. The acceptance gate, where the chain carries one, must
   have passed too.
3. **Update it with the development branch.** Check out the delivered branch
   at origin's commit and merge `origin/<development-branch>` into it. Already
   up to date, or the development branch still at the commit the recorded
   result says was merged in, means the merged head is the code the checks
   verified — go on. Otherwise the development branch moved after the checks
   ran and the merged head is code nothing verified: push the updated branch
   so the engine checks its new head, and halt naming that commit rather than
   merging it. With checks `none` there is nothing to verify, so go on either
   way. A conflict is resolved only when the resolution is mechanical;
   anything that needs a decision about intent is a halt.
4. **Merge.** Check out the development branch at origin's commit and merge
   the delivered branch into it with a merge commit — `git merge --no-ff`.
   Never squash or rebase: the cleanup stage proves this landing by the
   delivered branch's ancestry, which a squash or a rebase erases.
5. **Push and record.** Push the development branch, then read the landing
   back from origin rather than trusting the command's own exit:
   `git merge-base --is-ancestor <head> origin/<development-branch>` must
   hold for the head step 1 read. Name the merge commit's hash in the settle
   note — it is the commit this run landed.

Make no `gh pr` call on this path — not to list, open or merge a proposal.

Nothing ahead of this stage has landed that branch, and nothing behind it will.
The per-member merge and the per-member delete both take an integration branch
as their subject, so a run of this shape never walks either — this step
performs the merge rather than confirming one another step made.

Do not delete the branch here in either shape — the cleanup stage owns that,
and deleting it early takes the diff a later stage still has to read.

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
