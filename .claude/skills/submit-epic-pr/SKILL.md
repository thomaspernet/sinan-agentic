---
name: submit-epic-pr
description: Open the epic's pull request into the development branch, once every child has landed and no proposal is already open for the branch.
family: delivery
---
Open the epic's pull request into the development branch.

## 1. Confirm every child has landed

Read the epic's children. A child still open, or one whose branch has not
merged into the integration branch, means the epic is not ready to propose —
settle `halted` naming the child.

## 2. Check for a proposal already open

`gh pr list --head <integration-branch> --state open`. One open pull request
per branch: a second proposal for one branch splits review across two threads.
If one is open, this step is already done.

## 3. Open it

Open the pull request from the integration branch into the development branch.
The body summarises what the epic changed and how a reviewer convinces
themselves it works — the children's own titles, not a restatement of every
commit.

Write nothing a reader outside this run cannot understand: no run identifiers,
no phase names, no first-person agent voice.

## Settling

You were launched for one stage of this run as a whole, not for one document,
so there is nothing to claim and nothing to look up. Every child of the run has
already settled by the time this stage starts; the work below acts on what they
landed.

Settle with `worklist_set_stage_status`, which takes no uuid — the run and the
stage rode in with the launch: `passed` when the stage did what it says;
`halted` with a `halt_reason` when it could not run at all; `manual_review`
when it ran but nothing can vouch for the result.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on — not as an error string. Never leave the
unit `running`: a step that stops without settling is indistinguishable from
one still in flight.
