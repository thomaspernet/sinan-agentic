---
name: init-docs
description: Write or refresh the repository's documentation tree, each doc naming the skills and rules the bank should bind it to.
family: writing
---
Write or refresh the repository's documentation tree.

Documentation is what carries knowledge that cannot be recovered from the code —
why a thing is shaped as it is, how the parts connect, what the words mean. That
is what this writes; the code says the rest.

## 1. Read the repository first

Its layout, its entry points, its own README. A documentation set written before
reading the code describes a project that does not exist, and is worse than none
because the next reader believes it.

## 2. Decide what the set should hold

Three audiences, kept apart:

- The cross-project principles — how code is written here, regardless of the
  repository.
- This project's own shape — its architecture, its boundaries, the decisions
  behind them and what each cost.
- What the product does, for a reader who will never open the code.

Write only what this repository needs. A page describing a stack it does not use
is a page that will be wrong before anyone notices it was never right.

## 3. Say who each doc is for

Which skills and which rules need it, written as prose the reader can act on
rather than as a declaration the file expects something to act on. Nothing reads
a file to work out a reading list: a doc reaches an agent because it is in the
bank and somebody bound it there to the consumer that must read it, one consumer
at a time, and that binding is the whole of it.

So the audience you write down is what tells whoever binds the doc where it
goes. A doc nothing is bound to is a doc nothing will ever open.

## 4. Refresh rather than rewrite

On a second pass, read what is installed before changing it. A doc still
accurate is left alone — rewriting an accurate page to look busy is churn a
reviewer has to read. A doc someone adapted is theirs; report what diverged and
recommend, rather than overwriting their edit.

## 5. Leave it uncommitted

Write the files and stop. Documentation is read before it is trusted, so it goes
to a person to review rather than onto a branch. No commit, no pull request, no
issue.

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
  did what it says, `halted` with a `halt_reason` when it could not run at all,
  `manual_review` when it ran but nothing can vouch for the result.
- `claimed: false` with `already_running` — another session has it. Stop.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on, not as an error string. Never leave a
claimed unit `running`: a step that stops without settling is
indistinguishable from one still in flight.
