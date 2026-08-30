---
name: mint-umbrella-epic
description: Draft an umbrella epic grouping related work and create it only once a person approves the name, the body and the members.
family: planning
---
Draft an umbrella epic that groups related work, and create it only once a
person approves.

## 1. Read what would go under it

The issues named, or the follow-ups a scan produced. Read each one rather than
its title: an umbrella whose members turn out to be two unrelated patterns is
worse than the loose issues it replaced.

## 2. Draft it

A name that says what the group is, and a body carrying what the pattern is, why
it is worth one container, and a checklist with one line per member. The
checklist is the epic's whole substance — a member with no line on it is not in
the epic.

## 3. Present it and stop

Show the draft and wait. Creating an epic is a public, outward-facing act that
reorganises other people's work, and nothing here is written to GitHub until a
person says so. Present the name, the body, and the members it would claim.

## 4. Create it, once approved

`gh issue create` with the `epic` label, then link each member to it with a
`child-of` line in the member's own body. A member already claimed by another
epic is left where it is and named in the report — moving it is a decision the
approval did not cover.

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
