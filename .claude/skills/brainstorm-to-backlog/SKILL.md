---
name: brainstorm-to-backlog
description: Read a brainstorming session and propose the work its thinking arrived at, for a person to edit — or file it when asked.
family: analysis
---
Read a brainstorming session and say what work its thinking arrived at —
either as a proposal a person edits, or as the backlog item itself.

You were launched from the moment a session is being sent to the backlog, so
the session is the attachment you were given. Propose unless the person asked
you to file: a proposal opens in their composer and they change it before
anything lands, which is the whole reason to run this rather than type it.

## 1. Read the session

`read_brainstorm_session` with the session's uuid. It answers with the
session's notes and the entities gathered around it as uuids and names — not
as content — plus `backlog_item_uuid`, which is set when the session has
already produced its item. If it is set, stop and say so: a session produces
one item, and there is nothing left to propose or file.

Then read what matters with the `read` tool: the summary first, then the notes
whose names suggest they carry the conclusion, then the gathered documents the
notes lean on. Reading everything is rarely worth it; reading nothing makes
every step below a guess dressed as a plan.

## 2. Decide what the work is

The work is what the thinking arrived at, which is rarely what the session is
called — a session is named for the question it was opened on. State it in one
line, in the words the notes use.

Then decide whether it breaks into steps. It does when the session's own
thinking splits into parts that are done in an order and reviewed separately;
it does not when the session converged on one thing that happens to be large.
Name a step only where the session supports it. A plan invented to look
thorough is the failure this skill exists to avoid — the person can read the
notes too, and a step they cannot trace back to them costs them the trust they
would otherwise put in the rest.

## 3. Offer it

`propose_brainstorm_conversion` with the session's uuid, the title, and the
steps in the order they would be done. Nothing is filed: the proposal lands in
the composer the person has open, where they edit it and confirm what actually
reaches the backlog.

Then say what you proposed and what in the session it came from — one line per
step, naming the note or document behind it. That is what the person is
reviewing; a proposal with no provenance is one they have to re-derive.

## 4. File it only when asked

`convert_brainstorm_session`, same arguments, when the person has said to file
it outright rather than review it. It writes the backlog item, carries the
session's gathered entities onto it, and the session then reads as converted.
Safe to repeat — a session that already produced an item answers with that
item rather than filing a second.

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
