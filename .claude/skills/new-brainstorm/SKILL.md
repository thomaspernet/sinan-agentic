---
name: new-brainstorm
description: Open a brainstorming session — the pre-issue thinking space — and write the summary it exists to hold.
family: writing
---
Open a brainstorming session — the space the thinking happens in before there
is an issue to file.

A session is a folder holding a summary and, on disk, a `README.md` with a
`mockups/` folder beside it. It is scratch: nothing here is committed, pushed
or staged, and no branch carries it. That is the point — it is the thinking
that produces the issues, not a change to the code.

## 1. Establish what is being thought about

The question or the pain that started this, and the rough shape of what it
might produce — one feature, an epic, or an idea that gets dropped. If none of
that can be said yet, there is nothing to open a session for; say so.

## 2. Name it

Short, plain, and about the topic rather than the conclusion — the name
becomes the folder on disk, so a name a reader would scan for is one they can
also find. Name it for what is being worked out, not for the answer you expect
to arrive at.

## 3. Open it

`open_brainstorm_session` with that name. Opening is safe to repeat: the same
name on the same day resolves the session already open rather than making a
second one, so a session you are returning to is reached the same way it was
started.

It answers with the session's `summary_page_uuid`, the `directory` it was
written into, and the `mockups_directory` beside it. Both paths are null when
the project has no single synced folder — the session is open, but nothing is
on disk and nothing can be written there. Report that rather than picking a
path of your own.

## 4. Write the summary

`update_page_content` against `summary_page_uuid`. Keep it a summary: what
triggered the session, what is still open, and what would have to be true for
it to converge. The README carries an index of the session's files under it,
so the summary points outward rather than holding everything — a session whose
summary tries to hold the whole thinking is one nobody rereads.

## 5. Report where it is

Name the directory so it can be opened. Then stop — a session is not an issue,
and the thinking is not finished the moment it is written down. Filing a
feature or a bug from it is a separate act, taken once the thinking converges.

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
