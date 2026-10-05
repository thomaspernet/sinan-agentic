---
name: mockup
description: Build a self-contained page that opens from a plain file link, and put it in the mockups folder of the session it belongs to — or change one there against the comments left on it.
family: writing
---
Build a mockup and put it in the one place mockups belong — the `mockups/`
folder of a brainstorming session — or change one already there against the
comments a person dropped on it.

A mockup is pre-issue design thinking: a page opened in a browser to feel out a
layout before any code exists. It stays scratch, beside the session that
produced it, and never reaches the repository.

`mockups/` is the one directory an agent writes into. No sync reads a file
there back into the graph, so nothing you put in it becomes a page — the app
lists the folder on the session and opens a file from it as a tab (#3406),
which is where a person sees what you built. The session's root belongs to the
mirror, which writes it out from what the app holds and has folder sync read a
file left there back as a note.

## 1. Resolve the session it belongs to

`open_brainstorm_session` with the topic's name — it opens the session or
resolves the one already open for today, and answers with `mockups_directory`,
the folder the page goes in. A null path means the project has declared no
brainstorm folder to write into: there is nowhere to put a mockup, so say that
and stop rather than choosing a directory. An `error` of `unresolved` means it
opened nothing at all and repeating the call will not change that — reach the
session with `list_brainstorm_sessions` and build against the folder it names,
or say so and stop.

## 2. Read the comments on a mockup you are changing

A person comments on a mockup from its tab (#3968), and those comments are the
brief for the next version: read them before changing a file, never after. A
new mockup has none, so skip to building it.

The comments live as ordinary comment threads on one note of the session,
`Feedback: <path>`, where the path is the file's inside `mockups/`. When the
mockup is the tab the person has focused, `get_workspace` names it — a `mockup`
tab — and its `mockup_feedback` carries that note's uuid and how many comments
are open. Otherwise find the note among the `notes` `read_brainstorm_session`
lists. No note, or a null `note_uuid`, means nobody has commented yet.

`read` the note: its `comment_threads` are the comments. Each open thread
(`resolved` false) quotes the element it was dropped on as its
`anchored_text`, and its `comments` say what to change there. A resolved one is
already settled; leave it. Keep each open thread's `thread_id` — answering it
is the last thing this skill does.

A comment you cannot act on without an answer — it is ambiguous, contradicts
another, or asks for something the discussion ruled out — is not guessed at.
Build without it, and ask in its thread once the file is written.

## 3. Build it

One self-contained HTML page. Everything inline — the styles in the document,
any script in the document, any image as data or as drawn markup. No link to
anything on a network, and no build step.

That is the whole constraint, and it is what makes a mockup worth having: it
opens from a plain file link, offline, in one click. A page that needs a server
or a network fetch to render is not a mockup, it is an application nobody asked
for yet.

Build the layout that was actually discussed, with every open comment you read
applied. With nothing specific to render, lay down the smallest honest frame and
say it is a starting point — a mockup full of invented content is a design
decision taken by accident.

Put a `data-anchor="<region>"` on each major region and each control the page
draws, named for what it is and unique on the page. It is the first thing a
comment's pin is re-found by once the file is rewritten; without one the pin
falls back to the element's role and accessible name, its place in the page and
its text, and a rewrite that rewords the element as it moves it loses the pin.
So keep a region's anchor when you change it, and give it a new one only when
it has become something else.

## 4. Write it into the folder

`<mockups_directory>/<name>.html`, with your editing tools. Name the file for
what it shows, so a session holding several is readable. Overwriting one of the
same name is fine — it is scratch, and a version worth keeping was worth its
own name. A mockup changed against its comments keeps its name: the comments
belong to that path, and a new name leaves them on a file nobody opens.

## 5. Have the session index it

Call `open_brainstorm_session` again with the same name. The README's index is
written from what the folder holds, so the row for the page you just wrote
appears on that pass — which is also what makes the mockup reachable from the
session rather than only from the path you happen to be holding.

## 6. Answer the comments

Every open thread you read gets an answer on the feedback note, so the person
sees on the mockup's tab what became of each one:

- applied — `resolve_page_comment` with the note's uuid and the thread's
  `thread_id`;
- needing an answer first — `reply_to_page_comment` with the question or the
  reason it was not done, and the thread left open for the person.

Resolve only what the file now shows. A thread resolved over a change that did
not land hides the one request still outstanding.

## 7. Report the path

Name the file so it can be opened directly, and say which comments it resolved
and which wait on an answer. Then stop. Iterating on the mockup in place is the
next thing worth doing; filing an issue from it is a separate act, once the
design converges.

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
  did what it says, `failed` with a `halt_reason` when it did not, and
  `skipped` when the question no longer exists. A step that decided its work
  fails says so with the reason, never with `passed`.
- `claimed: false` with `already_running` — another session has it. Stop.

A claimed step whose work stops for the person's decision — a proposal they
must approve, a choice only they can make — does not settle at the pause. Call
`worklist_set_item_status` with the `item_uuid`, `status: "waiting"` and a
`question`: one line saying what the person must decide. That settles nothing:
the step stays open and yours, and the run shows them the question. Once they
have answered, do what the answer asks, then settle — never `passed` at the
pause, which reads the step done before they have decided anything.

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

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on, not as an error string. Never leave a
claimed unit `running`: a step that stops without settling is
indistinguishable from one still in flight.
