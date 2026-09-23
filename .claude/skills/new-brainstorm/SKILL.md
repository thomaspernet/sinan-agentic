---
name: new-brainstorm
description: Open a brainstorming session — the pre-issue thinking space — and write the notes and the summary it exists to hold.
family: writing
shipped-from: 10404494c7673992efe6f6443e2ac481c52144333019a5e8c7b033d150abf300
---
Open a brainstorming session — the space the thinking happens in before there
is an issue to file.

A session is a page holding a summary, with one note per sub-topic of the
thinking connected to it. On disk it is a `README.md`, a file per note beside
it, and a `mockups/` folder. It is scratch: nothing here is committed, pushed
or staged, and no branch carries it. That is the point — it is the thinking that produces
the issues, not a change to the code.

The directory belongs to the mirror. Every file at the session's root is
written out from what the app holds, and folder sync under the declared folder
reads one put there by hand back the other way — as one of the session's own
notes, named by the `# ` title you wrote at the top of it, on whatever pass
comes next rather than now. A file whose directory answers with no session is
left where it is and reported rather than read in as a page belonging to
nothing, so a write that cannot join the session does not quietly land beside
it. The call in step 5 writes the same note without the wait and without the
pass, which is why the thinking goes in through it.
`mockups/` is the one directory an agent writes into. No sync reads a file there
back into the graph, so nothing you put in it becomes a page; the app lists the
folder on the session and opens a file from it as a tab (#3406).

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
written into, and the `mockups_directory` beside it. A project that has already
declared a brainstorm folder answers with both paths on this first call, and
step 4 has nothing to do.

An `error` of `unresolved` is the one answer that opened nothing: whether a
session already exists for this name today could not be settled, so opening
would have made a second one for a topic that may already have had one. Do not
call it again — the answer will be the same. Reach the session with
`list_brainstorm_sessions`, which lists every session the project holds rather
than only today's, and work in the one it names. If it lists nothing, say so to
the person and stop: something the app reads is failing, and thinking written
anywhere now is thinking written twice.

## 4. Declare the folder, if the project has none

Both paths are null when the project has declared no brainstorm folder. The
session is open in the graph, but nothing is on disk — so the thinking is
written where nobody can open it, and no mockup can be put anywhere.

Ask the person which folder this project's sessions live in. Once, and with a
default they can accept in a word: `~/Brainstorms/<project name>`. Do not pick
one for them and do not derive one from a folder the project happens to
watch — it is where their thinking will live, and a session written under a
path nobody agreed to is a session they will not look in.

Declare their answer with `set_brainstorm_folder`, then call
`open_brainstorm_session` again with the same name. The second call resolves
the session already open rather than opening a second one, and this time it
answers with the `directory` and the `mockups_directory` — which is what says
the declaration took. A path that is not a directory on this machine is
refused: say which path was refused and ask again, rather than trying another
of your own.

## 5. Write a note per sub-topic

`add_brainstorm_note` against the session's uuid, once per part of the
thinking: `name` titles the note, and `content` is its body as markdown,
written in the same call. One note per sub-topic rather than one long one —
each is a separate thing a later conversion answers from, and each is written
out as a file of its own in that same call — there is no mirror pass to make
afterwards. `mirrored` says the file landed; `mirror_error` says which refusal
it was when it did not, and the note is in the session either way.

The notes are what the session's thinking *is*. Sub-topics written anywhere
else do not become any of it: a session read back holds its notes, its summary
and the entities gathered around it. A file left beside the README joins the
session on the next sync pass rather than never, but until that pass it is
thinking the conversion cannot see, and nothing tells you when the pass has
run.

## 6. Write the summary

`update_page_content` against `summary_page_uuid`, its body as markdown. It
writes the session out to disk as adding a note does, so the `README.md` holds
what you just wrote; the same is true of rewriting a note, against that note's
uuid. Keep it a summary: what
triggered the session, what is still open, what would have to be true for it
to converge — and a line per note saying what that note works out. The
generated index lists the notes by name alone, so the narration of what each
one holds is the summary's to carry, and it is what makes the index worth
following. A summary that tries to hold the whole thinking instead is one
nobody rereads.

Name each note as a `[[Note Name]]` wiki-link, spelled exactly as you titled it
in step 5. A link resolves by name into an edge to that note's page, so the
summary reaches its notes in the graph rather than only mentioning them — which
is what lets a reader arriving at the summary open the thinking it narrates,
and what a later conversion follows. A note named in plain prose reaches
nothing.

A session you return to already holds a summary.
`update_page_content` replaces the whole body with what it is given, as
markdown. Write it from the page's current body as `read` returns it, keeping
the headings, lists and tables it holds. Change what the change calls for and
leave every passage it does not touch exactly as it was: a house style (dash or
arrow substitutions, re-quoting, re-wrapping) is never applied across a page
the change did not otherwise affect.

## 7. Report where it is

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
  did what it says, `failed` with a `halt_reason` when it did not, and
  `skipped` when the question no longer exists. A step that decided its work
  fails says so with the reason, never with `passed`.
- `claimed: false` with `already_running` — another session has it. Stop.

A `skipped` settle also says what became of this step's work, as an `outcome`
with that outcome's evidence. A skip naming none is refused, and so is one
whose outcome has nothing behind it: you are the only one who knows, and a bare
skip leaves every reader after you guessing which of the three it was.

- `already_delivered` — the work is already done, in this repository or
  another. Give `references`, one per place it landed: a commit as
  `owner/name@sha`, a pull request or issue as `owner/name#123`. It is the one
  outcome that says something shipped, and a run reads it to know this step
  delivered even though nothing landed on its own branch.
- `left_out` — the run decided not to do this work. Give `outcome_reason`, one
  line saying why.
- `not_needed` — the question turned out not to exist. Give `outcome_reason`,
  one line saying why.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on, not as an error string. Never leave a
claimed unit `running`: a step that stops without settling is
indistinguishable from one still in flight.
