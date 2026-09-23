---
name: init-docs
description: Write or refresh the project's documentation pages, each naming the skills and rules the bank should bind it to.
family: writing
shipped-from: e4b9ac9519fb41fa6db30847a502c2d63e000c2a9baf8a215832a1a0f593a132
---
Write or refresh the project's documentation pages.

Documentation is what carries knowledge that cannot be recovered from the code —
why a thing is shaped as it is, how the parts connect, what the words mean. That
is what this writes; the code says the rest.

Documentation is a page in the app, not a file in the repository. Every doc
this writes is a documentation page, and nothing here touches the working tree:
run `git status --porcelain` when you start and again when you finish, and the
two readings are the same.

## 1. Read the repository first

Its layout, its entry points, its own README. A documentation set written before
reading the code describes a project that does not exist, and is worse than none
because the next reader believes it.

## 2. Read what the project already holds

Call `documentation_coverage`. It returns every documentation page of the
project with the `covers` globs it declares and whether it reads current, and
`never_written`: the directories holding changed code no page covers. Open a
page with `read` before deciding anything about it.

## 3. Decide what the set should hold

Three audiences, kept apart:

- The cross-project principles — how code is written here, regardless of the
  repository.
- This project's own shape — its architecture, its boundaries, the decisions
  behind them and what each cost.
- What the product does, for a reader who will never open the code.

Write only what this repository needs. A page describing a stack it does not use
is a page that will be wrong before anyone notices it was never right.

## 4. Write each page

Create it with `create_page`, `type_name` `documentation`, its body written as
markdown and named as a reader would look it up. Give it a `category`, the area it files under, and keep one
audience's pages out of another's category. Give it `covers`: the
repository-relative globs of the code it describes, narrow enough that a change
under them is one this page has to answer for. A page about a principle rather
than one part of the code covers the code that principle governs. The create is
refused without either, because a page covering nothing can never read stale.

Link docs to each other by wikilink, carrying the linked page's name exactly as
it is titled: `[[Status Model]]`. A concept is explained on one page and linked
from every other page that needs it, never explained twice. A wikilink resolves
by that name alone, so search the project's pages with `search` before naming a
page, both for the page a link should reach and for a name a page already
carries.

## 5. Say who each doc is for

Which skills and which rules need it, written as prose the reader can act on
rather than as a declaration the page expects something to act on. Nothing reads
a page to work out a reading list: a doc reaches an agent because it is in the
bank and somebody bound it there to the consumer that must read it, one consumer
at a time, and that binding is the whole of it.

So the audience you write down is what tells whoever binds the doc where it
goes. A doc nothing is bound to is a doc nothing will ever open.

## 6. Refresh rather than rewrite

On a second pass, read a page before changing it. A doc still accurate is left
alone — rewriting an accurate page to look busy is churn a reviewer has to read.
One the code has moved on from is rewritten with `update_page_content`.
One whose code was renamed or relocated keeps its body and has its globs
replaced with `update_documentation_covers`. A doc someone adapted is theirs;
report what diverged and recommend, rather than overwriting their edit.

`update_page_content` replaces the whole body with what it is given, as
markdown. Write it from the page's current body as `read` returns it, keeping
the headings, lists and tables it holds. Change what the change calls for and
leave every passage it does not touch exactly as it was: a house style (dash or
arrow substitutions, re-quoting, re-wrapping) is never applied across a page
the change did not otherwise affect.

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
