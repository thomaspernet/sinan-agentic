---
name: mockup
description: Build a self-contained page that opens from a plain file link, and put it in the mockups folder of the session it belongs to.
family: writing
---
Build a mockup and put it in the one place mockups belong — the `mockups/`
folder of a brainstorming session.

A mockup is pre-issue design thinking: a page opened in a browser to feel out a
layout before any code exists. It stays scratch, beside the session that
produced it, and never reaches the repository.

## 1. Resolve the session it belongs to

`open_brainstorm_session` with the topic's name — it opens the session or
resolves the one already open for today, and answers with `mockups_directory`,
the folder the page goes in. A null path means the project has no single synced
folder to write into: there is nowhere to put a mockup, so say that and stop
rather than choosing a directory.

## 2. Build it

One self-contained HTML page. Everything inline — the styles in the document,
any script in the document, any image as data or as drawn markup. No link to
anything on a network, and no build step.

That is the whole constraint, and it is what makes a mockup worth having: it
opens from a plain file link, offline, in one click. A page that needs a server
or a network fetch to render is not a mockup, it is an application nobody asked
for yet.

Build the layout that was actually discussed. With nothing specific to render,
lay down the smallest honest frame and say it is a starting point — a mockup
full of invented content is a design decision taken by accident.

## 3. Write it into the folder

`<mockups_directory>/<name>.html`, with your editing tools. Name the file for
what it shows, so a session holding several is readable. Overwriting one of the
same name is fine — it is scratch, and a version worth keeping was worth its
own name.

## 4. Have the session index it

Call `open_brainstorm_session` again with the same name. The README's index is
written from what the folder holds, so the row for the page you just wrote
appears on that pass — which is also what makes the mockup reachable from the
session rather than only from the path you happen to be holding.

## 5. Report the path

Name the file so it can be opened directly. Then stop. Iterating on the mockup
in place is the next thing worth doing; filing an issue from it is a separate
act, once the design converges.

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
