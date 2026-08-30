---
name: Worklist Producer
description: Template for a Claude Code worklist writer step — claim the item the session was launched for, produce its output, record it, and settle its status.
family: writing
is_template: true
---
Run one worklist item end to end: claim it, produce its output, record it, and
settle its status. Copy this file as the starting point for a Claude Code skill
you bind to a worklist's writer step.

## 1. Claim the item you were launched for

Call `worklist_claim_item` with no arguments. The session already carries the
item and step it was assigned, so there is nothing to look up and nothing to
guess.

- `claimed: true` — the returned `item` is yours. Note its `item_uuid`; every
  step below takes it.
- `claimed: false` with `reason: "already_running"` — another session got there
  first. Stop. Do not produce anything.
- `claimed: false` with `reason: "no_anchor"` — this session was not launched
  for a worklist item. Stop and say so.

The claimed item carries `title`, its `input_page`, and its `attachments` —
the documents this run works on. Read them with the `read` tool; the item does
not inline their content.

## 2. Produce the output

Do the work the item asks for against its input page and attachments, then
write the result with `create_page` (or `update_page_content` when you are
refining a page that already exists).

## 3. Record what you produced

Call `worklist_add_output` with the item's `item_uuid` and the page uuid you
just wrote. This is the link the surface renders as the item's output.

## 4. Settle the status

You do not record a verdict on your own output. A verdict is the verification
gate's reading of what you produced, and `worklist_record_verdict` refuses a
producing session with `not_a_gate`. Where the step is gated, the gate runs
after you and records it; where it is not, the item settles with no verdict and
reads as done but unverified, which is exactly what it is.

Call `worklist_set_item_status` with the `item_uuid` and one of:

- `passed` — the step did what it says.
- `needs_retry` — it did not, and another attempt is worth making.
- `retry_exhausted` — it did not, and no further attempt is allowed;
  this item has already been attempted as many times as it may be, so a run
  that advances without review leaves it for a person rather than re-running
  it.
- `manual_review` — nothing vouched for the output; a human decides.
- `halted` — you could not produce an output at all: the inputs the item names
  are missing, or the ask is not answerable from what it has. Pass a
  `halt_reason` saying what stopped you, in words the person picking it up can
  act on. Do not reach for this because you doubt an output you did produce —
  that is `manual_review`.

An item is not finished until its status is settled. Leaving it `running`
strands it: the surface shows a run in flight that nothing will ever complete.
