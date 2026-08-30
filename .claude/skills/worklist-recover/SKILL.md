---
name: Worklist Recover
description: Recover a halted worklist item on a run that advances without review — read the halt, fix the output, and settle the item.
family: delivery
---
A worklist item has halted and its run is configured to advance without
waiting for a human. Your job is to get that one item moving again, or to
establish that it genuinely needs a person.

You were launched for the halted item, so you do not need to search for it.

## 1. Read the halt

Call `worklist_claim_item` with no arguments.

- `claimed: true` — the item is yours to recover.
- `claimed: false` with `reason: "already_running"` — something else is already
  working on it. Stop.
- `claimed: false` with `reason: "no_anchor"` — no item was named. Stop and say
  so.

The claimed item carries two things that say why it stopped, and which one is
set tells you what kind of halt you are answering.

- `halt_reason` set — the run stopped before it could produce anything, and
  this is the reason it recorded. There is no output to judge; your job is to
  make the item runnable, or to establish that it cannot be.
- `verdict` set — an output exists and was judged. Its `reasoning` is the
  diagnosis you are answering.
- neither set — an output exists and nothing vouched for it. That absence is
  itself the diagnosis, and the judgement is yours to make.

Do not read `status` for any of this: claiming the item set it to `running`,
so it no longer records the halt. Both fields survive the claim for exactly
that reason.

Either way, read the item's `outputs` and its `attachments` with the `read`
tool to see what was actually produced against what was asked.

## 2. Decide what halted it

Establish which of these it is before changing anything:

- The output is wrong or incomplete against the item's own ask.
- The output answers the ask — either the verdict that rejected it was wrong,
  or no verdict was ever recorded for it.
- The item cannot be completed as specified — its inputs are missing,
  contradictory, or the ask is not answerable from what it has.

## 3. Act on that decision

You settle the item; you do not judge it. A verdict is the verification gate's
reading of an output, and `worklist_record_verdict` refuses this session with
`not_a_gate` — a recovery that both repaired the work and passed it would leave
the same self-certified record the halt came out of. Do not call it: settling
is the whole of what this session records, and an item settled with no verdict
reads as done but unverified, which is exactly what a recovered item is.

`worklist_set_item_status` takes the `item_uuid` and one status. These end a
recovery, and every branch below settles on one of them:

- `passed` — the step now does what it says.
- `manual_review` — an output exists and nothing can vouch for it, so a person
  decides.
- `halted` — there is no output and none can be produced from what the item
  has. Pass a `halt_reason` saying what stopped you, in words the person
  picking it up can act on.

The call accepts other values, and none of them ends a recovery. `to_do` and
`running` settle nothing at all — leaving the item in either strands it exactly
as the halt did. `needs_retry` and `retry_exhausted` are the gate's rejections:
each says a verdict failed, and this session records no verdict. `cancelled`
says the question no longer exists — a person closed it, or the run closed
before the step ran — and a recovery is answering the question, so it never
settles one cancelled. Those, with the ones above, are every status there is;
anything else is refused and records nothing.

- **Output wrong** — fix it. Rewrite the produced page (or write a fresh one
  and record it with `worklist_add_output`), then settle `passed`. Nothing
  re-judges the repair — the item carries whatever verdict it already had, and
  the settle is this session's own reading of the work it just did.
- **Output fine** — settle `passed`. If the verdict that rejected it was wrong,
  say so in the session's own report; the standing verdict is the gate's to
  overturn, not this session's.
- **Not completable** — do not invent an output. If the item produced
  something and it simply cannot be judged, settle `manual_review`. If it
  produced nothing and cannot — its inputs are missing, or the ask is not
  answerable from what it has — settle `halted` with a `halt_reason` naming
  the blocker, which is where an item with no output to judge belongs. Either
  way a human picks it up with the reason intact.

Settle the item exactly once. Do not re-run the whole framework, do not touch
any other item, and never leave the item `running` — a recovery that halts
without settling is indistinguishable from the halt it was sent to fix.
