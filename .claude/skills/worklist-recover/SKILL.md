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

- **Output wrong** — fix it. Correct the produced page with
  `edit_page_content`, or rewrite it with `update_page_content` (or write a
  fresh one with `create_page` and record it with `worklist_add_output`), its
  body as markdown; the item will settle `passed`.
  Nothing re-judges the repair — the item carries whatever verdict it already
  had, and the settle is this session's own reading of the work it just did.
- **Output fine** — the item will settle `passed`. If the verdict that
  rejected it was wrong, say so in the session's own report; the standing
  verdict is the gate's to overturn, not this session's.
- **Not completable** — do not invent an output. The item will settle `failed`
  with a `halt_reason` naming the blocker — its inputs are missing, the ask is
  not answerable from what it has, or the output it produced cannot be judged.
  A human picks it up with the reason intact.

Change some passages of an existing page with `edit_page_content`: each edit is
an `old` passage copied exactly from the markdown body `read` returns and the
`new` markdown that replaces it, and nothing else on the page is sent or
rewritten. It is the only write that reaches a page too large to send back
whole. Keep `update_page_content` for a full rewrite of a page small enough to
send whole: it replaces the whole body with what it is given, as markdown.
Write it from the page's current body as `read` returns it, keeping the
headings, lists and tables it holds. Either way, change what the change calls
for and leave every passage it does not touch exactly as it was: a house style
(dash or arrow substitutions, re-quoting, re-wrapping) is never applied across
a page the change did not otherwise affect.

## 4. Write the debrief a pass needs

A `passed` settle on the step that produces the item is refused with
`missing_debrief` until this attempt has recorded a debrief — its account of itself, one
page for a person reading the run who does not want to open each output. No
other output stands in for it, and a recovery that settles `passed` on an item
with none is refused like any other session. A `failed` settle needs none.

The claimed item's `debrief` says whether one exists. The session that halted
may have written it before it stopped; when it did and your repair changed
what was produced, bring it up to date with `edit_page_content` so it
accounts for the repair, held to the contract above, and record it again with
`worklist_add_output` — the recording is what makes it this attempt's, and one
an earlier attempt recorded does not answer the pass. When the item has none,
write one as markdown with `create_page`, `type_name: "debrief"`, in three
sections:

- **What a reader of the deliverable would notice** — the finding or the
  behaviour, in plain words.
- **What was produced and why it took that shape** — the pages the item
  carries, what the recovery changed, and the choices behind them.
- **What was weighed and left alone** — considerations not taken, follow-ups
  the item could not carry.

The page is filed against the item, run and step this session was launched
for, so there is nothing else to pass. Record it with `worklist_add_output`
like any other output.

## 5. Settle the item

You settle the item; you do not judge it. A verdict is the verification gate's
reading of an output, and `worklist_record_verdict` refuses this session with
`not_a_gate` — a recovery that both repaired the work and passed it would leave
the same self-certified record the halt came out of. Do not call it: settling
is the whole of what this session records, and an item settled with no verdict
reads as done but unverified, which is exactly what a recovered item is.

`worklist_set_item_status` takes the `item_uuid` and one status. These end a
recovery, and every branch above settles on one of them:

- `passed` — the step now does what it says.
- `failed` — there is no output and none can be produced from what the item
  has. Pass a `halt_reason` saying what stopped you, in words the person
  picking it up can act on.

The call accepts one other value, and it never ends a recovery: `skipped`
says the question no longer exists — a person closed it, or the run closed
before the step ran — and a recovery is answering the question.

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

`to_do` and `running` are not settlements — they are the run's own marks — and
are refused like anything else outside the three; a refusal records nothing.

Settle the item exactly once. Do not re-run the whole framework, do not touch
any other item, and never leave the item `running` — a recovery that halts
without settling is indistinguishable from the failure it was sent to fix.
