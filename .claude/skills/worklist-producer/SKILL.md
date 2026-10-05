---
name: Worklist Producer
description: Producer for a worklist step that runs once per member — claim its item, read its lineage, the run's story and its rules, produce, check it, record the output and the debrief, and settle.
family: writing
shipped-from: ba00aee36f06419f8df9f9e9889f34edc57f55d7ecb855a632170465ce562679
---
Run one worklist item end to end: claim it, read what came before it, produce
its output, check it, record it, and settle its status. It is bound to a step
that runs once per member of a run, and the step's own action and block say
what the output is. A step that runs once for the whole run binds Worklist Run
Stage instead.

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

## 2. Read what came before this attempt

The claim returns `lineage` beside the item: `attempts`, the earlier attempts
at each step of this member with the note each settled with and the pages it
wrote; `verdicts`, every verdict a gate recorded on it, oldest first, with its
findings; and `debriefs`, the accounts earlier attempts left. The item's own
`verdict` is the latest decision judging it. Read them before producing
anything. A `lineage` of `null` means the history could not be read, not that
there is none, so read the item's `verdict` and `debrief` on their own.

When the latest verdict failed, this attempt answers it rather than the
original ask alone: treat each finding as an instance of a class and clear the
class, not only the instance it quoted. Build on what an earlier attempt wrote
rather than starting over, unless a finding says its shape is the defect.

## 3. Read the run and the rules

Call `delivery_run_story` with no arguments. It reads the run this item belongs
to: the brainstorm the work was filed from, with the decisions it settled, and
the members beside this one with their debriefs and settle notes. A decision
the brainstorm settled is not this attempt's to reopen, and a member that
already answered part of the ask is built on rather than repeated.

The launch lists the rules this step is written under. A rule whose text the
launch carries is read there; one that arrives as headings or as a name only is
read with `read_rule`, opening the section the work needs by its `block_id`.
Read them before producing rather than after: a rule found at the settle is a
rewrite. The rules marked as applying before the settle are the ones a pass
attests against.

## 4. Produce the output

Do the work the item asks for against its input page and attachments, then
write the result as markdown with `create_page` (or, refining a page that
already exists, `edit_page_content` for the passages you change and
`update_page_content` for a rewrite of the whole page).

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

## 5. Check the output before recording it

Read what you wrote again against what was asked, as the gate would, and fix
what fails now: a unit settled with a known failing item is a round trip nobody
needed.

- **Answers the ask** — it does what the step asked, over the sources it
  names, and nothing the ask did not.
- **Grounded** — every claim traces to a source you read. A statement no
  source supports is sourced or removed, however plausible it reads.
- **Complete** — nothing the ask named is missing, and nothing is left as a
  placeholder or a note to come back to.
- **Consistent** — it agrees with itself, with the run's story, and with what
  the run's members already settled.
- **Rules** — each rule the launch listed holds for it.

Then ask where else the change belongs. An output has a family: a page that
restates what this one changed, a second copy of a figure it corrected, another
instance of a claim it fixed once. Search the project for them with `search`
and clear every one you find, or name the ones you leave in the debrief.
Everything you wrote this round is review surface on the same terms, including
the debrief.

## 6. Record what you produced

Call `worklist_add_output` with the item's `item_uuid` and the page uuid you
just wrote. This is the link the surface renders as the item's output. A step
whose work is files in a checkout rather than a page has nothing to record
here: its debrief names the files, and is all the step owes.

## 7. Write the debrief

What you produced is the item's deliverables; the debrief is its account of
itself — one page, whatever the deliverables were, for a person reading the
run who does not want to open each of them, and the page the gate reads before
it grades. A `passed` settle is refused with `missing_debrief` until
this attempt has recorded one, and no other output stands in for it.

Write it as markdown with `create_page`, `type_name: "debrief"`, in
three sections:

- **What a reader of the deliverable would notice** — the finding or the
  behaviour, in plain words.
- **What was produced and why it took that shape** — the pages you wrote, the
  findings of an earlier verdict this attempt cleared, and the choices behind
  them.
- **What was weighed and left alone** — considerations not taken, follow-ups
  the item could not carry, and what the family search looked for and cleared,
  so the gate can verify it rather than repeat it.

The page is filed against the item, run and step this session was launched
for, so there is nothing else to pass. Record it with `worklist_add_output`
like any other output.

## 8. Settle the status

You do not record a verdict on your own output. A verdict is the verification
gate's reading of what you produced, and `worklist_record_verdict` refuses a
producing session with `not_a_gate`. Where the step is gated, the gate
runs after you and records it; where it is not, the item settles with no
verdict and reads as done but unverified, which is exactly what it is.

When your work stops to wait for the person — a plan they must approve, a
choice only they can make — do not settle. Call `worklist_set_item_status`
with the `item_uuid`, `status: "waiting"` and a `question`: one line saying
what the person must decide. Never `passed` there: a pass reads the step done
before they have decided anything, and the run closes over a plan nobody
approved. `waiting` settles nothing — the step stays open and yours, the run
shows them your question, and your silence is not timed out while they think.
Once they have answered, do what the answer asks, then settle below.

Call `worklist_set_item_status` with the `item_uuid` and one of:

- `passed` — the step did what it says. An ungated pass reads as done but
  unverified on the run's own surfaces; nothing more is yours to record.
- `failed` — it did not: you could not produce the output, the inputs the
  item names are missing, or the ask is not answerable from what it has. Pass
  a `halt_reason` saying what stopped you, in words the person picking it up
  can act on.
- `skipped` — there is no output to produce, because the question this step
  answers no longer exists or its work is done elsewhere.

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

Give the settle a `note`: a few sentences in your own words for a person
reading the run later — what you found, what you chose, and what you left. It
is stored verbatim against this attempt, so write prose, not a status string.
It is not the `halt_reason`: the reason says what stopped the unit, the note
says what the work was.

A `passed` settle answers the rules the launch listed as applying before the
settle, with `attestations` — one entry per rule, each with its `rule_uuid`, a
`state` of `applied` or `not_applicable`, and a `note` of one sentence on how
this work meets it. A pass missing one is refused with `missing_attestation`,
records nothing, and hands the rule's text back; only a pass is held to it.

Your report is evidence beside the engine's own reading of the world: a pass
the world contradicts is outvoted, and a failing gate verdict routes the step
by policy whatever is written here.

An item is not finished until its status is settled. Leaving it `running`
strands it: the surface shows a run in flight that nothing will ever complete.
A `waiting` is not a settle: it holds the step open for the person's answer,
and the settle above still follows it.
