---
name: prepare-backlog-step
description: Prepare a backlog step to run — bind the library skill that executes it, or draft and propose one — and report what still holds it.
family: planning
---
Prepare one backlog step to run: bind it to the skill that executes it — one
the library already holds, or one you draft and propose — then say whether the
step is ready and what still holds it.

A backlog step sits between filing and running. It belongs to a backlog item,
not to a run, so it is not a worklist item: nothing here claims it, and the
claim the reporting contract below makes answers `no_anchor` when you were
asked to prepare a step directly. The step is the uuid you were given.

## 1. Read the step and its item

`read` with the step's uuid answers the backlog item holding it, with every
step under `sub_items`. Find yours and note its `title` and `description`, the
skill it is bound to (`skill_uuid`, `skill_name`), its gate, its
`argument_values` and `ready`. A step that is already
ready needs nothing from you: report its readiness and stop, rather than
replacing a binding someone chose.

## 2. Read where the step came from

The item's `brainstorm_session_uuid` names the session it was filed from. Read
it with `read_brainstorm_session`, then `read` the notes the step's title and
description trace to, and the entities the item carries under `attachments`.
The skill you choose or write does what that thinking asked of the step; a
skill written from the title alone does what the title happens to suggest. An
item filed directly has no session, and its own description is all there is.

## 3. Choose the skill

Look in the library first: `list_skills` for the project, then `get_skill` on
each candidate whose description fits the step. A skill that already does this
work is bound, not rewritten — skip to step 5.

When none does, write one. Take the closest existing skill as the model —
`get_skill` for its body — and follow its shape: how it opens, how it orders
its steps, how it reports back. Write the procedure for this kind of step
rather than for this one step, so the next step like it can bind the same
skill. Name it in lower-kebab case, for what it does.

## 4. Propose the skill you wrote

`propose_library_change` with `kind: "skill"`, the `name`, a one-line
`description` — what a later dispatch reads to decide whether to open it — the
`body`, a `rationale` naming the step it serves and the notes behind it, and
the item's `project_uuid`. Name the rules that govern it as `GOVERNS` edges
when the thinking names any.

Nothing is written into the library: a skill governs every later run that opens
it, so a person accepts it first. Do not write it into `.claude/skills/`
yourself either — the library writes it out once it holds it. The step's
preparation panel on the Backlog lists your draft above the library's own
skills, and accepting it there files the skill and binds it to the step in one
act. Not safe to repeat: every call makes another proposal, so read the answer
rather than calling again.

## 5. Bind a skill the library holds

`set_backlog_step_preparation` with the step's uuid and `skill_uuid`, plus
`argument_values` for what that skill declares it takes, when the thinking
answers them. An argument left unanswered never holds the step: it is
generated at launch. It refuses a skill the library does not hold, which is why a
draft is proposed rather than bound. A slot you leave out keeps its value, so
say only what you are deciding. Leave `model_tier` and `effort` unset unless the
thinking says otherwise: the skill's own default applies.

The gate is optional. Bind one with `gate_skill_uuid` only when a library skill
plainly checks this kind of result; otherwise leave it, and report the step as
ungated. A missing gate never holds a step.

## 6. Report the step's readiness

The item `set_backlog_step_preparation` answers carries the step's `ready` and
`argument_values` after the write; after a proposal, `read` the step again.
Report, for the step:

- the skill it is bound to, or the draft waiting for a person to accept on the
  step's preparation panel, with the proposal's uuid;
- the arguments it answered, as information: one left blank is generated at
  launch;
- the gate, or "no gate", as information rather than a blocker;
- whether it is `ready`, and if not, the one thing that holds it.

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
