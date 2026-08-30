---
name: add-documentation
description: Update the documentation the landed diff made wrong — including prose naming a symbol the diff deleted or renamed.
family: writing
---
Update the documentation the landed work changed.

## 1. Read the diff that landed

The merge commit's diff when the epic merged through a pull request; the
integration branch against the development branch when it did not.

## 2. Find what the diff makes wrong

A doc is stale when the code it describes moved, not when it was last edited.
Look for: a new entity, endpoint or configuration section that no doc mentions;
a renamed or relocated symbol a doc still names; a described pattern the diff
changed; a stated count or exhaustive list a new call site just made wrong.

Grep the whole tree for every symbol the diff deleted or renamed — source,
markdown, and rule files alike. A clean type-check only proves nothing still
imports the old name; it says nothing about prose that still names it.

## 3. Update, do not restate

Docs reference each other and never duplicate each other. If a concept is
explained in one doc, link it from the second rather than explaining it again —
two copies of one fact means one of them is already wrong.

A doc the diff did not affect is left alone. Rewriting an accurate doc to look
busy is churn a reviewer has to read.

## Settling

You were launched for one stage of this run as a whole, not for one document,
so there is nothing to claim and nothing to look up. Every child of the run has
already settled by the time this stage starts; the work below acts on what they
landed.

Settle with `worklist_set_stage_status`, which takes no uuid — the run and the
stage rode in with the launch: `passed` when the stage did what it says;
`halted` with a `halt_reason` when it could not run at all; `manual_review`
when it ran but nothing can vouch for the result.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on — not as an error string. Never leave the
unit `running`: a step that stops without settling is indistinguishable from
one still in flight.
