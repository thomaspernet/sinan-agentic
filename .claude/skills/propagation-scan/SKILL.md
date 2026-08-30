---
name: propagation-scan
description: Find the other sites a landed change should have been made at, and file one issue per site once a person approves the list.
family: analysis
---
Find the other places a landed change should have been made, and file each one.

## 1. Read the change

The diff the issue's branch landed, and the issue it closed — `gh issue view`
for the ask the diff was answering. You are looking for what the change
*introduced*, not what it fixed: a helper that now exists, a pattern now
established, a performance fix now proven, or the shape of the bug it removed.

A change that introduced none of those — a copy edit, a dependency bump, a
one-off local fix — has nothing to propagate. Say so and stop; a scan that
invents a pattern to have something to file wastes everyone downstream.

## 2. Name the pattern, then find its other sites

For each thing the change introduced, write down the contract in one line — what
the shared definition guarantees — then search the tree for every other place
carrying that *same* contract.

Search by more than the literal text. An independently written copy can reach an
identical contract through different source, so search by the shape of the
transform and by the name of the concept as well, and confirm a candidate by
what it does rather than by whether it reads the same. Search for existing
consumers of the new shared definition too: a site that already adopted half of
it will never match a search for what it replaced.

## 3. Keep out what merely looks similar

A site is in scope only when its contract is identical. A different merge key, a
different data shape, a streaming form against a one-shot one, a different
threshold — each is its own pattern, and folding one in is the drive-by change
this scan exists to avoid, not the work it exists to do. What is *not* a
difference: which caller triggers it, how many items it happens to handle today,
and whether it collects a result or only raises.

## 4. Present before filing

Show the candidate sites with the contract each one matches, and stop. Filing
turns a scan into other people's work; a person decides how much of it to
create.

## 5. File what was approved

One issue per approved site: `gh issue create` for each, naming the file, the
line range, the contract it matches, and the change that surfaced it. When
several sites share one mechanical sweep, say so in each — that is what lets
them be gathered afterwards instead of run one at a time.

Never edit code here. A scan that fixes what it finds leaves nothing to review
the fix.

## Writing for GitHub

Anything written onto an issue is public, permanent, and read months later by
someone with no knowledge of the run that produced it. Write for that reader:
third person, present tense, naming the change rather than the process that
produced it. No run identifiers, no internal phase names, no first-person
agent voice, no real names or addresses — a role (`the reporter`, `the
reviewer`) says everything the reader needs.

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
