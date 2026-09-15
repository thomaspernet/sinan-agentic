---
name: propagation-scan
description: Find the other sites a landed change should have been made at, and file one issue per site with an umbrella per mechanical sweep — once a person approves the list, or at once as a stage of a run.
family: analysis
shipped-from: e1a54249e4b517031f1687c3c3192e140f72e3c2915346aee41f8b1aaad58bf4
---
Find the other places a landed change should have been made, and file each one.

## 1. Read the change

The whole change, once. As a stage of a run it is the run's finished diff,
pinned: the *Pinned diff* line of the launch's *The run you were launched for*
block names it as two commits, where the branch the run delivers left the
development branch and where every member had landed. The stage runs beside
the run's pull request in a copy of the code checked out at the second, so read
the diff there as `git diff <base>..HEAD` — never the delivered branch against
the development branch by name, which reads as no change at all once the merge
beside this stage lands. Read the issues it closes — the epic and each member —
with `gh issue view` for the ask the diff was answering. On demand it is the
change the person names. You are looking for what the change *introduced*,
not what it fixed: a helper that now exists, a pattern now established, a
performance fix now proven, or the shape of the bug it removed.

A change that introduced none of those — a copy edit, a dependency bump, a
one-off local fix — has nothing to propagate. Say so and stop; a scan that
invents a pattern to have something to file wastes everyone downstream.

Read the quality gate's verdict too, wherever one is in front of you — a
claimed item's `verdict`, or a gate's findings the launch or the issues carry.
A gate that passed over a sibling site it was forbidden to edit names the
shared definition and every site it swept up in its `reasoning`: those
candidates are already located and already argued, so start from them rather
than making a person restate them. Confirm each still stands in the tree — the
change has landed since — and then carry on with the search for the ones the
gate never reached.

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

## 4. Decide what gathers under one umbrella

Sort the candidates on the reading `propagation-consolidate` applies. Sites
that share one contract and take one mechanical change — one shared definition
adopted at each — are a sweep, and a sweep of two or more sites is gathered
under one umbrella epic, because the chain runs a pass as an epic with members
rather than as loose issues. A recurring bug shape is not a sweep: each of its
sites needs its own reading, so each is filed on its own with no umbrella. So
is a sweep that found a single site.

## 5. Present, unless a run is the approval

Invoked on demand, show the candidate sites with the contract each one matches
and the grouping you would file, and stop. Filing turns a scan into other
people's work; the person who asked decides how much of it to create, and file
only what they approve.

As a stage of a run — the launch carries a *The run you were launched for*
block — nobody is there to answer, and the run is the approval: the run chose
to scan, so file without presenting and without waiting.

## 6. File

One issue per site: `gh issue create` for each, naming the file, the line
range, the contract it matches, and the change that surfaced it.

Then mint one umbrella per sweep by following the `mint-umbrella-epic` skill,
with that sweep's issues as its members. Its body names the change that
surfaced the sweep — the issue whose diff it was and the commits — so the
lineage survives the members linking to the umbrella instead. It carries the
`epic` label and every label its members share, so the backlog lists it as
work that can be run. Each member links `child-of` its umbrella, never the
issue whose diff surfaced it; a site filed on its own links `child-of` that
issue.

As a stage, record what the scan left behind before settling:
`worklist_record_scan` with `issues` every per-site issue you created and
`umbrellas` every epic you minted, each as `#N`. A scan that found nothing to
propagate calls it with `none_found` true instead. The record is what the run
reads the stage's outcome from — a stage settled `passed` with no record reads
as a scan that left nothing, whatever the settle says.

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

You are invoked on demand — by a person who already knows what they want — or
as one stage of a run, which runs you once for the run as a whole rather than
once per member. The two report back differently, so establish which before
doing anything.

Call `worklist_claim_item` with no arguments.

- `no_anchor`, and the launch carries a *The run you were launched for* block —
  you are a stage of that run, and the run is the unit: there is nothing to
  claim. Do the work above against the run's finished change, then settle with
  `worklist_set_stage_status`, which takes no uuid: `passed` when the stage did
  what it says, `failed` with a `halt_reason` when it could not.
- `no_anchor`, and the launch carries no such block — you were invoked on
  demand. There is no unit to settle: do the work above, then report what you
  produced to the person who asked, naming it by issue number or path so they
  can open it.
- `claimed: true` — you are one member's step of a run over a workflow someone
  wrote, which binds this skill to a step of its own. Do the work above against
  that member's change, read from the claimed item's `title` and
  `attachments`, then settle with `worklist_set_item_status` and the item's
  `item_uuid`: `passed` when the step did what it says, `failed` with a
  `halt_reason` when it did not, and `skipped` when the question no longer
  exists.
- `claimed: false` with `already_running` — another session has it. Stop.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on, not as an error string. Never leave a
claimed unit `running`: a step that stops without settling is
indistinguishable from one still in flight.
