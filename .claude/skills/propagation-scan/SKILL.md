---
name: propagation-scan
description: Find the other sites a landed change should have been made at, and file one issue per site, gathering them under umbrella epics — once a person approves the list, or at once as a stage of a run.
family: analysis
shipped-from: b86c47a362cc334c32101ede842e152cd949e6f5217e76fe47bb92ba03b6c0d0
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
sites needs its own reading, so no site of one is ever folded into another.

Whether a group that is not a sweep of two or more also gets an umbrella
depends on who approved the scan. Invoked on demand it does not: a lone site,
and each site of a bug shape, is filed on its own under the issue whose diff
surfaced it, and the person who asked is there to pick it up. As a stage of a
run every group gets one — a sweep, a bug shape, and a group of a single site
alike — because that issue belongs to the epic the run is executing, and a run
seals its membership at its first run-scope stage: a child linked to it now is
never converged onto the run, never delivered by it, and read by nobody after
it closes. An umbrella of one member is work the backlog goes on offering.

Gathering is not collapsing. An umbrella holds one checklist line per site, so
a bug shape's sites keep their own issues and their own readings under it —
what the umbrella adds is a parent nothing has sealed.

## 5. Present, unless a run is the approval

Invoked on demand, show the candidate sites with the contract each one matches
and the grouping you would file, and stop. Filing turns a scan into other
people's work; the person who asked decides how much of it to create, and file
only what they approve.

As a stage of a run — the launch carries a *The run you were launched for*
block — nobody is there to answer, and the run is the approval: the run chose
to scan, so file without presenting and without waiting.

## 6. File

One issue per site: `file_issue` for each, naming the file, the line range,
the contract it matches, and the change that surfaced it.

Then mint one umbrella per group the step above gave one by following the
`mint-umbrella-epic` skill, with that group's issues as its members. Its body
names the change that surfaced the group — the issue whose diff it was and the
commits — so the lineage survives the members linking to the umbrella instead.
It carries the `epic` label and every label its members share, so the backlog
lists it as work that can be run. Each member links `child-of` its umbrella,
never the issue whose diff surfaced it.

Only a scan invoked on demand files a site under that issue directly, and only
for a group it left without an umbrella. As a stage there is no such group:
nothing this scan files links `child-of` the issue the run is executing.

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

Every issue write here goes through the app's issue tools — `file_issue`,
`edit_issue` and `delete_issue` — which put the write in the Backlog before
they return. In a session where those tools are not loaded, make the same
write with `gh` instead:
`gh issue create`, `gh issue edit`, `gh issue close` or `gh issue comment`.
A write made that way reaches the Backlog only on the repository's next
refresh, so say so when reporting it rather than reading its absence there as
a failure.

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

A claimed step whose work stops for the person's decision — a proposal they
must approve, a choice only they can make — does not settle at the pause. Call
`worklist_set_item_status` with the `item_uuid`, `status: "waiting"` and a
`question`: one line saying what the person must decide. That settles nothing:
the step stays open and yours, and the run shows them the question. Once they
have answered, do what the answer asks, then settle — never `passed` at the
pause, which reads the step done before they have decided anything.

Only the claimed-item branch settles `skipped` at all:
`worklist_set_stage_status` takes no outcome, and an on-demand invocation
settles nothing.

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
