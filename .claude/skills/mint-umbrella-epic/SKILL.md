---
description: "Draft a fresh umbrella epic (name + body with a child checklist) from selected issues or propagation follow-ups, present it for human approval, and only on explicit approval create it on GitHub — rooting a draft workflow now or leaving it as a Backlog epic to convert later, as the human chooses. Never creates the epic unasked."
capability: core
---

Mint a **fresh umbrella epic** to group related work. The agent **drafts** the epic name + body; the human **approves** before anything is created on GitHub, and — for a selected set — **chooses whether the epic roots a workflow now or stays a Backlog epic to convert later**. This is the shared primitive every growth/assembly path uses (assemble, propagation attach, growth re-parent) — built once here (#2590, epic #2586).

Two hard rules, no exceptions:

1. **Drafting never creates the epic.** You produce a name + body and show it. You do **not** run `devwatch create-issue --epic` and you do **not** run `devwatch mint-umbrella-epic --approve` during drafting. Nothing lands on GitHub until the human says so.
2. **Always a fresh umbrella, never promote a worker.** When a working issue grows, you mint a *new* clean epic and the originating issue becomes a *child* of it. Never turn a working issue into both the worker and the container (locked decision, "Option B for growth").

The umbrella is **metadata only** — when finally created it carries the `epic` label and **no branch**. The branch is the workflow's job, not the epic's.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill mint-umbrella-epic --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` and `gh` command.

## Parse arguments — which input shape?

This skill drafts from exactly **one** of two documented input shapes:

- **Assemble** — a *set of selected issues* the operator wants to group. You summarise the selection into an umbrella name + body whose `## Children` checklist lists each selected issue. This shape has **two destinations** and the human picks between them at the §3 gate — see there.
- **Propagation / growth** — an *originating issue or diff* plus its *scan-hit follow-ups*. You summarise the originating change and the second-wave follow-ups into an umbrella whose checklist lists the originating issue (adopted as a child) and each follow-up.

Decide the shape from the invocation context. If it is ambiguous, ask the human which one before drafting — do not guess.

This skill always mints a **fresh** umbrella. Moving issues onto an epic that already exists is not this skill — that is `regroup-onto-existing-epic` (a whole group) or `rehome-member` (one issue). If the theme is already on file as an open epic, say so and stop rather than minting a duplicate.

## 1. Gather the inputs (read-only)

**Assemble shape.** For each selected issue, read its title (and skim its body for one line of context):

```bash
gh issue view <N> --repo "$REPO" --json number,title,state -q '{number,title,state}'
```

Keep `(number, one-line summary)` for each. This is read-only — file nothing.

**Propagation/growth shape.** Identify the originating issue (the founding fix that grew, or the diff under review) and collect the follow-up hits. Follow-ups may already be filed issues (propagation files per-site issues first) or not-yet-filed candidate sites. For each, keep a one-line summary and the filed issue number if it exists. Read-only — file nothing here either; this skill groups follow-ups, it does not discover or file them.

## 2. Draft the umbrella (no GitHub write)

Author a clean umbrella epic:

- **Name** — a short, accurate title for the grouping. Not the title of any one member; the *theme* that unites them.
- **Body** — one intro paragraph stating what the umbrella groups and why, then a `## Children` checklist with one `- [ ]` line per member (each prefixed with `#<N>` when the member is an already-filed issue).

The body must NOT contain a `child-of:` line (an epic is a root, never a child) and must NOT reference any `epic/...` branch (the epic owns no branch).

You may also produce the draft deterministically by piping the inputs through the primitive's draft phase — but its only effect is to normalise the same name + body; it makes no GitHub call either. The authored draft is the point of this skill; the human is approving *your* name + body.

## 3. Approval gate — present and STOP

Show the human the drafted name and body in full and ask them to approve, edit, or reject:

> Drafted umbrella epic — review before I create it on GitHub:
>
> **Title:** `<name>`
>
> **Body:**
> ```
> <body>
> ```
>
> Approve as-is, edit, or reject? I will not create anything until you approve.

- **Edit** → apply the human's edits to name/body, re-show, ask again.
- **Reject** → stop. Nothing was created. Report `status: skipped`.
- **Approve** → and only then, go to §4.

Do **not** proceed past this gate without an explicit approval in the conversation. There is no auto-approve for epic creation — minting the epic is the human's call (locked decision, "Agent never acts unasked").

**Assemble shape — ask the destination in the same breath.** A selected set can land two ways, and which one is the human's call, not yours:

> Where should it land?
>
> **(a) Workflow now** — the epic roots a `draft` workflow and the selected issues become its steps. Ready to start; the group leaves the Backlog.
>
> **(b) Backlog epic** — the epic carries the selected issues as `child-of` children and roots **no** workflow. It stays in the Backlog as a collapse/expand row, and you convert it yourself with **Convert to workflow** when you are ready.

Do not default to (a). "Bundle these under an epic" does not say which destination, so ask — the answer changes what lands and is not recoverable by re-running this skill. The propagation/growth shape has no such choice: it re-parents an *existing* workflow onto the new epic, so (a) is the only thing it can mean.

## 4. Create the umbrella (only after approval)

Once — and only once — the human approves, the approved name and body go to the `devwatch` command that matches the shape and — for assemble — the destination they chose:

| Input shape | Command | What lands |
|---|---|---|
| **Assemble → (a) workflow now** | `assemble-epic` (§5) | the epic, a `draft` workflow rooted on it, every selected issue moved onto it as a step |
| **Assemble → (b) backlog epic** | `mint-umbrella-epic` then one `link` per member (§5) | the epic and a `child-of` edge per member — **no** workflow |
| **Propagation / growth** — an originating issue plus its follow-ups | `attach-propagation-followups` (§5) | the epic, the originating issue's workflow re-parented onto it, and the follow-ups adopted as members |
| **Neither** — a theme placeholder with no members to bind | `mint-umbrella-epic` (§5) | the epic, and nothing else |

Every row creates the same epic through the same primitive — the `epic` label, **no branch** (#1116) — and every row prints the new epic number. What differs is which binding the members get, and whether a workflow is rooted on it now or later.

**Membership and grouping are two different bindings — take the one the human chose.** An issue is a *member* of a **workflow**, never of an epic, and every membership command refuses a destination that does not already root a non-terminal workflow. No `devwatch` command roots a workflow on an epic that already exists, so a label-only epic can never gain *members* from a terminal afterwards. If the group is meant to run as a workflow, take the `assemble-epic` row now — do not mint bare intending to attach members later.

A `child-of` edge is the **other** binding, and it is not a failed membership write. An epic with children and no workflow is the Backlog's **un-converted epic** — a supported, first-class row, not a dead end: the birth seed deliberately skips epics, so a minted umbrella owns no workflow; the Backlog groups its children by that `child-of` edge rather than by workflow membership; and **Convert to workflow** on that row seeds the new workflow's steps from those same edges, retiring each member's birth draft as it inserts. That is destination (b), and it is the whole point of offering the choice. What it is *not* is a back door onto an epic that was meant to have a workflow — the two destinations answer different questions, and the human picked between them in §3.

`--approve` is the machine-level assertion that the human signed off in §3. Every row refuses without it, so the approval gate holds whichever command carries the draft.

The approved name and the approved body are both drafted prose that name issues and symbols in backticks, so pass each through its own **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command. Mangling either one between the approval gate and the create defeats what the gate guarantees: the epic that lands must be the epic the human signed off on. Build them once — every row uses `"$TITLE"` and `"$BODY"`:

```bash
TITLE=$(cat <<'TITLE_EOF'
<approved name>
TITLE_EOF
)

BODY=$(cat <<'BODY_EOF'
<approved body>
BODY_EOF
)
```

## 5. Bind the members — finish it from this terminal

**There is no dashboard flow to hand off to.** The browser client ships the mint call with no screen that invokes it, and even invoked it would only mint — nothing there binds membership. Do not report the epic and wait for a UI to finish the job — there is no UI, and the commands below are the whole job.

Run the command the §4 row selected, reusing the `"$TITLE"` and `"$BODY"` already built.

### Assemble → (a) workflow now

One call. It mints the epic, roots a `draft` workflow on it, and moves **every** selected issue onto that workflow:

```bash
devwatch --repo "$REPO" assemble-epic \
  --title "$TITLE" \
  --body "$BODY" \
  --member <issue> \
  --member <issue> \
  --area <backend|frontend|agents|infrastructure> \
  --priority <P0-critical|P1-high|P2-medium|P3-low> \
  --approve
```

Pass the whole set — `--member` is repeatable and this door expects a selection spanning many source workflows, because every freshly filed issue is born into its own one-member draft. It resolves the base branch from the repo's own `dev` branch and the action set from the workflow defaults, which is exactly what those drafts carry.

Only an issue **still in its birth draft** is bundled. One that has already joined a real epic, or that has already executed (a branch, a run), owns a branch contract this door does not read, so it is left in place and named in the output. Move those separately: `rehome-member` for one issue, `regroup-onto-existing-epic` / `regroup-onto-new-epic` when a whole group is leaving one epic for another.

The epic lands unstarted: `draft` status, every autonomy toggle off, nothing queued.

### Assemble → (b) backlog epic

Two steps, because no single door mints-and-links. First mint the umbrella — it roots no workflow, since the birth seed skips epics:

```bash
devwatch --repo "$REPO" mint-umbrella-epic \
  --title "$TITLE" \
  --body "$BODY" \
  --area <backend|frontend|agents|infrastructure> \
  --priority <P0-critical|P1-high|P2-medium|P3-low> \
  --approve
```

Then write one `child-of` edge per member onto the epic number it printed:

```bash
devwatch --repo "$REPO" link <issue> <epic-number> --type child-of
```

Each `link` writes the edge and rewrites that issue's GitHub body `Links:` section. Members keep their own birth drafts, and nothing is started.

Unlike the single-call rows this one **can land half-done**: if a `link` fails, the epic still exists and the edges that already landed still hold. Do not re-mint. Report exactly which members are linked and which are not, with the `link` command to retry each — the epic number is the part that cannot be recreated.

The result is the Backlog's un-converted epic row, children nested under it. The human converts it when ready; **Convert to workflow** seeds the steps from these same edges.

### Neither — no members to bind

```bash
devwatch --repo "$REPO" mint-umbrella-epic \
  --title "$TITLE" \
  --body "$BODY" \
  --area <backend|frontend|agents|infrastructure> \
  --priority <P0-critical|P1-high|P2-medium|P3-low> \
  --approve
```

### Propagation / growth — an originating issue plus its follow-ups

One call. It mints the epic, re-parents the originating issue's workflow onto it, and adopts each follow-up as a member:

```bash
devwatch --repo "$REPO" attach-propagation-followups \
  --origin-issue <originating-issue> \
  --followup <follow-up> \
  --followup <follow-up> \
  --title "$TITLE" \
  --body "$BODY" \
  --area <backend|frontend|agents|infrastructure> \
  --priority <P0-critical|P1-high|P2-medium|P3-low> \
  --approve
```

### Then report

Nothing auto-runs on any row — no run is opened and the dispatcher is never fired, even where auto-execute is on. Report the new epic number, then what the chosen row actually bound: for a workflow row, the members that landed as **pending** steps; for the backlog row, the members now linked as children and the fact that the epic roots no workflow yet, so the human knows to convert it when ready.

Report — verbatim — anything the command named as left in place or not linked, together with the retry command it printed for each. An unbound member is the one part of the job left undone, and naming it is what lets the human finish it.
