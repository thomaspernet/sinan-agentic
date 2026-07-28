---
description: "Draft a fresh umbrella epic (name + body with a child checklist) from selected issues or propagation follow-ups, present it for human approval, and only on explicit approval create it on GitHub. Never creates the epic unasked."
capability: core
---

Mint a **fresh umbrella epic** to group related work under one workflow. The agent **drafts** the epic name + body; the human **approves** before anything is created on GitHub. This is the shared primitive every growth/assembly path uses (assemble, propagation attach, growth re-parent) — built once here (#2590, epic #2586).

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

- **Assemble** — a *set of selected issues* the operator wants to start as one. You summarise the selection into an umbrella name + body whose `## Children` checklist lists each selected issue.
- **Propagation / growth** — an *originating issue or diff* plus its *scan-hit follow-ups*. You summarise the originating change and the second-wave follow-ups into an umbrella whose checklist lists the originating issue (adopted as a child) and each follow-up.

Decide the shape from the invocation context. If it is ambiguous, ask the human which one before drafting — do not guess.

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

## 4. Create the umbrella (only after approval)

Once — and only once — the human approves, the approved name and body go to **exactly one** `devwatch` command. Which one depends on whether the umbrella has members to bind:

| Input shape | Command | What lands |
|---|---|---|
| **Assemble** — a set of selected issues | `regroup-onto-new-epic` (§5) | the epic, a `draft` workflow rooted on it, and the seed member moved onto that workflow |
| **Propagation / growth** — an originating issue plus its follow-ups | `attach-propagation-followups` (§5) | the epic, the originating issue's workflow re-parented onto it, and the follow-ups adopted as members |
| **Neither** — a theme placeholder with no members to bind | `mint-umbrella-epic` (below) | the epic, and nothing else |

Every row creates the same epic through the same primitive — the `epic` label, **no branch** (#1116) — and every row prints the new epic number. What differs is whether a workflow ends up rooted on it, and only that decides whether issues can ever be its members.

**Pick the row before running anything: a bare create is a one-way door.** An issue is a member of a *workflow*, never of an epic, and every membership command refuses a destination that does not already root a non-terminal workflow. No `devwatch` command roots a workflow on an epic that already exists, so from a terminal a label-only epic can never be filled in later: a `child-of` edge pointing at it writes the link and stops there, leaving issues that read as children on GitHub and are members of nothing. If the umbrella has members, take one of the first two rows — do not mint bare first and look for a way to attach them afterwards.

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

For the third row only — an umbrella with no members to bind:

```bash
devwatch --repo "$REPO" mint-umbrella-epic \
  --title "$TITLE" \
  --body "$BODY" \
  --area <backend|frontend|agents|infrastructure> \
  --priority <P0-critical|P1-high|P2-medium|P3-low> \
  --approve
```

## 5. Bind the members — finish it from this terminal

**There is no dashboard flow to hand off to.** The browser client ships the mint call with no screen that invokes it, and even invoked it would only mint — nothing there binds membership. Do not report the epic and wait for a UI to finish the job — there is no UI, and the commands below are the whole job.

Run the command the §4 row selected, reusing the `"$TITLE"` and `"$BODY"` already built.

### Assemble — a set of selected issues

Seed the umbrella with **exactly one** member. That call mints the epic, roots a `draft` workflow on it, and moves the seed onto that workflow:

```bash
devwatch --repo "$REPO" regroup-onto-new-epic \
  --title "$TITLE" \
  --body "$BODY" \
  --member <seed-issue> \
  --area <backend|frontend|agents|infrastructure> \
  --priority <P0-critical|P1-high|P2-medium|P3-low> \
  --approve
```

Then move each remaining member onto the epic number that command printed, one call per member:

```bash
devwatch --repo "$REPO" rehome-member <issue> --to <epic-number>
```

**One seed, not the whole set.** `--member` is repeatable, but the command refuses a member set drawn from more than one source workflow: the new epic inherits that source's base branch and action set, and a mixed set answers no question it needs. Every freshly filed issue is born into its own single-member workflow, so any two separately filed issues always trip that guard. Seed-then-re-home is not a style preference — it is the form that converges for a set of separately filed issues. Seed with the member whose workflow carries the base branch and action set the whole group should ship with, because the new workflow inherits them from that one.

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

Members land as **pending** steps and nothing auto-runs — no run is opened and the dispatcher is never fired, even where auto-execute is on. Report the new epic number, the members that landed as steps, and — verbatim — anything the command named as left in place or not linked, together with the retry command it printed for each. An unmoved member is the one part of the job left undone, and naming it is what lets the human finish it.
