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

Once — and only once — the human approves, create the epic via the approval-gated CLI command. `--approve` is the machine-level assertion that the human signed off; the command refuses to create anything without it.

```bash
devwatch --repo "$REPO" mint-umbrella-epic \
  --title "<approved name>" \
  --body "<approved body>" \
  --area <backend|frontend|agents|infrastructure> \
  --priority <P0-critical|P1-high|P2-medium|P3-low> \
  --approve
```

This applies the `epic` label and creates **no branch** (#1116). The command prints the new epic number.

## 5. Hand off — do NOT re-parent here

This skill mints the epic and stops. Re-parenting the members onto it (turning the selection/originating issue + follow-ups into children of the new epic) and binding it as a workflow's root is the consumer flow's job — assemble (#2593), propagation attach (#2592), or growth re-parent (#2591) — not this primitive's. Report the new epic number and the members that should become its children, then stop.
