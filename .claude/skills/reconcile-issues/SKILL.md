---
description: "Reconcile orphan issues that have no child-of: pick a parent for each, write the child-of link, and let convergence re-home them. Confirms before touching any issue with a run/branch, and reports misrouted members and workflow-less parents without touching them."
capability: core
---

Reconcile the orphan issues the **convergence engine deliberately does not auto-heal**
(#2951, epic #2948). Once convergence (#2949) shipped, the common case heals
itself — a pristine birth-draft re-homes the instant a `child-of`-to-root link
appears. This skill is the human-in-the-loop remainder:

- **No `child-of` at all** — Lingtai cannot infer the parent, so it asks you
  which parent each orphan belongs under, then writes the link.
- **Work already started** — re-homing could disturb an in-flight run, so it
  **confirms** before touching any issue that has a run/branch.
- **Backlog backfill** — running this skill over the full orphan list in one
  pass is the one-time sweep of issues orphaned before the convergence fix.
- **Misrouted members** (#3727) — a `child-of` that reaches a different workflow
  root than the one owning the step. Reported here, never touched: the link is
  already right, so the fix is a membership re-home, not another link.
- **Unrooted parents** (#3827) — a `child-of` that reaches no workflow root at
  all, because the declared parent is a label-only epic. Reported here, never
  touched: the link is already right too, and no second link helps — the parent
  needs a workflow rooted on it before anything can converge onto it.

This skill **reuses the membership service** — it lists candidates via
`devwatch attach-candidates` (backed by `attach_service`) and writes the
**single uniform operation** convergence projects from: a `child-of: #parent`
edge via `devwatch link`. It never re-implements attach, never starts a
workflow, and never cuts a branch.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` and `gh` command to ensure the correct
repo is targeted.

## 1. List the candidates

```bash
devwatch --repo "$REPO" attach-candidates --json
```

Each entry is an open, non-epic issue whose membership needs a decision. Fields:

- `number`, `title` — the issue.
- `disposition` — `pristine`, `work_started`, `misrouted`, or `unrooted_parent`
  (see below).
- `roots` — the candidate parents the link may point at: each is an **epic** or
  a **workflow root** (never a mere step — #1316). Picking one of these is what
  lets convergence re-home the orphan, because the chain reaches a workflow root.
  Empty for a `misrouted` or `unrooted_parent` entry — there is nothing to pick.
- `divergence` — present only on a `misrouted` entry: `link_root`,
  `workflow_root`, and the `reason` sentence naming both. `null` otherwise.
- `unrooted_parent` — present only on an `unrooted_parent` entry: `parent` and
  the `reason` sentence naming it. `null` otherwise.

Three shapes, and **only the orphan shape is this skill's job**:

- **Orphans** (`pristine`, `work_started`) — no `child-of` at all, so their only
  home is their own single-member self-rooted workflow and convergence has
  nothing to project membership from. `pristine` means no run and no branch, so
  convergence re-homes it the moment a link is written; `work_started` means a
  run/branch exists, so convergence will **not** auto-move it (the commit
  boundary holds). Continue to step 2 with these.
- **Misrouted** (`misrouted`) — the issue already has a `child-of`, but that
  chain reaches a different workflow root than the one owning its step (#3727):
  it renders under one epic and executes on another's integration branch.
  **Writing another link fixes nothing** — the link is already correct; it is
  membership that is wrong. Do not offer a parent for these and do not run
  `devwatch link` on them. Report them to the human, quoting `divergence.reason`
  verbatim, and say the fix is a re-home of the member onto the workflow rooted
  at `link_root` — an explicit operator action outside this skill.
- **Unrooted parent** (`unrooted_parent`) — the issue already has a `child-of`
  too, but that chain reaches **no** workflow root at all (#3827): the declared
  parent is a label-only epic, so convergence has nowhere to project the
  membership to and the issue re-roots its own draft forever. **Writing another
  link fixes nothing here either** — the link is already correct; what is
  missing is a workflow rooted on the parent. Do not offer a parent for these
  and do not run `devwatch link` on them. Report them to the human, quoting
  `unrooted_parent.reason` verbatim, and say the fix is to root a workflow on
  that parent (`devwatch regroup-onto-new-epic`, or the dashboard's workflow
  create) — an explicit operator action outside this skill. Once one exists,
  these converge on their own with no further link.

If the list is empty, report "Nothing to reconcile — every open issue has a home,
and every link agrees with it." and stop. (Run the plain
`devwatch --repo "$REPO" attach-candidates` without `--json` for a readable table
when reporting to the human.)

## 2. Present the orphans and ask for a parent

Show the orphans as a table: number, title, disposition, and the candidate
parents (`#N — title`). List any `misrouted` and `unrooted_parent` entries in a
separate, read-only section — they are reported, not placed. Then, for each
**orphan** the human wants to place, ask **which candidate parent** it belongs
under.

- Offer the orphan's `roots` as the choices. Picking one of them guarantees the
  re-home, because each is a valid workflow root.
- The human may name a parent **not** in `roots`. Only proceed if that parent is
  itself an epic or a workflow root — otherwise convergence will not re-home the
  orphan (the chain must reach a *root*, not a mere step). If no offered root
  fits, the right move is to create a workflow/epic for the missing branch (or
  root an existing workflow higher) **first** — do not write a link to a non-root.
- The human may skip any orphan. Skipping writes nothing.

## 3. Confirm before moving a work-started orphan

For any orphan whose `disposition` is `work_started` (it has a run or branch),
**confirm explicitly** before writing the link:

> #N has a run/branch in flight. Writing `child-of: #parent` records the
> relationship, but convergence will **not** auto-move committed work — the
> physical re-home stays an explicit operator action (detach in the dashboard's
> "Attach to…" flow). Write the link anyway? [y/N]

Only proceed on an explicit yes. A `pristine` orphan needs no such confirmation
beyond the parent choice — it has no in-flight work to disturb.

## 4. Write the link

The uniform operation, identical for every parent kind (epic, workflow root,
standalone root):

```bash
devwatch --repo "$REPO" link <orphan> <parent> --type child-of
```

This writes the `child-of` edge to the cache and mirrors it to the issue body's
`Links:` section on GitHub (the source of truth). Repeat per orphan the human
placed.

## 5. What happens next

You are done once the links are written — **do not** start a workflow, cut a
branch, or drive any pipeline step.

- **Pristine** orphans re-home automatically: the next server sync runs
  `_auto_attach_if_orphaned`, and convergence (#2949) retires the birth-draft
  and appends the issue onto the parent's workflow. Re-run
  `devwatch --repo "$REPO" attach-candidates` to confirm the list shrank.
- **Work-started** orphans keep their own workflow (the commit boundary). The
  link you wrote surfaces the relationship in the issue tree; the physical
  membership move is the operator's explicit follow-up in the dashboard.

## Boundary

This skill **lists candidates and writes `child-of` links for orphans** —
nothing else. It does not create workflows, cut branches, implement, force-move
committed work, or act on a `misrouted` / `unrooted_parent` entry beyond
reporting it. Membership lands via convergence on its own schedule.
