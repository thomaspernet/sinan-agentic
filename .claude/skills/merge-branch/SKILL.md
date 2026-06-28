---
description: "Merge a branch into another. With --pr, merges a GitHub PR; without --pr, performs a local branch merge with conflict auto-resolution gated by tests/lint/typecheck."
capability: core
---

Land branch `<head>` into branch `<base>`. One skill, two surfaces (#1453):

- **GitHub mode** (`--pr <num>` supplied): merge the PR via `gh pr merge`.
- **Local mode** (no `--pr`): `git fetch && git checkout <base> && git pull && git merge --no-ff <head>`. On conflict, attempt auto-resolution and refuse to push unless project tests / typecheck / lint pass.

The base branch is supplied by the caller. Do not ask the user for it.

## Parse arguments

`$ARGUMENTS` shape:

```
<head> --into <base> [--pr <num>] [--repo <owner/name>] [--issue <num>] [--run <id>] [--delete-source] [--auto-approve]
```

Examples:

- `feat/901-foo --into feat/900-epic --repo o/r --issue 901 --run 5` -> local merge, child #901 into the epic branch.
- `feat/900-epic --into local-dev-next --pr 123 --repo o/r --run 7` -> close PR #123 with `gh pr merge`.
- `fix/42-bug --into local-dev-next --repo o/r --delete-source` -> local merge, then delete `fix/42-bug` on origin and locally.
- `feat/901-foo --into feat/900-epic --repo o/r --issue 901 --run 5 --auto-approve` -> local merge under auto-approve: a semantic conflict is attempted, not aborted (the owning workflow opted into auto-approve gates).

Extract `HEAD`, `BASE`, optional `PR_NUMBER`, `REPO`, `ISSUE`, `RUN_ID`. Set `DELETE_SOURCE=1` when `--delete-source` is present, otherwise leave it unset. Set `AUTO_APPROVE=1` when `--auto-approve` is present, otherwise leave it unset.

## Detect repo

Use `--repo` if supplied; otherwise read it from the working directory:

```bash
REPO="${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
```

Pass `--repo "$REPO"` to every `devwatch` and `gh` command.

## Mode A — GitHub PR merge (`PR_NUMBER` supplied)

1. Read the PR: `gh pr view "$PR_NUMBER" --repo "$REPO" --json title,baseRefName,headRefName,state,statusCheckRollup,mergeable`.
2. Verify:
   - PR is **open** (not already merged or closed).
   - CI checks are **passing** (warn if failing or pending; refuse on failing).
   - PR is **mergeable** (no conflicts).
3. Merge (append `--delete-branch` when `DELETE_SOURCE=1` so `gh` cleans up the source ref on origin):

   ```bash
   gh pr merge "$PR_NUMBER" --repo "$REPO" --merge ${DELETE_SOURCE:+--delete-branch}
   ```

4. Refresh the poller and pull the dev branch locally:

   ```bash
   devwatch --repo "$REPO" poll-prs
   DEV="$(devwatch --repo "$REPO" branches dev)"
   git checkout "$DEV" && git pull
   ```

5. Record completion:

   ```bash
   devwatch --repo "$REPO" agent-update \
     --run-id "$RUN_ID" \
     --status completed \
     --summary "Merged PR #$PR_NUMBER"
   ```

   Omit `--run-id` if no `RUN_ID` was parsed.

Skip Mode B entirely.

## Mode B — Local branch merge (no `PR_NUMBER`)

The merge target is `BASE`; the source is `HEAD`. Both come from the caller — never ask the user.

### 1. Prepare the working tree

```bash
git fetch origin
git checkout "$BASE"
git pull --ff-only origin "$BASE"
```

Refuse to continue if `git pull --ff-only` fails — the local base diverged from origin and a human must reconcile.

### 2. Merge with `--no-ff`

```bash
git merge --no-ff --no-edit "origin/$HEAD"
```

If the merge commits cleanly, jump to **step 5 (verify and push)**.

### 3. Conflict auto-resolution (only when step 2 reports conflicts)

Inspect every conflicted file via `git status --porcelain` (lines starting with `UU`, `AA`, etc.). For each:

- **Trivial conflicts** (non-overlapping import/export edits, lockfile regeneration, version bumps with no semantic clash, formatting-only changes) — resolve by inspection. Use `git checkout --ours`, `git checkout --theirs`, or hand-edit the file as appropriate. Do **not** blindly accept one side; read both halves of the conflict marker before choosing.
- **Lockfile / generated files** (`uv.lock`, `package-lock.json`, `Cargo.lock`, generated migrations) — accept the incoming version (`--theirs`) and regenerate from scratch:

  ```bash
  uv lock      # for uv.lock
  npm install  # for package-lock.json
  ```

- **Semantic conflicts** — overlapping logic edits, conflicting test changes, anything that requires understanding what the user intended in both branches:
  - **Default (no `--auto-approve`)**: **stop**. Run `git merge --abort` and proceed to **step 6 (halt)** — a human reconciles the intent of both branches.
  - **Under `AUTO_APPROVE=1`**: do **not** abort. Attempt the resolution — read both halves of every conflict marker and reconcile the logic so the intent of *both* branches survives (not a blind `--ours` / `--theirs`), then `git add` each resolved file. You own this decision for this merge; the owning workflow opted into auto-approve gates. The post-merge tests/typecheck/lint gate (**step 4**) is the objective safety net: if your reconciliation is wrong the gate goes red, the merge is discarded, and the run halts — a wrong guess is caught, never pushed. Never weaken, skip, or work around step 4 to force a conflicted merge through. If a conflict is genuinely beyond reconciliation (you cannot produce a tree that preserves both intents), fall through to **step 6 (halt)** even under `--auto-approve`.

After every resolution, `git add` the affected file. Once `git status` shows no remaining conflicts, finalise the merge:

```bash
git commit --no-edit
```

### 4. Sanity check the merged tree

Before pushing, prove the merge is buildable. Detect the project shape and run the relevant checks:

- Python project (any `pyproject.toml`): `uv run pytest -q --ignore=tests/test_terminal.py` (or the project's documented test command).
- Node / TS project (`package.json` with a `test` script): `npm test`.
- Typecheck: `uv run mypy .`, `npm run typecheck`, or the project's documented command.
- Lint: `uv run ruff check .`, `npm run lint`, or the project's documented command.

Run **every** test/typecheck/lint command the project advertises. If any fail, **do not push**. Run `git reset --hard "origin/$BASE"` to discard the merge and proceed to **step 6 (halt)** with the failure summary.

### 5. Push

```bash
git push origin "$BASE"
```

When `DELETE_SOURCE=1`, clean up the source branch on origin and locally. This is best-effort — the merge already succeeded and is the load-bearing operation, so a cleanup failure (already deleted, protected branch, no local copy) logs a warning and does not fail the run:

```bash
if [ "$DELETE_SOURCE" = "1" ]; then
  git push origin --delete "$HEAD" || echo "warn: failed to delete origin/$HEAD (already gone or protected)"
  git branch -D "$HEAD" 2>/dev/null || true
fi
```

This skill never closes the child issue. Pre-#2247 a `--close-issue` flag (#1638) closed the issue here to satisfy `delete-branch`'s old "issue must be closed" gate; that gate now accepts a completed `merge-to-base` row as proof of integration, so the child issue stays open until the epic-level PR ships and GitHub auto-closes it.

Record completion:

```bash
devwatch --repo "$REPO" agent-update \
  --run-id "$RUN_ID" \
  --status completed \
  --summary "Merged $HEAD into $BASE"
```

Omit `--run-id` if no `RUN_ID` was parsed.

### 6. Halt on unresolved conflict / failed gate

When auto-resolution leaves the tree broken, when the post-merge gate (tests / typecheck / lint) fails, or — **without `--auto-approve`** — when conflicts are semantically ambiguous:

Under `AUTO_APPROVE=1` a semantic conflict is *attempted* in step 3 rather than halted on sight, so this halt is reached only when the post-merge gate fails or the conflict is genuinely beyond reconciliation (no tree preserves both intents). The gate never softens — auto-approve never pushes a red tree.

```bash
git merge --abort 2>/dev/null || git reset --hard "origin/$BASE"
devwatch --repo "$REPO" agent-update \
  --run-id "$RUN_ID" \
  --status needs_human \
  --summary "merge_conflict: $HEAD into $BASE — <one-line cause>"
```

Surface a structured `merge_conflict` reason so the dashboard halts the run with a specific cause and a human can resolve and resume.

## Boundary

This skill merges. It does not release, file follow-up issues, or update docs. Tell the user to run `/release` (when the merge ships a workflow PR) or to trigger the next workflow action manually for the local-mode case.
