---
description: "Run Playwright scenarios with /run-scenarios <target>."
capability: acceptance
---

Run Playwright tests in this repo and record the result in lingtai. The CLI
spawns Playwright, copies artifacts, and POSTs the run row through the local
lingtai server (the server is the single SQLite writer). This skill is a thin
wrapper that interprets the result and offers next steps.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill run-scenarios --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done. The mandatory-reads include the authoritative `acceptance-scenarios` doc — the contract for the `<suite>::<file>::<title>` coordinate this skill accepts as a target, plus the title-stability invariant.

## Parse arguments

Extract `TARGET` from `$ARGUMENTS`. Examples:

- `123` → numeric scenario id (the dashboard's per-row Run button always
  passes this shape — the catalogue already pinned the id at click
  time). Forward as `--id 123` so the CLI skips coord parsing entirely.
- `--spec e2e/login.spec.ts` → spec-scoped run (#1860). Runs every
  active scenario under that file as one Playwright invocation and
  records one `target_kind='scenario'` row per scenario id so each pill
  on the workflow Acceptance panel flips independently. Used by the
  panel's per-file "Run spec" button when ordering-dependent tests
  (test 2 needs test 1's setup) must execute together.
- `smoke::e2e/login.spec.ts::logs in` → coordinate `<suite>::<file>::<title>`.
  Operator-only escape hatch — use the id form when the dashboard
  surfaces one.
- `e2e/smoke-chat.spec.ts` → spec path positional, runs the whole file
  but records one `target_kind='suite'` row (operator escape hatch
  only). Prefer `--spec <path>` when you want per-pill refresh.

Optional flags forwarded to the CLI:

- `--base-url <url>` — override the baseURL probe (env: `PW_BASE_URL`).
- `--repo <owner/name>` — explicit repo, otherwise auto-detected.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command.

## Pre-flight

1. Confirm Playwright is installed in this repo:

   ```bash
   command -v npx >/dev/null || {
     echo "npx not found on PATH. Install Node.js, then run 'npm install' in this repo."
     exit 1
   }
   ```

2. Confirm the watched app is running. The CLI probes the baseURL itself
   and fails fast with `errored: app_not_running` if it can't reach the
   app. If you already know the app is down, start it first (e.g.
   `/app-control open` or the project-specific dev command).

## Run

A purely-numeric `$TARGET` is a scenario id; pass it through `--id` so
the CLI dispatches via the per-id service path without trying to parse
it as a coordinate. A `$TARGET` starting with `--spec ` (the workflow
Acceptance panel's "Run spec" button) is forwarded verbatim so the CLI
runs the whole spec and writes per-scenario rows. Anything else
(coordinate / bare spec path) goes through the positional form.

```bash
if [[ "$TARGET" =~ ^[0-9]+$ ]]; then
  devwatch --repo "$REPO" scenarios run --id "$TARGET"
elif [[ "$TARGET" == --spec\ * ]]; then
  # shellcheck disable=SC2086 — splitting --spec from its path is intentional.
  devwatch --repo "$REPO" scenarios run $TARGET
else
  devwatch --repo "$REPO" scenarios run "$TARGET"
fi
```

The CLI:

- Resolves `<target>` → Playwright args (spec path / `--grep` for single
  scenarios).
- Probes the baseURL.
- Acquires a per-suite file lock at `<repo>/.lingtai/scenarios-<hash>.lock`
  so concurrent runs of the same spec serialise. Different specs run in
  parallel.
- Spawns `npx playwright test … --reporter=json` with `CI=1` and the
  ambient `DEVWATCH_DATA_DIR`.
- Records `commit_sha = git rev-parse HEAD`.
- Copies `test-results/` → `~/.devwatch/artifacts/scenario-runs/<uuid>/`
  before returning.
- POSTs the result row to `/api/scenarios/runs`.

The command exits `0` on a green run and non-zero on red — propagate the
exit code so callers (e.g. `/check-code-quality`, `/run-issue`) can gate
on it.

## Read the result

The CLI prints a one-line summary on green:

```
  Scenario run #42: passed — e2e/smoke-chat.spec.ts (artifacts: ~/.devwatch/artifacts/scenario-runs/<uuid>)
```

On red, the summary line is followed by a `Failures:` block listing
`<file>:<line> — <title>: <error>` for each failed test. The artifact
dir contains PW traces / screenshots for deeper inspection.

If you need richer detail, reach for the catalogue surfaces:

```bash
devwatch --repo "$REPO" scenarios show <scenario_id>
```

## Outcomes and next steps

- **passed** — done. Continue with the caller's pipeline (e.g.
  `/check-code-quality` → `/submit-pr`).

- **failed** — the suite ran but at least one test was red. Decide:
  - **flaky** → re-run once: `devwatch --repo "$REPO" scenarios run "$TARGET"`.
  - **real regression** → file a bug. The follow-up flag
    `/new-bug --from-scenario "<coord>"` is tracked separately
    (#1385); until it lands, run `/new-bug` and paste the failure
    summary + artifact path manually.
  - **expected red while iterating** → fix the test or the code, then
    re-run.

- **errored: app_not_running** — the baseURL probe failed. Start the
  watched app (e.g. `/app-control open`), then re-run this skill.

- **other errors** (npx missing, git rev-parse failure, timeout) — the
  CLI prints a one-line ClickException without a stack trace. Resolve
  the underlying issue and re-run.

## Boundary

This skill runs scenarios. It does not file bugs, open PRs, or fix
broken tests. Use the dedicated skills for those (`/new-bug`,
`/submit-pr`).
