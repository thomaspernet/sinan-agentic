---
description: "Run the workflow's acceptance scenarios as a single Playwright invocation."
capability: acceptance
---

Run every linked acceptance scenario for the current workflow and gate the
workflow PR on the result. The skill is launched by the workflow-scoped
``run-acceptance`` action that sits between ``quality`` and
``submit-workflow-pr``; it bundles every coordinate into one
``npx playwright test`` call when possible (#1390).

Boundary: this skill **runs** the acceptance scenarios — it does not
file bugs, edit issues, or open PRs. Use ``/new-bug --from-scenario``
for regression bugs and ``/submit-pr`` for the workflow PR.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill run-acceptance --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done. The mandatory-reads include the authoritative `acceptance-scenarios` doc — the contract this skill enforces as the workflow's pre-PR gate, including the title-stability invariant and the regression-vs-acceptance distinction.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

## Parse arguments

``$ARGUMENTS`` is one or more space-separated coordinates. Each
coordinate is the canonical ``<suite>::<file>::<title>`` triple the
catalogue and ``scenario_links`` table speak. The dispatcher passes the
**full** linked-scenario list on first run; on resume after a halt the
dispatcher passes only the previously-failed coordinates so the skill
re-runs only those.

```bash
COORDS=( $ARGUMENTS )
if [ "${#COORDS[@]}" -eq 0 ]; then
  echo "run-acceptance: no scenarios passed — gate is a no-op."
  exit 0
fi
```

## Pre-flight

```bash
command -v npx >/dev/null || {
  echo "npx not found on PATH. Install Node.js + Playwright in this repo."
  exit 1
}
```

The watched app must be running. The ``devwatch scenarios run`` CLI
probes the baseURL itself; if you know the app is down, start it first
(``/app-control open``) before re-running this skill.

## Run

The skill passes every coordinate verbatim to ``devwatch scenarios
run-many``. The CLI extracts each coordinate's ``<file>`` segment for
PW's spec-path argument list (deduped so PW sees each file once) and
keeps the full coordinate around so the per-scenario run row carries
``target_kind=scenario,target_id=<scenario id>`` — the only shape that
refreshes the dashboard pill and the acceptance gate's per-scenario
``last_run_status``/``last_run_commit_sha`` columns (#1848).

```bash
devwatch --repo "$REPO" scenarios run-many "${COORDS[@]}"
```

The CLI:

- Resolves every coordinate to its active scenario row via
  ``POST /api/scenarios/resolve-coordinate``. A coord that doesn't
  resolve hard-fails with a "re-sync the catalogue" message.
- Spawns one ``npx playwright test <spec1> <spec2> … --reporter=json``
  invocation. PW dedupes specs internally; running the same suite
  twice is a no-op.
- Records ``commit_sha = git rev-parse HEAD`` so the dispatcher's
  classifier can compare against the workflow branch's HEAD.
- Parses the PW JSON reporter, matches each per-test result back to
  its coordinate by ``(projectName, file, title)``, and POSTs one
  ``target_kind=scenario`` row per coord to ``/api/scenarios/runs`` so
  the dashboard pill and the gate's truth source share one write site.

The command exits ``0`` on a green run and non-zero on red.

## Outcomes

- **All passed** — the dispatcher's classifier sees every linked
  scenario as ``passed`` on HEAD and flips ``run-acceptance`` to
  ``done``. ``submit-workflow-pr`` becomes eligible.

- **One or more failed** — the dispatcher halts the workflow PR action
  chain with reason ``acceptance_scenarios_failed:<coord>,<coord>...``
  carrying the failed coordinates. Resume re-fires this skill with only
  those coordinates as arguments.

- **All passed but stale** — every link is green but at least one ran
  on a different SHA than the workflow branch's HEAD. The dispatcher
  halts with reason ``acceptance_scenarios_stale:<coord>...``; the user
  re-runs the gate to refresh the recorded SHA.

- **errored: app_not_running** — the baseURL probe failed. Start the
  watched app, then resume the workflow.

- **other errors** (npx missing, git rev-parse failure) — the CLI
  prints a one-line ``ClickException`` without a stack trace. Resolve
  the underlying issue and resume.
