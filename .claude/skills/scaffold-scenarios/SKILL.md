---
description: "Scaffold real Playwright specs from acceptance-scenario-draft entries on issue #$ARGUMENTS."
capability: acceptance
---

Scaffold a Playwright `test.skip(...)` stub for every
`acceptance-scenario-draft: "<title>"` line on an issue body so the
human can fill the test body in. The skill is the bridge from "I want
this covered" (a draft on the issue body) to "there is a real spec
file" (a `test.skip` stub committed in the watched repo). After the
human fleshes out the test body and removes `.skip`, the next
`devwatch scenarios sync` matches the new title against the catalogue
and binds the draft automatically (#1392 bind-on-sync).

Boundary: this skill **only** scaffolds spec files. It does not edit
issue bodies — bind-on-sync rewrites
`acceptance-scenario-draft: "<title>"` to `acceptance-scenario: <coord>`
once the real test exists. To add or remove draft lines on the issue
itself, use `/edit-acceptance-scenarios`.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill scaffold-scenarios --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done. The mandatory-reads include the authoritative `acceptance-scenarios` doc — the contract for the `Links:` section, the `<suite>::<file>::<title>` coordinate, the title-stability invariant, and the draft lifecycle this skill is one step of.

## Parse arguments

```bash
ISSUE_NUMBER="$ARGUMENTS"
if [ -z "$ISSUE_NUMBER" ]; then
  echo "Usage: /scaffold-scenarios <issue-number>"
  exit 1
fi
```

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

## Read the drafts

```bash
DRAFTS_JSON=$(devwatch scenarios drafts "$ISSUE_NUMBER")
DRAFT_COUNT=$(printf '%s' "$DRAFTS_JSON" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')
if [ "$DRAFT_COUNT" -eq 0 ]; then
  echo "No acceptance-scenario-draft entries on #$ISSUE_NUMBER — nothing to scaffold."
  exit 0
fi
```

Filter out drafts that are already bound — those have a real spec and
the body will be rewritten on the next sync; scaffolding them again
would only churn the working tree.

```bash
PENDING_JSON=$(printf '%s' "$DRAFTS_JSON" | python3 -c '
import json, sys
items = json.load(sys.stdin)
pending = [d for d in items if d.get("bound_scenario_id") is None and d.get("status") != "bound"]
json.dump(pending, sys.stdout)
')
PENDING_COUNT=$(printf '%s' "$PENDING_JSON" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')
if [ "$PENDING_COUNT" -eq 0 ]; then
  echo "Every draft on #$ISSUE_NUMBER is already bound — nothing to scaffold."
  exit 0
fi
```

## Scaffold each draft

For each pending draft:

1. Suggest a default spec path (`<slug>.spec.ts` directly under the
   configured suite cwd). The suite cwd lives in
   `~/.devwatch/projects/<project>.yaml` under `repos[].scenarios.cwd`
   and is the directory where `npx playwright test --list` is invoked,
   so file paths Playwright reports are already relative to it.
   Different watched repos place specs differently (some under `e2e/`,
   some under `tests/`, some directly under the suite cwd) — pick the
   default that matches your Playwright config and override per draft
   when needed.
2. Allow the user to override the path before any file is written.
3. Append a `test.skip("<title>", async ({ page }) => { /* TODO #N */ });`
   block to the chosen file, creating the file (with the right import)
   if it does not exist yet.

Use a Python one-liner to do the appends deterministically — slug
generation and idempotent import insertion are both fragile in pure
bash:

```bash
python3 - "$ISSUE_NUMBER" "$PENDING_JSON" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

issue = sys.argv[1]
pending = json.loads(sys.argv[2])

# Suite cwd resolution: read repos[0].scenarios.cwd from the active
# project YAML so the spec lands inside the right Playwright tree.
project_marker = Path.home() / ".devwatch" / "active_project"
project = project_marker.read_text().strip() if project_marker.is_file() else ""
project_cfg = Path.home() / ".devwatch" / "projects" / f"{project}.yaml" if project else None

suite_cwd = ""
if project_cfg and project_cfg.is_file():
    # Tiny YAML peek — we only need the first ``cwd:`` under
    # ``scenarios:``. Avoid pulling in PyYAML for one field.
    in_scenarios = False
    for line in project_cfg.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("scenarios:"):
            in_scenarios = True
            continue
        if in_scenarios and stripped.startswith("cwd:"):
            suite_cwd = stripped.split("cwd:", 1)[1].strip().strip('"').strip("'")
            break
        if in_scenarios and stripped and not line.startswith((" ", "\t")):
            break

base_dir = Path(suite_cwd) if suite_cwd else Path(".")

def slugify(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower())
    return s.strip("-") or "scenario"

import_line = "import { test } from '@playwright/test';\n"

scaffolded = []
for draft in pending:
    title = draft["title"]
    # Default to ``<suite_cwd>/<slug>.spec.ts``. The suite cwd is
    # already the directory Playwright runs from, so file paths
    # in the catalogue are relative to it; an extra ``e2e/`` segment
    # would mismatch repos whose Playwright config places specs
    # elsewhere (#1769).
    suggested = base_dir / f"{slugify(title)}.spec.ts"
    spec_path = Path(os.environ.get("SCAFFOLD_OVERRIDE_PATH") or suggested)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    if spec_path.exists():
        existing = spec_path.read_text(encoding="utf-8")
    else:
        existing = ""
    block = (
        f'\ntest.skip({json.dumps(title)}, async ({{ page }}) => {{\n'
        f'  /* TODO #{issue}: implement */\n'
        f'}});\n'
    )
    new_text = existing
    if "@playwright/test" not in existing:
        new_text = import_line + new_text
    new_text = new_text.rstrip() + "\n" + block
    spec_path.write_text(new_text, encoding="utf-8")
    scaffolded.append(str(spec_path))

print(json.dumps(scaffolded))
PY
```

The script reads `SCAFFOLD_OVERRIDE_PATH` from the environment so the
operator can route every draft to a single shared spec file; leave it
unset to take the default `e2e/<slug>.spec.ts` per draft.

## Commit the scaffolds

```bash
git add -- "${SCAFFOLDED[@]}" 2>/dev/null || git add -A
git commit -m "test(scenario): scaffold for #$ISSUE_NUMBER"
```

## Tell the user what comes next

```
Scaffolded N spec stub(s) for #<ISSUE_NUMBER>.

Next:
1. Fill in the test body and remove `test.skip`.
2. Run `devwatch scenarios sync` once the real test is in place.
3. Bind-on-sync will rewrite the draft line on #<ISSUE_NUMBER> to
   `acceptance-scenario: <suite>::<file>::<title>` automatically.
```

## Boundary

This skill **never** edits the issue body. It does **not** call
`devwatch scenarios sync` (the user runs that on their own schedule).
It does **not** delete draft lines — bind-on-sync handles the
rewrite once the real test ships.
