---
description: "Add or remove acceptance / regression scenario lines on an issue body."
capability: acceptance
---

Edit the `Links:` section on a GitHub issue body to add or remove
`acceptance-scenario:` / `regression-scenario:` coordinate lines. The
dashboard side panel's Acceptance tab (#1389) launches this skill from
its Add and Remove buttons; you can also invoke it directly.

The skill is body-edit only. It does **not** touch the lingtai cache
directly — the next GitHub auto-sync re-parses the body and rebuilds
the `scenario_links` cache (#1386).

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill edit-acceptance-scenarios --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done. The mandatory-reads include the authoritative `acceptance-scenarios` doc — the contract for the `Links:` section header, the `<suite>::<file>::<title>` coordinate this skill writes, and the title-stability invariant.

## Parse arguments

`$ARGUMENTS` is `<issue-number> [--add "<coord>" | --remove "<coord>"]...`.
Both flags are repeatable so a single invocation can apply multiple edits.

Examples:

- `1389 --add "smoke::e2e/login.spec.ts::logs in"`
- `1389 --remove "smoke::e2e/login.spec.ts::logs in"`
- `1389 --add "smoke::a.spec.ts::t1" --add "smoke::b.spec.ts::t2" --remove "smoke::old.spec.ts::gone"`

The first non-flag token is the issue number. Each `--add` /
`--remove` is followed by a double-quoted coordinate. Preserve the
quotes when shelling out and keep the value intact — do not split on
`::` yourself, the body parser does that.

By default a coordinate added by `--add` is rendered as
`acceptance-scenario:`. To emit a `regression-scenario:` line use
`--add-regression "<coord>"` instead. `--remove` strips matching
lines regardless of which key (`acceptance-scenario` /
`regression-scenario`) they used.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

## Read the current body

```bash
ISSUE_NUMBER=<N>
CURRENT_BODY=$(gh issue view "$ISSUE_NUMBER" -R "$REPO" --json body -q .body)
```

If the issue does not exist, `gh` exits non-zero — propagate.

## Compute the new body

The `Links:` section uses the bare `Links:` header (no `##`). The
parser at `src/watchdog_sensor/workflow/issue_links_sync.py` is the
authority — it terminates the section on the first blank line, accepts
`acceptance-scenario:` / `regression-scenario:` lines with or without
a leading `- ` bullet, and treats each coordinate as a literal.

Apply the edits in this order:

1. **Removes first.** For every `--remove "<coord>"`, drop any line in
   the `Links:` section whose right-hand side equals `<coord>`,
   regardless of whether the key was `acceptance-scenario:` or
   `regression-scenario:`. Lines outside the section are never touched.
2. **Adds next.** For every `--add` / `--add-regression`, append the
   matching line to the `Links:` section. Skip the append when an
   identical line is already present (idempotent — re-running the
   skill must not duplicate lines).
3. **No `Links:` section yet?** If the body has no `Links:` section,
   create one (separated from the body by a single blank line) before
   appending the new lines.

A small Python one-liner is the simplest way to do the edit
deterministically — bash string-rewriting is fragile when titles
contain `::`, slashes, or punctuation. Stage the new body to a tempfile:

```bash
TMP=$(mktemp)
python3 - "$TMP" <<'PY'
import os, sys
import textwrap
import re

tmp = sys.argv[1]
body = os.environ.get("CURRENT_BODY", "")
removes = [c for c in os.environ.get("REMOVES", "").split("\n") if c]
adds_acc = [c for c in os.environ.get("ADDS_ACC", "").split("\n") if c]
adds_reg = [c for c in os.environ.get("ADDS_REG", "").split("\n") if c]

LINK_KEYS = ("acceptance-scenario", "regression-scenario")
LINE_RE = re.compile(
    r"^\s*[-*]?\s*(?P<key>acceptance-scenario|regression-scenario)\s*:\s*(?P<val>.+?)\s*$"
)

# Split into pre-section / section / post-section. The section starts
# at a bare 'Links:' header line and ends at the first blank line.
lines = body.splitlines()
header_idx = None
for i, ln in enumerate(lines):
    if ln.strip().lower() == "links:":
        header_idx = i
        break

if header_idx is None:
    pre = lines
    section = []
    post = []
else:
    pre = lines[:header_idx]
    section_start = header_idx + 1
    section_end = section_start
    while section_end < len(lines) and lines[section_end].strip():
        section_end += 1
    section = lines[section_start:section_end]
    post = lines[section_end:]

# Drop matching scenario lines.
def keep(line: str) -> bool:
    m = LINE_RE.match(line)
    if not m:
        return True
    val = m.group("val").strip()
    return val not in removes

section = [ln for ln in section if keep(ln)]

# Build the set of scenario lines already present so we skip duplicates.
existing = set()
for ln in section:
    m = LINE_RE.match(ln)
    if m:
        existing.add((m.group("key"), m.group("val").strip()))

for coord in adds_acc:
    key = ("acceptance-scenario", coord)
    if key in existing:
        continue
    section.append(f"acceptance-scenario: {coord}")
    existing.add(key)
for coord in adds_reg:
    key = ("regression-scenario", coord)
    if key in existing:
        continue
    section.append(f"regression-scenario: {coord}")
    existing.add(key)

if section:
    out_lines = list(pre)
    if out_lines and out_lines[-1].strip():
        out_lines.append("")
    out_lines.append("Links:")
    out_lines.extend(section)
    if post:
        out_lines.append("")
        out_lines.extend(post)
else:
    # Empty section: drop the Links: header entirely so the body is clean.
    out_lines = list(pre)
    if post:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        out_lines.extend(post)

# Strip trailing blank lines to keep the body tidy.
while out_lines and not out_lines[-1].strip():
    out_lines.pop()

with open(tmp, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
    f.write("\n")
PY
```

Set the env vars before running the Python block:

```bash
export CURRENT_BODY
REMOVES=$(printf '%s\n' "${REMOVE_COORDS[@]}")
ADDS_ACC=$(printf '%s\n' "${ADD_ACCEPTANCE_COORDS[@]}")
ADDS_REG=$(printf '%s\n' "${ADD_REGRESSION_COORDS[@]}")
export REMOVES ADDS_ACC ADDS_REG
```

## Push the new body

```bash
gh issue edit "$ISSUE_NUMBER" -R "$REPO" --body-file "$TMP"
rm -f "$TMP"
```

## Confirm

Print a one-line summary listing the adds / removes that were applied
and remind the user that the dashboard cache rebuilds on the next
auto-sync (no extra command — the server reconciles automatically; see
`CLAUDE.md` → Top Rules → "GitHub is the source of truth").

```
Edited issue #<N> on <REPO>: +<n> acceptance, +<n> regression, -<n> removed.
The Acceptance tab will refresh after the next auto-sync.
```

## Boundary

This skill edits the issue body. It does **not** edit the `scenarios`
catalogue, run Playwright, or write to the `scenario_links` table —
the next GitHub sync rebuilds the cache from the body. To run a
scenario, use `/run-scenarios`. To create a regression bug from a
failed scenario, use `/new-bug --from-scenario`.
