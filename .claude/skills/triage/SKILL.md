---
description: "List and triage open issues."
capability: core
---

Review open issues, highlight what needs attention. Help user decide what to work on.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` and `gh` command to ensure the correct repo is targeted.

## Execution

1. `gh issue list --repo "$REPO" --state open --limit 30 --json number,title,labels,createdAt`

## Intelligence (what you decide)

Present the issues as a summary table. Highlight:
- Issues without a priority label
- Issues with `source:auto` label (auto-created -- may need human review)
- P0/P1 issues not yet in progress

Ask which issues to work on next.

## Boundary

This command triages. It does NOT fix or implement anything. User decides the next action.
