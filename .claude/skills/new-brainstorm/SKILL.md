---
description: "Open a brainstorming session — an untracked folder under the project's brainstorm tree at <DD-MM-YY>/<slug>/ with a frontmatter README. Optionally pre-link it to an issue."
capability: core
---

Create a brainstorming session folder. Stop.

A brainstorming session is the pre-issue thinking space — the "why" that produced a feature or epic. It lives on disk under the project's configured brainstorm tree (`brainstorming.root`) at `<DD-MM-YY>/<slug>/` with a mandatory `README.md` carrying frontmatter (title, status, linked_issues). The brainstorm tree is a plain sibling folder, never git-tracked. The session is a folder so it can hold many files and subfolders as the thinking grows.

This command only creates the scaffold. It does NOT open an issue. Use `/new-feature --from-brainstorm <session>` (or `/new-bug --from-brainstorm <session>`) when the brainstorming converges into something actionable.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill new-brainstorm --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Parse arguments

Extract slug and optional flags from `$ARGUMENTS`:
- `$ARGUMENTS` = `"dark-mode-rollout"` -> SLUG="dark-mode-rollout", EPIC=(none), ISSUE=(none), DESCRIPTION=(none)
- `$ARGUMENTS` = `"dark-mode-rollout --description 'figure out theming primitives'"` -> SLUG="dark-mode-rollout", DESCRIPTION="figure out theming primitives"
- `$ARGUMENTS` = `"dark-mode-rollout --epic 1234"` -> EPIC=1234 (pre-link to epic #1234)
- `$ARGUMENTS` = `"dark-mode-rollout --issue 1234"` -> ISSUE=1234 (pre-link to feature #1234)
- `$ARGUMENTS` = `"dark-mode-rollout --body-file /tmp/devwatch-brainstorm-body-XXX.json"` -> BODY_FILE="/tmp/..."

`--epic` and `--issue` are mutually exclusive — a session pre-links to at most one issue at creation time. Use `/link-brainstorm <session> <issue>` later to add more.

SLUG must be kebab-case (`[a-z0-9-]+`). The CLI rejects anything else.

**If `--body-file <PATH>` is present:** read the file with your Read tool — it is a JSON object with `{title, body}` keys. Use those as the starter README's title and body. Do NOT paste the file contents into Bash; read it with the Read tool.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Intelligence (what you decide)

The README is a **scaffold and index**, not a container for the whole session. It gives a reader scanning the folder a fast orientation; the detailed thinking lives in separate, topic-specific child files that grow over time. A session collapses into an unreadable wall of prose the moment the README tries to hold everything — keep it a summary that points outward.

1. Pick a good slug. Kebab-case, short, descriptive (`dark-mode-rollout`, not `dm` or `dark_mode`). Reflects the topic, not the conclusion.
2. Author a starter README body as **summary + status + index**, not a body of prose:
   - `## Summary` — one or two sentences: the question or pain that triggered the session, plus the rough shape of the expected output (single feature? epic? scrapped idea?). Weave `--description` in here if it was passed.
   - `**Status:** draft` — a human-readable status line that mirrors the frontmatter; the human moves it forward as the session matures.
   - `## Files` — the links/index section. A one-line note that detailed thinking lives in topic-specific files next to the README, followed by a small table listing each child file and its purpose. Seed the table with this `README.md` row plus the first child file you expect, so the convention is visible from the start.
3. Do NOT pour the full Why / Open Questions / What's next prose into the README. As the thinking grows, write each sub-topic into its own small `.md` file inside the session folder (`open-questions.md`, `tooling.md`, `per-repo-plan.md`, …) and add a row for it under `## Files`. The README stays a short summary + index; the child files carry the detail.
4. Non-markdown artifacts go in a subfolder, never loose in the session root — and HTML mockups have a fixed home: `<session>/mockups/<name>.html`. A mockup is pre-issue design thinking you open in a browser over a plain `file://` link; index it as a `mockups/<name>.html` row under `## Files` alongside the markdown notes. Don't hand-place it — `/mockup <slug>` resolves-or-creates the session and writes the file there. Images and diagrams the notes reference sit in their own subfolders (`diagrams/`, …) the same way.
5. Pick a title for the frontmatter. Title-case, short — what a reader scanning a list of sessions would scan for.

Pass the starter README content via a `--body-file` JSON file rather than `--body` on the command line so newlines round-trip cleanly:

```bash
BODY_FILE=$(mktemp -t devwatch-brainstorm-body-XXXX.json)
cat > "$BODY_FILE" <<'JSON'
{
  "title": "<title>",
  "body": "## Summary\n\n<one or two sentences: what triggered the session and the rough shape of the output>\n\n**Status:** draft\n\n## Files\n\nDetailed thinking lives in topic-specific files next to this README — one per sub-topic, created as the thinking grows. Add each here as you create it.\n\n| File | Purpose |\n| --- | --- |\n| `README.md` | This file. Summary + status + index. |\n| `open-questions.md` | The unknowns this session needs to resolve. |\n"
}
JSON
```

## Execution

```bash
devwatch --repo "$REPO" new-brainstorm "$SLUG" \
  --body-file "$BODY_FILE" \
  [--epic $EPIC | --issue $ISSUE]
```

Add `--epic <N>` OR `--issue <N>` only if the user passed one (never both — they are mutually exclusive). The CLI:

- Creates `<DD-MM-YY>/<SLUG>/README.md` under the project's resolved brainstorm tree (`brainstorming.root`), with frontmatter (title, status=draft, linked_issues=[N] when pre-linked, else `[]`). It prints the absolute folder path.
- When `--epic` or `--issue` is set, appends a `brainstorm: documentation/project/brainstorming/<DD-MM-YY>/<SLUG>` line under the issue body's `Links:` block. This is the stable label form (decoupled from the physical location), so moving the tree out of the repo never rewrites the body links.
- Fails cleanly if the folder already exists — overwriting a session is never an accident.

Delete the body-file after the CLI succeeds:

```bash
rm -f "$BODY_FILE"
```

## The session is scratch — never git-tracked

The brainstorm tree is a plain sibling folder, not part of any code repo. The CLI wrote the session under the project's configured `brainstorming.root` (an untracked location outside the code tree). **Do NOT commit, push, or `git add` anything** — there is no branch to put it on and nothing to track. The session stays on disk as scratch; the dashboard brainstorm primitive scans the folder directly (no git involved).

Take no git action here. The CLI already printed the absolute session folder — use that path when you report back.

## Boundary

This command creates the session scaffold under the project's brainstorm tree and takes **no git action** — the brainstorm folder is untracked scratch. It does NOT open an issue and does NOT write code. Report the absolute session path (the folder the CLI printed) and ask: *"Want to expand it? Keep `<path>/README.md` a short summary + index, and push detailed thinking into topic-specific files next to it (`open-questions.md`, …), linking each under `## Files`. When the thinking converges, run `/new-feature --from-brainstorm <session>` (or `/new-bug --from-brainstorm <session>`)."*

When you launched from a `--body-file`, delete the file after `devwatch new-brainstorm` succeeds:

```bash
rm -f "$BODY_FILE"
```
