---
description: "Scaffold an HTML mockup into a brainstorm session — resolve-or-create the session, write mockups/<name>.html from a self-contained scaffold, index it in the README, and report a file:// path. Untracked scratch; no git action."
capability: core
---

Build an HTML mockup and drop it in the one deterministic place mockups belong: the brainstorm session's `mockups/` folder. Stop.

A mockup is pre-issue design thinking — a throwaway HTML page you open in a browser to feel out a layout before any code exists. Its home is `<brainstorming.root>/<DD-MM-YY>/<slug>/mockups/<name>.html` (epic #2849): untracked like the rest of the session, so it never pollutes a code PR; indexed in the session README's `## Files` table; opened via a plain `file://` link with no server or build step. This skill resolves-or-creates that session, writes the file, indexes it, and reports the path. It takes **no git action** — the session is scratch.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill mockup --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Parse arguments

Extract the session slug and optional mockup name from `$ARGUMENTS`:
- `$ARGUMENTS` = `"checkout-flow"` -> SLUG="checkout-flow", NAME=(defaults to SLUG -> "checkout-flow")
- `$ARGUMENTS` = `"checkout-flow --name payment-step"` -> SLUG="checkout-flow", NAME="payment-step"

SLUG names the brainstorm session; it must be kebab-case (`[a-z0-9-]+`). The CLI rejects anything else. NAME is the mockup file's basename (also kebab-case) and defaults to SLUG when not passed — pass `--name` to keep several mockups in one session (`checkout-flow/mockups/payment-step.html`, `.../review-step.html`). Never include the `.html` extension or a directory in NAME; the skill adds the extension and the `mockups/` folder is fixed.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Resolve the session's `mockups/` directory

Do not invent the path. Ask the CLI for it — it resolves-or-creates today's `<DD-MM-YY>/<SLUG>/` session under the project's configured brainstorm tree (`brainstorming.root`), creates the session README if absent, creates the `mockups/` subfolder, and prints its absolute path:

```bash
MOCKUPS_DIR=$(devwatch --repo "$REPO" mockup-dir "$SLUG")
```

This is idempotent (#2851): re-running `/mockup` for the same slug on the same day reuses the existing session and `mockups/` folder rather than failing. `MOCKUPS_DIR` is the absolute path to `<session>/mockups`; the session folder is its parent and the session README is `<session>/README.md`.

## Intelligence (what you build)

1. Build the actual mockup the user asked for as a **single self-contained HTML5 document** — everything inline (`<style>` in the `<head>`, any script in a `<script>` tag), **no external resources** (no CDN links, no `<img src>` to the network, no web fonts). It must render correctly opened directly from disk over `file://`, offline. Use system fonts and inline SVG or CSS for any visuals.
2. When there is no specific design in the conversation to render, lay down the minimal scaffold below as a starting point and tell the user it is a blank canvas to iterate on. When there *is* a design in context, start from this scaffold and fill `<title>` and `<body>` with the real layout.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title><MOCKUP TITLE></title>
  <style>
    :root { --bg: #0b0c10; --panel: #16181d; --fg: #e8eaed; --muted: #9aa0a6; --accent: #4f7cff; }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      background: var(--bg); color: var(--fg); line-height: 1.5;
    }
    main { max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem; }
    h1 { margin: 0 0 0.5rem; font-size: 1.5rem; }
    p.lead { color: var(--muted); margin: 0 0 2rem; }
    .panel { background: var(--panel); border: 1px solid #23262e; border-radius: 12px; padding: 1.5rem; }
  </style>
</head>
<body>
  <main>
    <h1><MOCKUP TITLE></h1>
    <p class="lead">Replace this scaffold with the real layout.</p>
    <section class="panel"><!-- mockup content here --></section>
  </main>
</body>
</html>
```

3. Write the file with your editor tools (not `git`) to `$MOCKUPS_DIR/<NAME>.html`. Overwriting an existing same-named mockup is fine — it is scratch; if the user wants to keep the old one, they passed a different `--name`.

## Index the mockup in the session README

The session README is the index for the folder; a new mockup must show up there. The README lives at `<session>/README.md` — the parent of `$MOCKUPS_DIR`. Read it, then add a row to its `## Files` table:

- If a `## Files` table already exists (sessions created by `/new-brainstorm` seed one), append a row to it.
- If there is no `## Files` section (a minimal session the `mockup-dir` resolver just scaffolded has only frontmatter + a title), add a `## Files` section with a header row, then the mockup row.

The row points at the mockup with a path relative to the session folder and a one-line purpose:

```markdown
| `mockups/<NAME>.html` | <one line: what this mockup shows> |
```

Edit the README with your editor tools only. Do not `git add` it — the session is untracked (see below).

## Report the `file://` path

Print the absolute browser link so the user can open it in one click:

```bash
echo "file://$MOCKUPS_DIR/<NAME>.html"
```

## The session is scratch — never git-tracked

The brainstorm tree is a plain sibling folder, not part of any code repo. Everything you wrote — the mockup HTML and the README index row — lives under the project's configured `brainstorming.root`, outside the code tree. **Do NOT commit, push, or `git add` anything.** There is no branch to put it on and nothing to track; the dashboard brainstorm primitive scans the folder directly. Take no git action.

## Boundary

This command resolves-or-creates the session, writes `mockups/<NAME>.html`, indexes it in the session README, reports the `file://` path, and **takes no git action**. It does NOT open an issue and does NOT write code into the repo. Report the `file://` path and offer to iterate on the mockup in place; when the design converges into something actionable, point the user at `/new-feature --from-brainstorm <session>` (or `/new-bug --from-brainstorm <session>`).
