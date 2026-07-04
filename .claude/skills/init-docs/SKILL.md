---
description: "Seed or refresh a project's documentation set — one skill, two modes (--mode create|update). Create seeds the recommended docs from the project's capabilities; update re-derives the installed tree's status so you can refresh what's stale."
capability: core
---

Seed or refresh this repo's documentation set. **One skill, two modes** — there is no separate create-only and update-only path (epic #2907):

- `--mode create` seeds the recommended doc set from the project's declared capabilities, then you adapt it to the repo.
- `--mode update` re-derives the status of the installed docs tree so you can refresh what is stale.

Leave everything **uncommitted** for human review. This skill takes no git action and opens no issues, branches, or PRs.

## Parse arguments

The launch embeds:

- `--mode` — `create` or `update`. Required. Run the matching section below and ignore the other.
- `--capabilities` — *(create)* comma-separated project shape, e.g. `python,typescript,duckdb`. When the launch omits it, omit it downstream too — the CLI seeds the catalog baseline.
- `--docs-root` — *(update)* the installed docs tree to refresh status against.

Fail fast if `--mode` is missing or is neither `create` nor `update`.

## Mode: create

The recommended set is seeded by the installed CLI — **no agent reasoning is needed for the default**:

```bash
devwatch init-docs --mode create --capabilities <capabilities>
```

It seeds into the current directory (the repo this terminal opened in) — do not pass a path positional. Pass the `--capabilities` value verbatim from the launch; omit the flag entirely when the launch did, so the CLI falls back to the catalog baseline (`general` docs + the `critical` rule). This copies the resolved docs into `documentation/general/`, the project / product scaffolds into `documentation/`, the matching rule files into `.claude/rules/`, and generates a starter `CLAUDE.md`. Existing files are skipped — pass `--force` only if the user explicitly asks for a refresh. Record the created vs skipped output.

Then **adapt** the scaffolded docs to this repo's actual identity:

- Read `./README.md` (if any) and the real project layout to learn the domain, stack, and conventions. If `documentation/general/documentation.md` was just seeded, follow its guidance.
- **Prune** docs that clearly do not apply (e.g. a TypeScript doc set in a pure-Python CLI). State the reason for each removal.
- **Adapt** placeholder language in the docs that stay — replace "the application" / "your project" with what this repo actually is.
- **Fill `CLAUDE.md`** — project name, one-line stack summary, key directories, top rules, dev commands. Mark unknowns `TBD`; do not invent commands.

## Mode: update

Re-derive the status of the installed tree and report it:

```bash
devwatch init-docs --mode update --docs-root <docs-root>
```

The output classifies every catalog doc as `installed` / `update_ready` / `adapted` / `available` and lists the ones needing attention. Status is read fresh every call — there is no cached state to invalidate; running this *is* the re-analysis that refreshes the settings Documentation view. Applying an individual change is the human's call through that view's **Update** / **Keep mine** buttons — your job is to refresh the analysis and explain it, not to overwrite docs the user adapted. Read the report and:

- **Flag `update_ready` docs** — unedited copies that lag the current template; a safe fast-forward the user can apply with one click.
- **Assess `adapted` docs** — locally edited, so they differ from the template. Read them, summarise what diverged, and recommend update-or-keep. Never overwrite without the user's say-so.
- **Surface `available` docs** the repo could add for its stack.

Report the status summary and your recommendations.

## Boundary

This skill seeds or refreshes documentation and stops. Leave the changes uncommitted for human review — no commits, no branches, no PRs, no issue creation.
