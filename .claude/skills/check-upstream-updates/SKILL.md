---
description: "Check a watched upstream library for changes this project can benefit from, and file recommendations into the Inbox."
capability: core
---

Survey one **watched library** (an upstream third-party repo this project
depends on) for changes since the last check, judge whether *this* project can
actually benefit, and file each applicable finding into the devwatch **Inbox**
as an `upstream` recommendation. Then advance the library's mirror marker.

The value is **not** "a new version exists" — Dependabot/Renovate already do
that. The value is the **applicability judgment**: tie an upstream change to
*this* codebase ("v0.0.13 added `Session`; you hand-roll this in
`runner.py:88` — here is the recommendation to adopt it"). A bare version-bump
notice with no link to local code is noise — do not file it.

## Targeting

Determine the project repo (so every `devwatch` command targets the right
project):

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command. Note: `$REPO` is *this*
project's repo. The library's `repo` field (read below) is the **upstream**
repo you survey with `gh` — keep the two distinct.

## Parse arguments

`$ARGUMENTS` is the watched-library **id** to check (an integer). The schedule
tick passes it; a human can pass it for an on-demand check. If `$ARGUMENTS` is
empty, stop and ask which library to check.

## 1. Read the library config + marker

```bash
devwatch --repo "$REPO" watched-libraries list
```

This prints the project's watched libraries as JSON. Find the row whose `id`
matches `$ARGUMENTS` and read:

- `repo` — the **upstream** repo to survey (e.g. `openai/openai-agents-python`).
- `package_name` — the dependency name this project pins it under. May be a
  scoped npm name (`@udecode/plate`) or a path-qualified module
  (`github.com/foo/bar`), not just a bare name.
- `manifest_path` — the manifest carrying the pin (`pyproject.toml`,
  `package.json`, `Cargo.toml`, `go.mod`, `Gemfile`/`*.gemspec`,
  `composer.json`, …), possibly in a **subdirectory**
  (`packages/x/package.json`). **May be empty** — the project watches this
  upstream without a local pin (tracking releases to port ideas, not as a
  literal dependency). When empty, skip the version-behind comparison and judge
  applicability from how the project would *use* the change (step 4).
- `track_mode` — `releases` (inspect releases/tags only) or `releases_commits`
  (also inspect the commit stream).
- `last_checked_sha` / `last_checked_tag` / `last_checked_at` — the **marker**:
  the upstream point the last check covered. All null means this is the first
  check — span the recent history and set a baseline.

If no row matches the id, stop — the library was unwatched.

## 2. Read the upstream diff since the marker

Use `gh` against the **upstream** repo. Reads are read-only and need no auth
beyond the ambient `gh` login.

```bash
# Latest releases (newest first):
gh release list --repo "<upstream repo>" --limit 30 --json tagName,publishedAt,isDraft
# Tags, when the repo tags without cutting releases:
gh api "repos/<upstream repo>/tags" --jq '.[].name'
```

Everything newer than `last_checked_tag` is the release delta. For
`track_mode = releases_commits`, also read the commit stream since the marker:

```bash
gh api "repos/<upstream repo>/commits?since=<last_checked_at>" --jq '.[].commit.message'
# or, for a precise diff against the marker sha:
gh api "repos/<upstream repo>/compare/<last_checked_sha>...<default branch>"
```

Read the release notes / changelog entries for the new tags — that is where
the substantive "what changed" lives.

## 3. Inspect this project's pin + actual usage

This is the differentiator. For each candidate upstream change:

- Read `manifest_path` for the **pinned version** of `package_name` — how far
  behind is this project, concretely. Parse it per ecosystem:
  `pyproject.toml` (PEP 621 `dependencies` / `[tool.poetry.dependencies]`),
  `package.json` (`dependencies` / `devDependencies`, scoped keys like
  `@udecode/plate`), `Cargo.toml` (`[dependencies]`), `go.mod` (`require`
  lines), `Gemfile`/`*.gemspec` (`gem` / `add_dependency`), `composer.json`
  (`require`). If `manifest_path` is **empty**, there is no local pin — skip the
  "how far behind" math and lean entirely on the usage judgment below.
- Search the codebase for **how the package is actually used** (imports, the
  specific APIs called). A change to an API this project never touches is not
  applicable; a change to one it hand-rolls or calls heavily is. With no pin,
  this is the *only* signal — file a rec only when you can name a concrete local
  hook the upstream change would improve.

## 4. Judge applicability (what you decide)

For each upstream change, decide: **does this project benefit, and how?** Keep
only findings where you can name a concrete local hook — a file/line that would
change, an API to adopt, a workaround to delete. Discard the rest. Better to
file zero recs than a version-bump notice.

Classify each kept finding as a `feature` (new capability to adopt), `chore`
(maintenance / dependency hygiene), or `refactor` (replace a local workaround
with the upstream primitive).

## 5. File each recommendation into the Inbox

For each kept finding, write the body to a temp file (so multi-line markdown
survives the shell) and file it:

```bash
cat > /tmp/devwatch-upstream-rec.md <<'BODY'
**Upstream:** <upstream repo> <from tag> → <to tag>

<one-paragraph rationale tying the change to THIS project>

**Affected here:**
- `path/to/file.py:NN` — <what changes>
BODY

devwatch --repo "$REPO" watched-libraries add-rec \
  --repo "<upstream repo>" \
  --title "<concise recommendation title>" \
  --type feature \
  --external-id "<upstream repo>@<to tag>#<short-slug>" \
  --body-file /tmp/devwatch-upstream-rec.md
```

The `--external-id` is the dedup key: a later check that re-surfaces the same
finding updates the existing Inbox item in place instead of duplicating. Use a
stable shape like `<repo>@<tag>#<slug>`.

File **0..N** recs — zero is a valid, common outcome.

## 6. Advance the marker — exactly once, at the end

After filing every rec (or none), record the upstream point this check covered
so the next run starts from here:

```bash
devwatch --repo "$REPO" watched-libraries record-check <library id> \
  --tag "<latest upstream tag>" \
  --sha "<latest upstream commit sha, for releases_commits>"
```

Pass `--tag` for `releases` mode; add `--sha` for `releases_commits`. Update
the marker even when you filed zero recs — the window has still been covered.

## Boundary

- Survey + judge + file recs + advance the marker. Do **not** create GitHub
  issues directly — a human promotes a rec from the Inbox.
- All persistence goes through `devwatch` (the server is the single writer).
  Never touch the devwatch database.
- Advance the marker exactly once, after the recs are filed.
