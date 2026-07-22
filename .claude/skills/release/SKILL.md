---
description: "Create a release from the dev branch to production."
capability: releasable
---

Ship a release through this repo's configured release stages: one PR per
stage, then a GitHub release tag.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill release --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Release shape is per-repo (not always staging → main)

The repo's `branches:` config decides how many release stages there are —
the CLI derives the flow and `--step auto` emits the right next step. Do
**not** assume a staging hop exists:

- **2-stage** (`dev` + `prod`, no `staging`): one `dev → prod` PR, then tag.
  There is **no** staging PR and **no** staging tag. `--step auto` goes
  straight to `production`.
- **3-tier** (`dev` + `staging` + `prod`): `dev → staging` PR, optional
  staging tag, then `staging → prod` PR, then tag.
- **single-branch** (`dev` only): nothing to release.

Trust `--step auto`. Run whatever step it reports — never force a staging
PR on a repo that has no staging stage.

## Parse arguments

Extract the optional workflow id and run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `""` → WORKFLOW_ID=(none), RUN_ID=(none)
- `$ARGUMENTS` = `"--run 7"` → RUN_ID=7
- `$ARGUMENTS` = `"--workflow-id 18 --run 7"` → WORKFLOW_ID=18, RUN_ID=7

If RUN_ID is present, forward it as `--run-id` to all `devwatch --repo "$REPO" release` calls.

WORKFLOW_ID is set by the workflow-scoped ship dispatch (#2917); the ad-hoc repo-level release leaves it empty. It is read only by the **Fold in the workflow's follow-up/risk notes** section below — when it is empty, that section is skipped entirely.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Intelligence (what you decide)

1. Read commits since the last tag. Group by type (feat, fix, refactor, other).
2. Determine the version tag (patch vs minor vs major).

## Compute the tag first

Resolve the version to a **real number** before running anything, and hold it in
`TAG` so no command below carries a literal. Read the current version from the
repo's version files (`pyproject.toml`, `package.json`, …) or the last release,
apply the patch/minor/major bump you decided above, and assign the result:

```bash
TAG="v<the version you computed>"   # replace with the real number, e.g. v1.2.3
```

`TAG` is `v` + semver — nothing else. `devwatch release` refuses any other shape
and publishes nothing, so a placeholder that slips through fails loudly instead
of shipping (#3657): a real GitHub Release marked Latest was once published on
the literal `vX.Y.Z` this skill used to print in its example commands, and a
published release is the hardest artifact in the pipeline to walk back.

Never run `gh release create` (or `git tag`) yourself. `devwatch release` is the
only thing that publishes — it derives the staging tag, keeps the create
idempotent across retries, and records the release against the run.

## Execution

The CLI auto-detects which step to run. Pass `--tag "$TAG"` on every call — the
first PR off `dev` uses it to bump `pyproject.toml`, `dashboard/package.json`,
and `src-electron/package.json` on the dev branch before the PR opens, so the
bump rides the flow to prod. The first PR is the `dev → staging` PR in a 3-tier
repo, or the `dev → prod` PR in a 2-stage repo:

The release notes are your own prose and name commits and symbols in backticks by convention, so pass them through a **quoted heredoc** — an apostrophe or a `$` in a hand-quoted string is eaten by the shell, and a backtick is executed as a command. These notes reach `gh release create` and publish a permanent public Release page, so a mangled value is the hardest one in the pipeline to walk back:

```bash
NOTES=$(cat <<'NOTES_EOF'
<release notes>
NOTES_EOF
)

devwatch --repo "$REPO" release --step auto --tag "$TAG" --notes "$NOTES" --run-id <RUN_ID>
```

Omit `--run-id` if no RUN_ID was parsed from arguments.

Possible outcomes (which fire depends on the repo's release shape above):
- **staging** -- 3-tier only: bumps version files, creates staging PR (dev -> staging). Wait for CI + merge.
- **staging-tag** -- 3-tier only, when `tag_staging: true` in repo config: creates the staging tag and its GitHub Release. Stays pending until that release exists, so a run that tagged and then died before publishing is picked up again instead of skipped, in which case the tag is derived from the staging version file and no `--tag` is needed. Wait for build if `build: true`.
- **production** -- creates the production PR. In a 3-tier repo this is `staging -> prod`; in a 2-stage repo it is `dev -> prod` (and carries the version bump itself). Wait for CI + merge.
- **wait** -- a release PR is already open or a build is pending. Report its status.
- **tag** -- the production branch is ready to publish: creates the tag and the GitHub Release. Also fires when prod already carries a version that was never published, in which case the tag is derived from the prod version file and no `--tag` is needed.
- **done** -- every branch is in sync *and* the prod version is published. Nothing to release.

A 2-stage repo therefore ships in a single `dev → prod` PR + one tag — no
staging PR and no staging-tag step are attempted.

After the production PR is merged, run the same command again to tag:

```bash
devwatch --repo "$REPO" release --step auto --tag "$TAG" --run-id <RUN_ID>
```

Each step is idempotent. Running `/release` again after a partial release picks up where it left off — the bump step skips silently when files already match the target version.

### PATH CLI reinstall (self-hosted devwatch only)

When the repo being released **is** devwatch itself, the production `tag` step
reinstalls the PATH CLI at `~/.local/bin/devwatch` from the release tag
automatically — the release is the *only* sanctioned installer, so the PATH CLI
and the shipped app can never diverge (#3173, epic #3172). Do **not** run
`uv tool install` by hand; after the tag step `devwatch --version` on `PATH`
equals the shipped app version. Releasing any other watched repo does not touch
the PATH CLI.

### Fold in the workflow's follow-up/risk notes

Once the production release tag exists — the `--step auto` run reported `Release <tag>: <url>`, so `gh release create` published the GitHub Release — fold the shipping workflow's accumulated step notes into the Release body as a **Known follow-ups** section, so the release page carries the risks and follow-ups the implementing agents recorded. This edits the Release body `gh release create` just wrote — no new posting mechanism.

**No WORKFLOW_ID → skip.** An ad-hoc release with no owning workflow has no notes to fold; do nothing here and go straight to the boundary.

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to the notes section. The digest is assembled from notes earlier agents wrote, so it may carry banned tokens or personal data — review and redact before posting.

2. Read the workflow's notes digest:

   ```bash
   devwatch --repo "$REPO" get-report --workflow "$WORKFLOW_ID"
   ```

   It prints a category-grouped markdown digest (`### Risks` / `### Decisions` / `### Follow-ups`), or nothing when there are no notes.

3. **Empty digest → stop.** Leave the Release body exactly as `gh release create` wrote it; no section is added.

4. Non-empty digest → append the reviewed digest to the Release body under a `## Known follow-ups` heading. Read the current body, keep it intact (the auto-generated changelog stays), and write it back for the production tag in `$TAG`:

   ```bash
   gh release view "$TAG" --repo "$REPO" --json body -q .body > "/tmp/release-body-$TAG.md"
   printf '\n\n## Known follow-ups\n\n' >> "/tmp/release-body-$TAG.md"
   # append the reviewed digest (redacted per step 1) below the heading, then:
   gh release edit "$TAG" --repo "$REPO" --notes-file "/tmp/release-body-$TAG.md"
   ```

Fold the notes into the **production** release (`$TAG`) only — not the staging tag. The staging tag is a build checkpoint; the production GitHub Release is the one a reader opens for "what shipped, and what to watch." `gh release edit` only rewrites the body of a release that already exists, so it cannot publish anything on its own.

## Boundary

Does NOT build or deploy. Deployment is handled by CI/CD.
