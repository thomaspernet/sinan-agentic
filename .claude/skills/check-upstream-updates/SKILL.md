---
name: check-upstream-updates
description: Check one watched upstream library for changes this codebase can use, file each as a recommendation, and advance the watch's marker.
family: analysis
shipped-from: 48837489aae7b0403149db7ccdafeb1308b340813f43c9fad9b63fa64cf3b778
---
Survey one watched library, an upstream repository this project depends on, for
what changed since the last check. Judge whether this project can use each
change, file each one it can as a recommendation on the project the watch
belongs to, and then advance the watch's marker.

The value is the applicability judgment, not the news that a version exists:
dependency bots already say that. A finding ties an upstream change to this
codebase ("the new release adds a session primitive; `runner.py:88` hand-rolls
one"). A version bump with no local site attached is noise, so do not file it.

## Reaching the app

The watch list and the recommendations both live in this app's graph and are
reached over its HTTP API, never its database. Two things are needed: the key
this session already talks to the app with, and the address the app answers on.

The key is on the `digital-brain` entry of the session's MCP config. Resolve
that entry the way the client does: the working folder's `.mcp.json` first, and
the account's own `.claude.json` when the folder has none.

```bash
ENTRY=""
for CONFIG in .mcp.json "${CLAUDE_CONFIG_DIR:-$HOME}/.claude.json"; do
  ENTRY=$(jq -ce '.mcpServers["digital-brain"] // empty' "$CONFIG" 2>/dev/null) && break
done
KEY=$(jq -r '.headers["X-API-Key"] // empty' <<<"$ENTRY")
API="${WUNOVA_API_URL:-}"
```

The address is `WUNOVA_API_URL`, which the app exports to every session it
launches. It is not taken from the entry's `url`: that names the proxy whenever
the proxy is enabled, and the key is only good against the app itself. The
session always runs on the machine the app runs on, so the launch hands it that
machine's own address.

If neither file holds a `digital-brain` entry with a key, stop and say so: the
check has nothing to authenticate with. If `WUNOVA_API_URL` is unset, the session
was not launched by the app; stop and say so rather than guessing an address.
The key is a credential. Never print it, and never write it into a
recommendation or a report.

## Which watch

`$ARGUMENTS` names the watch to check by its uuid, either bare or as
`watch="<uuid>"`, the form the app's own launches pass it in. Take the uuid out
of whichever form arrived:

```bash
WATCH=$(grep -oE '[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}' <<<'$ARGUMENTS' | head -n 1)
```

With no uuid in it, stop and say which input is missing. A check never picks a
library of its own.

## 1. Read the watch and its marker

```bash
curl -sS --fail-with-body -H "X-API-Key: $KEY" "$API/watched-libraries/$WATCH/"
```

The answer is the watch as stored.

- A 404 means the watch was removed or its repository was unregistered. Stop:
  there is nothing to check and no marker to move. Report that the watch is
  gone.
- `status: paused` means nobody is following this watch any more. Stop without
  surveying anything and without touching the marker. A paused watch keeps its
  marker so that resuming it reports what moved while it was quiet, and checking
  it now would use up that window. Report that the watch is paused.

Otherwise read:

- `watched_repo`, the upstream repository you survey with `gh`. It is not the
  repository that consumes it.
- `package_name`, the name this project pins it under. It may be a scoped npm
  name (`@scope/pkg`) or a path-qualified module, not only a bare name.
- `manifest_path`, the manifest carrying the pin (`pyproject.toml`,
  `package.json`, `Cargo.toml`, `go.mod`, a `Gemfile`, `composer.json`),
  possibly in a subdirectory. It may be empty: the project follows the upstream
  to port ideas from it rather than as a dependency. Then there is no pin to
  compare, and applicability rests on usage alone (step 3).
- `track_mode`, which is `releases` (follow releases and tags) or `commits`
  (follow the default branch, for an upstream that ships without releases).
- `last_seen_tag`, `last_seen_sha` and `last_seen_at`, the marker: the upstream
  point the last check covered.

When all three marker fields are null, this is the watch's first check. It
sets the baseline rather than reporting history. Survey only the window step 2's
listing returns (the newest 30 releases, or the default branch's last 30
commits), file what applies from it, and advance the marker to the newest point
in it. Nothing older is surveyed, now or later.

## 2. Read the upstream since the marker

Reads against the upstream need nothing beyond the ambient `gh` login.

```bash
gh release list --repo "<watched_repo>" --limit 30 --json tagName,publishedAt,isDraft
gh api "repos/<watched_repo>/tags" --jq '.[].name'
```

Everything newer than `last_seen_tag` is the delta; use the tag listing when
the upstream tags without cutting releases, and skip drafts. For
`track_mode = commits`, read the commit stream instead:

```bash
gh api "repos/<watched_repo>/commits?since=<last_seen_at>&per_page=30" --jq '.[].commit.message'
gh api "repos/<watched_repo>/compare/<last_seen_sha>...<default branch>"
```

Read the release notes or changelog entries for what is new; that is where
the substance lives. When nothing is newer than the marker, file nothing and go
to step 6.

## 3. Read this project's pin and its usage

This is what separates a recommendation from a version notice.

- Read `manifest_path` for the version of `package_name` this project pins, so
  you can say concretely how far behind it is. Skip this when `manifest_path`
  is empty.
- Search the codebase for how the package is actually used: the imports and
  the specific APIs it calls. A change to an API this project never touches does
  not apply. A change to one it hand-rolls or leans on does. With no pin, usage
  is the only signal, and a finding needs a concrete local site to stand.

## 4. Judge what applies

For each upstream change, decide whether this project benefits and how. Keep
only findings where you can name a local site: a file and line that would
change, an API to adopt, a workaround to delete. Discard the rest. Zero
recommendations is a common and valid outcome.

Classify each kept finding as `feature` (a capability to adopt), `chore`
(maintenance or dependency hygiene) or `refactor` (a local workaround the
upstream primitive replaces).

## 5. File each recommendation

File each finding against the watch you were handed. The watch tells the app
which project it belongs to and which upstream produced it, so you state
neither. The title and body are your own prose, so build the request with `jq`
from quoted heredocs: an apostrophe or a `$` in a hand-quoted string is eaten by
the shell, a backtick runs as a command, and a raw newline breaks the JSON.

```bash
TITLE=$(cat <<'TITLE_EOF'
<concise recommendation title>
TITLE_EOF
)
BODY=$(cat <<'BODY_EOF'
<one paragraph tying the change to this project, naming the upstream range it covers>
BODY_EOF
)
jq -n --arg external_id "<watched_repo>@<tag or sha>#<short-slug>" --arg title "$TITLE" --arg body "$BODY" --arg local_site "path/to/file.py:NN" --arg item_type feature '$ARGS.named' | curl -sS --fail-with-body -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" --data @- "$API/recommendations/watch/$WATCH/"
```

`local_site` is the file, and the line where there is one, that the finding
names. A finding with nothing to put there should have been discarded in step 4.

`external_id` is the dedup key. A later check that surfaces the same finding
resolves to the item already recorded rather than filing a second one, and keeps
whatever was decided about it, so a recommendation somebody turned down stays
turned down. Keep its shape stable: `<repo>@<tag>#<slug>`.

A 404 here means the watch went away during the check. Stop without advancing
the marker, and report what was filed before it did.

## 6. Advance the marker, once, at the end

After every recommendation is filed, or none was, record the upstream point
this check covered so the next one starts there. It is the field the project's
Upstreams view shows as "last checked", so the two cannot disagree.

```bash
curl -sS --fail-with-body -X POST -H "X-API-Key: $KEY" "$API/watched-libraries/$WATCH/seen/?tag=<newest upstream tag>"
```

Pass `tag` for `releases` and `sha` for `commits`. Whichever you leave out is
cleared, so the marker never names something the watch no longer follows.
Advance it even when you filed nothing: the window was still covered, and a
watch whose marker never moves reads as one that has never been checked.

Report the watch, the range surveyed, and each recommendation filed by title,
or that none applied.

## Boundary

- Survey, judge, file recommendations, advance the marker. Never open a GitHub
  issue: a person promotes a recommendation from the project's Recommendations
  view, which files the issue and records what it became.
- Never change what a watch follows, and never pause or resume one. The
  subscription is authored on the Upstreams view; a check only reads it and
  moves its marker.
- Never decide a recommendation, and never re-file one. Deciding is the
  reader's, and the dedup key resolves a repeat finding to the item already
  recorded.

## Reporting back

You are invoked either on demand — by a person who already knows what they want
— or as one step of a run. The two report back differently, so establish which
before doing anything.

Call `worklist_claim_item` with no arguments.

- `no_anchor` — you were invoked on demand. There is no unit to settle: do the
  work above, then report what you produced to the person who asked, naming it
  by issue number or path so they can open it.
- `claimed: true` — you are a step of a run. Do the work above against the
  claimed item's `title` and `attachments`, then settle with
  `worklist_set_item_status` and the item's `item_uuid`: `passed` when the step
  did what it says, `failed` with a `halt_reason` when it did not, and
  `skipped` when the question no longer exists. A step that decided its work
  fails says so with the reason, never with `passed`.
- `claimed: false` with `already_running` — another session has it. Stop.

A claimed step whose work stops for the person's decision — a proposal they
must approve, a choice only they can make — does not settle at the pause. Call
`worklist_set_item_status` with the `item_uuid`, `status: "waiting"` and a
`question`: one line saying what the person must decide. That settles nothing:
the step stays open and yours, and the run shows them the question. Once they
have answered, do what the answer asks, then settle — never `passed` at the
pause, which reads the step done before they have decided anything.

A `skipped` settle also says what became of this step's work, as an `outcome`
with that outcome's evidence. A skip naming none is refused, and so is one
whose outcome has nothing behind it: you are the only one who knows, and a bare
skip leaves every reader after you guessing which of the three it was.

- `already_delivered` — the work is already done, in this repository or
  another. Give `references`, one per place it landed: a commit as
  `owner/name@sha`, a pull request or issue as `owner/name#123`. Each must
  already be on its repository's development branch, or, in this run's own
  repository, on the run's integration branch when it has one: a commit
  reachable from it, a pull request merged into it, an issue closed by a change
  merged there. Work that sits on an unmerged branch is not delivered, and a
  reference to it is refused by name. It is the one outcome that says
  something shipped, and a run reads it to know this step delivered even
  though nothing landed on its own branch.
- `left_out` — the run decided not to do this work. Give `outcome_reason`, one
  line saying why.
- `not_needed` — the question turned out not to exist. Give `outcome_reason`,
  one line saying why.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on, not as an error string. Never leave a
claimed unit `running`: a step that stops without settling is
indistinguishable from one still in flight.
