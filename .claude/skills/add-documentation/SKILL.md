---
name: add-documentation
description: Update the documentation the landed diff made wrong — including prose naming a symbol the diff deleted or renamed.
family: writing
shipped-from: 16977c40f6984744c92434aa21410ba34060b6fe0a33235288f2ffa21e9ddd81
---
Update the documentation pages the landed work changed, written from the story
of that work.

Documentation is a page in the app, not a file in the repository. This stage
does not start from a list of changed files. It reads the story of the run —
why the work was wanted, what each piece of it did, and the diff that proves
it — and answers every documentation page that story touches: rewritten when
the merged behaviour is now misrepresented, or confirmed when the change left
the page accurate. Where the diff lands a surface no page describes, it creates
the page. It takes no git action beyond reading the diff — no branch, no
commit, no push — and writes no file: run `git status --porcelain` when you
start and again before you settle, and the two readings are the same.

This stage is one of the run's tracks. It starts beside the run's pull request,
alongside the propagation scan and the rule pass where the run takes those on,
in a copy of the code of its own pinned to the commit where every member had
landed — the directory this session opened in. The pull request opens while you
work, and the merge may land before you finish, so read the code in this copy
rather than in the repository's own checkout.

A run over pages has no repository, no pull request and no diff: its members
are pages, and what it delivered is what they produced. This stage then starts
beside that run's chain in the run's own directory. Read *On a run over pages*
below before you gather anything; every other step reads as written.

## 1. Read the story

Call `delivery_run_story` with no arguments: in a session launched for a run it
reads that run. Each part of what it answers has a use here.

- `brainstorm`: the session the work was filed from, with its summary and its
  decisions note. This is why the work was wanted and what was settled, so a
  page written from the story states the decision as well as the mechanism.
- `members`: one per child issue, with the debrief its producing step wrote and
  the settle notes its steps left. This is what each piece did, in the words of
  the session that did it.
- `diff`: the paths the run's branch changed — the merge commit's diff once
  merged, the branch's diff from its fork point before. This is the proof:
  nothing is documented that the diff does not show. Read the code behind a
  path once a page describing it has to be decided.
- `candidates`: per debrief, the documentation pages nearest to it by
  similarity, each with its `covers` and its coverage `status`. How many pages
  each debrief proposes, and the score below which a page is none, are yours
  to set: call again with `candidate_count` and `candidate_floor` to widen the
  search when the nearest pages do not match the change, and to tighten it when
  they are noise. Either left out takes the configured value.

A `diff` carrying a `gap` has no paths to confirm any page against: settle
`failed` and name the gap, unless the gap is `no_repository` on a run over
pages, which is that run's ordinary shape. A member with no debrief is read
from its settle notes and its issue instead, and named in the settle note.

## On a run over pages

The story names no `epic`, no member names an issue, and its `diff` carries the
`no_repository` gap. What the run delivered is what its members produced, so
their outputs and debriefs are the proof the diff is on a run over code, and
the steps below change in three places.

- Gather the pages from `candidates`, and `search` the project's pages for the
  names and terms the members' outputs introduce. Skip `documentation_coverage`:
  it reads repositories, and this run changed none. Confirm a page by reading it
  against the outputs and the debriefs; a page they do not bear on is left alone
  and named in the settle note.
- Answer each page with `answer_documentation_page` and no `covers`. It joins
  the page to every item of the run that produced an output. A run whose items
  produced nothing is refused, and that refusal is the halt to settle `failed`
  with.
- Write no new documentation page. A page is created with the code globs it
  covers, and a run over pages has none, so name the surface a page is missing
  for in the settle note, for a person to decide.

The coverage checks that close step 4 read code, so they do not apply here:
settle once every page gathered carries its answer.

## 2. Gather the pages to answer

Two sources propose pages, and the diff decides which of them this stage
answers.

1. Call `documentation_coverage`. A page whose `covers` match a path in the
   diff is a page this stage answers. A page that reads stale only for paths
   outside the diff was made stale by other work: leave it, since this stage
   documents what this run landed. When `repos_unread` names the run's
   repository, its paths were never read against any page: settle `failed` and
   name the checkout.
2. Take every page in `candidates`. Similarity proposes; it does not decide.
   Confirm a candidate that declares `covers` by reading the code under them
   against the diff, and one that declares none by reading the page against the
   diff and the debrief it was proposed for. A candidate is confirmed when the
   diff changes something the page says. One that is not is left alone and
   named in the settle note, so a person sees what was considered; a rewrite on
   similarity alone is churn a reviewer has to read.

A project whose pages carry no `covers` is not a reason to stop: the candidates
are how its pages are found, and every page this stage answers leaves carrying
the covers it was answered for.

## 3. Answer every page gathered

For each page:

1. Open it with `read`.
2. When the diff moved code the page describes — renamed or relocated a path
   out from under one of its globs — replace its globs with
   `update_documentation_covers`, passing every glob the page now covers. A
   glob naming a path that no longer exists covers nothing, and the page would
   read current however its code changes from here on.
3. Decide whether the merged behaviour, architecture or API is now
   misrepresented, or whether the change was internal. A page is wrong when it
   misses a new entity, endpoint or configuration section; names a symbol the
   diff renamed, relocated or deleted; describes a pattern the diff changed; or
   states a count or exhaustive list a new call site made wrong. The decisions
   and the debriefs say what the change meant; the diff says what it did.
4. When it is misrepresented, rewrite it with `update_page_content`, passing
   the whole page as markdown: one cohesive rewrite per page, not one edit per
   changed path.
5. Record the answer with `answer_documentation_page`: `answer` `rewritten`
   after a rewrite, `confirmed` when the change was internal. It joins the page
   to the work this run delivered, so the page can later be read back to the
   epics that shaped it, and a `confirmed` answer records the page as re-read
   against the change, which is what keeps "confirmed accurate"
   distinguishable from "nobody looked". Pass `covers`: the repository-relative
   globs of the diff's paths the page was answered for, narrow enough that a
   change under them is one this page has to answer for. They are required for
   a page that declares none, and are added to the globs of a page that does.

Every page gathered gets one of the two answers; a covering page left
unanswered still reads stale, and the stage fails on it.

`update_page_content` replaces the whole body with what it is given, as
markdown. Write it from the page's current body as `read` returns it, keeping
the headings, lists and tables it holds. Change what the change calls for and
leave every passage it does not touch exactly as it was: a house style (dash or
arrow substitutions, re-quoting, re-wrapping) is never applied across a page
the change did not otherwise affect.

Search the project's pages with `search` for every symbol the diff deleted or
renamed. A page still naming one is wrong whether or not its `covers` match the
diff, and gets the same rewrite and the same answer.

## 4. Write the page nobody has written

`never_written` lists every directory holding changed code no page covers,
including code other work changed, so an entry there is not on its own a page
to write. Write one when this diff lands a surface a reader needs explained —
a new entity, service, endpoint family or subsystem — and neither a covering
page nor a confirmed candidate describes it.

1. Create it with `create_page`, `type_name` `documentation`, named as a reader
   would look it up. Give it the `category` the pages describing the
   neighbouring code already file under, and `covers`: the
   repository-relative globs of the code it describes, narrow enough that a
   change under them is one this page has to answer for. The create is refused
   without either. Write its body as markdown, from the story: the decision
   behind the surface as well as its shape.
2. Record it with `answer_documentation_page`, `answer` `rewritten`, so the new
   page is joined to the work that made it necessary.
3. Link it from the concept page for its surface: open that page with `read`,
   add a wikilink to the new page where the page discusses the surface, and
   rewrite it with `update_page_content` from its markdown body with only the
   link added, held to the same rewrite as a page answered above.

The stage's verdict reads the coverage once you settle. It fails while a page
covering the diff reads stale. When no page of the project declares `covers`
at all nothing could read stale, so the verdict passes and names the missing
covers rather than calling the docs current; every page this stage answers
leaves carrying covers, and the next run is read against them.

Name every page this stage answered in the settle note, with its answer, and
every page it created, so a person can find them in the Documentation view. A
`never_written` area this diff did not make necessary is named there too, and
left for a person to decide.

## 5. Link, do not restate

Pages link each other and never duplicate each other. If a concept is
explained on one page, link it from the second rather than explaining it again
— two copies of one fact means one of them is already wrong.

A link is a wikilink carrying the linked page's name exactly as it is titled:
`[[Status Model]]`. It resolves by that name alone, so search the project's
pages with `search` for the name before writing it: a name no page carries
links to nothing.

A page the diff did not affect is left alone. Rewriting an accurate page to
look busy is churn a reviewer has to read.

## Settling

You were launched for one stage of this run as a whole, not for one document,
so there is nothing to claim. What the run is working — its repository, its
epic and its integration branch — is stated in the launch's own *The run you
were launched for* block; read them there, never from the checkout, which can
hold several epic branches and proposals that are not this run's. Every child
of the run has already settled by the time this stage starts; the work below
acts on what they landed.

Settle with `worklist_set_stage_status`, which takes no uuid — the run and the
stage rode in with the launch: `passed` when the stage did what it says;
`failed` with a `halt_reason` when it could not — the reason in words a
person can act on.

Give the settle a `note`: a few sentences in your own words for a person
reading the issue later — what you found, what you chose, and what you left.
It is stored verbatim against this attempt, so write prose, not a status
string and not a commit message. It is optional — a settle with no note is
valid — and it is not the `halt_reason`: the reason says what stopped the
unit, the note says what the work was.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on — not as an error string. Never leave the
unit `running`: a step that stops without settling is indistinguishable from
one still in flight.
