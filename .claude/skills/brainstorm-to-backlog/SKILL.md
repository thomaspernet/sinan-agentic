---
name: brainstorm-to-backlog
description: Read a brainstorming session and propose the work its thinking arrived at, for a person to edit — or file it when asked.
family: analysis
shipped-from: ab11264b4a7020d06e229e0595e12909811becd926431fe84a3129bca9ac8552
---
Read a brainstorming session and say what work its thinking arrived at —
either as a proposal a person edits, or as the work itself. Where that work
lands is its lane: issues on the project's repository, or the app-native
backlog item. A project registering no repository holds only the backlog lane
and is never asked. A project registering one holds both, and which one the
work files on is the person's choice — not the project's, and not yours — so
it is asked whenever the project holds both and the person has not said. The
conversion states which lane it took.

You were launched from the moment a session's thinking is being filed as work,
so the session is the attachment you were given. Propose unless the person
asked you to file: a proposal waits on the session page, where they open it
and change it before anything lands, which is the whole reason to run this
rather than type it.

## 1. Read the session

`read_brainstorm_session` with the session's uuid. It answers with the
session's notes and the entities gathered around it as uuids and names — not
as content — plus `work`: every issue and backlog item the session is already
linked to. A session's thinking can lead to more than one piece of work, so
linked work does not finish it — but name what is already linked before you
propose anything, and propose only work the session does not already hold.

Then read what matters with the `read` tool: the summary first, then the notes
whose names suggest they carry the conclusion, then the gathered documents the
notes lean on. Reading everything is rarely worth it; reading nothing makes
every step below a guess dressed as a plan.

## 2. Decide what the work is

The work is what the thinking arrived at, which is rarely what the session is
called — a session is named for the question it was opened on. Name it in one
short line, in the words the notes use, and put what the work involves in its
description rather than in the name.

A title is short because it is what the Backlog, the run screen and GitHub's
issue list show, one row beside other names — each renders the title alone and
cuts what does not fit. A title that holds the detail loses what the work is in
how it is done, and the detail it holds is read by nobody.

Then decide whether it breaks into steps. It does when the session's own
thinking splits into parts that are done in an order and reviewed separately;
it does not when the session converged on one thing that happens to be large.
Name a step only where the session supports it. A plan invented to look
thorough is the failure this skill exists to avoid — the person can read the
notes too, and a step they cannot trace back to them costs them the trust they
would otherwise put in the rest.

## 3. Offer it

`propose_brainstorm_conversion` with the session's uuid, the title, the steps
in the order they would be done, and the `lane` — `repository_issues` or
`backlog_item` — when the session's thinking or the person has said which.
Nothing is filed: the proposal waits on the session, and its page names it in
the File as work section, where the person opens it with "Review and file",
sees the lane it suggests, edits it, chooses the lane, and confirms what
actually gets filed. Proposing asks nothing of the project and is
never refused for the lane it names.

Each step is an object, `{"title": ..., "description": ...}`. Its `title` is one
short line naming what the step delivers — the same kind of name the work's
title is, for the same reason. Its `description` holds the detail: how the step
is done, what it touches, what the notes say about it. The description is
optional, but a step with anything more to say than its name says it there,
never in the title. On the repository lane the description becomes the child
issue's body, the one place an issue holds more than a name.

A title longer than the limit is refused, naming the limit and every title over
it, and nothing is written — on proposing as on filing. Shorten each title it
names to what the step is, move what you cut into that step's description, and
call again. Never cut a title to fit without moving the rest: the cut text is
the only copy of that detail.

Then say what you proposed and what in the session it came from — one line per
step, naming the note or document behind it. That is what the person is
reviewing; a proposal with no provenance is one they have to re-derive. End the
report with "Open the session, File as work" — where the proposal waits. The
Backlog lists only filed work, so a person told to look anywhere else finds
nothing and reads the proposal as lost.

## 4. File it only when asked

`convert_brainstorm_session`, same arguments and the same step shape, when the
person has said to file it outright rather than review it. It writes on the lane it is given:
`lane: "repository_issues"` files an issue for the work and one child issue per
step, answering `kind: "repository_issues"` with the numbers it minted;
`lane: "backlog_item"` writes the app-native backlog item, answering
`kind: "backlog_item"`. Branch on `kind`, never on which keys came back. On
the backlog lane the session's gathered entities are carried onto the item; on
the repository lane each issue it files is linked to the session. Either way the
session then reads as converted.

The lane is asked, not assumed. Omit it only on a project registering no
repository. On a project registering one, a conversion naming no lane answers
`converted: false` with an error naming the two lanes — ask the person which
lane the work files on and call again with it named, rather than picking one
because a repository happens to be registered.

`repo_uuid` names which repository to file on, on the repository lane. Omit it
when the project holds exactly one; a project holding several answers
`converted: false` with an error naming the repositories to choose between,
because a create is irreversible on GitHub. That refusal is an answer to read
on and call again with a repository named, not one to retry unchanged.

Not safe to repeat — every conversion files new work, even on a session that
already holds some. The answer's `already_linked` names the work the session
was linked to before it; read it, and never convert again to retry a filing
that already landed.

## 5. Link work filed another way

Work filed without converting — an issue created by hand or through another
skill, a backlog item made in the app — is linked rather than filed again:
`link_brainstorm_work` with the session's uuid and either `issue`, as its uuid
or `owner/name#N`, or `backlog_item_uuid`. Safe to repeat. A link that is wrong
— work that did not come out of this session — is removed with
`unlink_brainstorm_work`, naming the work the same way. Never write a
`brainstorm` line into an issue body to say where it came from: the link is the
record, and `read_brainstorm_session` lists it under `work`.

## 6. Name the steps that are not ready

A backlog item runs once each of its steps is ready: bound to the skill that
executes it, with every argument that skill requires answered. A step filed a
moment ago is bound to nothing, so a conversion on the backlog lane leaves work
that cannot run yet. After one, `read` the `backlog_item_uuid` it answered and
end your report by naming each step whose `ready` is false, with its uuid, and
pointing at `/prepare-backlog-step` as the skill that prepares it. The
repository lane files issues rather than steps and has nothing to name here,
and so does a proposal, which files nothing.

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

A `skipped` settle also says what became of this step's work, as an `outcome`
with that outcome's evidence. A skip naming none is refused, and so is one
whose outcome has nothing behind it: you are the only one who knows, and a bare
skip leaves every reader after you guessing which of the three it was.

- `already_delivered` — the work is already done, in this repository or
  another. Give `references`, one per place it landed: a commit as
  `owner/name@sha`, a pull request or issue as `owner/name#123`. It is the one
  outcome that says something shipped, and a run reads it to know this step
  delivered even though nothing landed on its own branch.
- `left_out` — the run decided not to do this work. Give `outcome_reason`, one
  line saying why.
- `not_needed` — the question turned out not to exist. Give `outcome_reason`,
  one line saying why.

A `halt_reason` is read by a person deciding what to do next, so write it as
the blocker in words they can act on, not as an error string. Never leave a
claimed unit `running`: a step that stops without settling is
indistinguishable from one still in flight.
