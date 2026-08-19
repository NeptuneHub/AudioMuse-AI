---
name: code-review
description: Review a target - a commit, a range, a PR, or the current staged/unstaged working tree - against the head of the REMOTE MAIN branch, along three axes - Correctness (does it work, everywhere, without regressing callers?), Guideline (does it follow CONTRIBUTING.md and the project's own hard rules?), and Resource utilization (does it still hold at 1 to 10 million songs?). Each axis runs as its own parallel sub-agent and the findings are reported side by side. Use when the user wants to review a commit, a branch, a PR, or work-in-progress changes.
---

Three-axis review of a diff. Each axis is a separate question, and each one runs as **its own sub-agent**, in parallel, so they don't pollute each other's context:

- **Correctness** - does the change actually do what was asked, does it work on every build flavour, and does it break anything that worked before?
- **Guideline** - does it follow `CONTRIBUTING.md` and the project's own hard rules, and does it leave dead or duplicated code behind?
- **Resource utilization** - is the CPU and RAM cost right for a catalogue of 1 million songs, and still right at 10 million?

This skill resolves the diff, gathers the shared context, spawns the three agents, and aggregates what they report.

## 0. Ground rules

- **Read-only.** A review never edits, stages, unstages, commits, stashes, checks out, resets, reverts, or rebases anything. `git fetch` is the only command allowed to write to the repo, and it only moves remote-tracking refs.
- **Other agents may be editing this working tree live.** Files that changed without your doing are not yours to fix or revert - review them, leave them alone.
- **No scratch files in the repo.** Write the diff and any notes to the session scratchpad directory, never inside the working tree: a stray `review_diff.tmp` ends up in the very diff being reviewed.

## 1. Resolve the baseline - the head of remote main

**The baseline is the head of the remote main branch. Always, by default**, including when the user says only "review my staged changes" or "review commit abc123".

```sh
git fetch --quiet origin main
BASE=$(git rev-parse origin/main)
```

- **The remote is always `origin`, and `origin` is `NeptuneHub/AudioMuse-AI`.** Do not run `git remote -v`, do not reason about which remote looks like upstream, do not "prefer" one over another - there is nothing to choose. Any other remote in the list is somebody's fork; a fork's `main` is never the baseline. Ignore them entirely, and never write a sentence in the report about which remote is upstream.
- **The fetch is mandatory.** A stale remote-tracking ref silently reviews against yesterday's main, and every conclusion inherits the error.
- If `origin` or `origin/main` does not resolve, **stop and ask the user**. Do not substitute another remote and do not guess another branch name.

### The exception: the current branch head

Use `HEAD` as the baseline **only when the user clearly asks for it** - "against HEAD", "against the current branch", "vs my last commit", "just what I changed since HEAD". This is rare. When the request is ambiguous, use remote main and say so in the header; never fall back to `HEAD` silently.

A user-supplied base is also an explicit override: a range (`abc123..def456`), "since v0.9.0", or "against devel" all name their own base. Honor it.

## 2. Resolve the target - what is being reviewed

| What the user gave | Target | Diff to review |
| :---- | :---- | :---- |
| nothing | working tree: staged + unstaged + untracked | `git diff $BASE` |
| "staged" / "what I have staged" | the index | `git diff --cached $BASE` |
| "unstaged only" | worktree vs index | `git diff` - note in the header that this one is necessarily against the index, not remote main |
| "from commit X to the last one" | X through the working tree | `git diff <sha>^` - see below, both ends have a trap |
| a commit SHA / tag / branch, on its own | that ref's tree | `git diff $BASE..<ref>` |
| an explicit range `A..B` | as given | `git diff A..B` - the user named both ends |
| a PR number | the PR head | `gh pr diff <N>`; the base is then the PR's base branch, state it |

With no argument the default target is the whole working tree, so commits made on the current branch but not yet on remote main are part of the review too - that is the point of baselining on main.

### "From commit X to the last one" - both ends have a trap

**The start is inclusive.** When the user names a commit as the start - "starting from `0a8352c7` to the last one", "review from `abc123` onward" - they mean **that commit's own changes are in scope**. Git's `A..B` excludes `A`, so the naive reading silently drops a whole commit. Use the parent, `<sha>^`.

**The end is the working tree, not `HEAD`.** "To the last one", "to now", "onward", or no stated end means **everything up to and including what is staged and unstaged right now**, plus untracked files. Uncommitted work is the code most likely to hold the bug - never leave it out. Omit the end of the range entirely and git diffs against the working tree:

```sh
git diff <sha>^            # correct: <sha> itself, every commit after it, and all staged + unstaged work
git diff <sha>^..HEAD      # wrong: stops at the last commit, uncommitted changes invisible
git diff <sha>..HEAD       # wrong twice over
```

Then list untracked files with `git status --porcelain` and read them in full, since no diff form shows them.

Only when the user names an **end commit** explicitly ("from `abc123` to `def456`") does the range stop there and leave the working tree out.

"Since X" or "after X" is the exception on the start side and stays exclusive. Whenever it is ambiguous, take the widest honest reading - inclusive start, working-tree end - and state it in the header (`Base: 0a8352c7^ (parent of the named commit)`, `Target: working tree`) so the reader can catch it if it was wrong. Then cross-check the start against remote main: if `<sha>^` resolves to the same commit as `$BASE`, the two readings agree and there is nothing to flag.

## 3. Guards before spawning anything

A bad ref or an empty diff must fail here, not inside three parallel sub-agents. When the target is the working tree it is not a ref, so use `HEAD` as its stand-in for the ref-only checks below - and remember the review still covers the uncommitted work on top of it.

- `git rev-parse` resolves both base and target.
- The diff is non-empty.
- **Has main moved ahead?** `git rev-list --count <target>..$BASE`. If greater than 0, main carries commits the target does not. Say so in the header and review the merge-base diff (`git diff $BASE...<target>`, three-dot) instead - otherwise main's own newer commits show up as deletions and generate junk findings.
- **Is the target already merged?** If `git merge-base --is-ancestor <target> $BASE` succeeds, comparing it against main's head is meaningless. Review the commit on its own (`git show <target>`) and say why in the header.
- **Untracked files never appear in `git diff`.** List them with `git status --porcelain` (`??` entries) and read them in full - every line is new code. Do not `git add -N` them; that writes the index.

Also capture the commit list for the header: `git log $BASE..<target> --oneline`.

## 4. Identify the spec - what this change was supposed to do

The spec is what the change is measured against, so all three axes get it. Gather every source that exists; they compose (a PR body usually links the issue):

1. **What the user wrote in the prompt.** If the user described the intent when invoking the review ("this should make the wizard hide advanced fields"), that *is* the spec. It is the freshest statement of intent and it wins any conflict.
2. **The PR.** If the target is a PR, or the current branch has one: `gh pr view <N> --json title,body,baseRefName` (the `gh` CLI is installed). Follow whatever issue the body links.
3. **The issue.** References in the commit messages, branch name, or PR body (`#123`, `Closes #45`): `gh issue view <N>`. If `gh` fails, the GitHub REST API works too.
4. **A spec file** under `docs/`, `specs/`, or `.scratch/` matching the branch name or feature.
5. **Backup: the commit descriptions themselves.** `git log $BASE..<target>` - when nothing else exists, the commit messages state the intent, and the review measures the diff against them.

Because of 5 there is always a spec, so **no axis is ever skipped**. Say in the header which source was used, and if the only source is the commit messages say that too - a finding of the form "the diff does something the commit message never mentions" is weaker evidence than one against a real issue, and the report should let the reader see that.

## 5. The shared context pack

Every sub-agent prompt carries the same base pack, so the three of them see one identical picture:

- The **resolved SHAs**, never symbolic refs - `$BASE` as a full SHA and the target likewise - so the agents cannot drift apart if a ref moves mid-review.
- The exact diff command, the commit list, and the paths of any untracked files to read in full.
- The spec from step 4 (fetched text, not just a link).
- The instruction to read the actual files around each hunk - a diff hides the context a real finding needs.
- "Report at most your strongest findings, most severe first. Under 400 words. No praise, no summary of what the change does - only findings. Cite locations as relative markdown links, e.g. [app.py:120](app.py#L120), and quote the offending lines."

Spawn all three with the Agent tool as `general-purpose` sub-agents **in a single message**, so they really do run in parallel.

**If the user emphasized one axis** - "check they don't introduce bug or regression", "is this fast enough", "does it follow our rules" - still run all three; they are parallel and the other two cost nothing extra in wall time. The emphasis changes the *report*: the named axis leads, and its agent gets the user's exact wording appended to its brief.

Ownership is exclusive - each axis reports only what it owns, so the same problem is never filed three times:

| Finding | Axis |
| :---- | :---- |
| spec requirement missing, partial, or implemented wrong | Correctness |
| bug, crash, regression, caller left behind, breaks a build flavour | Correctness |
| missing test for new or changed functionality | Correctness |
| breaks a rule in `CONTRIBUTING.md` or in the project memory | Guideline |
| dead code, duplicated code, scope creep, code smell | Guideline |
| complexity, per-item round trips, unbounded RAM at catalogue scale | Resource utilization |

## 6. Axis 1 - Correctness

**The question: will this really work, everywhere, without breaking what already worked?**

Give this agent the context pack plus this brief:

- **Spec fidelity.** Requirements the spec asked for that are missing or partial, and requirements that look implemented but are implemented wrong. Quote the spec line for each.
- **Bugs in the new code.** Wrong conditions, off-by-one, `None`/`KeyError` paths, swallowed exceptions, transactions that commit half a change, resources never closed, error paths that leave state inconsistent, races between Flask and the workers.
- **Regressions in the callers.** This is the part reviewers skip and it matters most. For every function, method, route, DB column, config key, or return shape the diff changed, find **every** call site (`grep -rn "name("` across the repo, templates and JS included) and check each one still holds. A changed return type with one un-updated caller is a confirmed finding, not a suspicion.
- **The build matrix.** The change MUST work on every flavour this project ships - the containers and the frozen native apps alike. They are separate CI jobs, and a change that only works on one of them is broken. The matrix, and the concrete list of what breaks it, are in [.github/skills/PROJECT-RULES.md](../PROJECT-RULES.md); paste that section into this agent's prompt and have it check the diff against every entry.
- **Tests.** `CONTRIBUTING.md` requires automated tests for any new or changed major functionality. New behaviour with no test under `test/` is a finding. So is a test that asserts nothing real.
- Mark each finding **confirmed** (you traced it) or **suspected** (it needs a run to prove). Never report a style opinion here.

## 7. Axis 2 - Guideline

**The question: does this follow the rules this project has already written down?**

Give this agent the context pack plus these sources and this brief:

- **[.github/skills/PROJECT-RULES.md](../PROJECT-RULES.md)** - the standing rules, the documents that carry the rest (`CONTRIBUTING.md`, `docs/ARCHITECTURE.md`, `docs/PARAMETERS.md` against `config.py`, and the rest of `docs/`), and what CI already enforces. A breach of a standing rule is the most severe Guideline finding there is - report it first and quote the rule.
- **Dead code and duplicated code.** Code the diff adds but nothing calls, code the diff orphans (the old path left behind after the new one lands), a copy of logic that already exists elsewhere in the repo, and behaviour added that no spec asked for. Search the repo before claiming something is unused.
- **The smell baseline** below - it applies even where the repo documents nothing. Two rules bind it: **the repo overrides** (a documented repo standard or a memory rule always wins; where it endorses what the baseline would flag, suppress the smell), and **it is always a judgement call** ("possible Feature Envy"), never a hard violation.
- **Skip anything CI already enforces** (`.github/workflows/lint-*.yml`: flake8, ruff, codespell, mypy, LF endings, no-emoji). Those come back as red CI, and spending the word budget on them buries the real findings.
- Distinguish **hard violations** (a written rule, from `CONTRIBUTING.md` or from the project's standing rules) from **judgement calls** (the smell baseline).

### The project's standing rules

They live in [.github/skills/PROJECT-RULES.md](../PROJECT-RULES.md), shared with the `arch-review` skill so the two can never drift. Read it and **paste the standing-rules section into this agent's prompt verbatim** - the sub-agent has no other access to it. That file is itself a snapshot, so check the project memory for updated information and treat what you find there as the authority over both.

Each smell reads *what it is* -> *how to fix*; match it against the diff:

- **Mysterious Name** - a function, variable, or type whose name doesn't reveal what it does or holds. -> rename it; if no honest name comes, the design's murky.
- **Duplicated Code** - the same logic shape appears in more than one hunk or file in the change. -> extract the shared shape, call it from both.
- **Feature Envy** - a method that reaches into another object's data more than its own. -> move the method onto the data it envies.
- **Data Clumps** - the same few fields or params keep travelling together (a type wanting to be born). -> bundle them into one type, pass that.
- **Primitive Obsession** - a primitive or string standing in for a domain concept that deserves its own type. -> give the concept its own small type.
- **Repeated Switches** - the same `switch`/`if`-cascade on the same type recurs across the change. -> replace with polymorphism, or one map both sites share.
- **Shotgun Surgery** - one logical change forces scattered edits across many files in the diff. -> gather what changes together into one module.
- **Divergent Change** - one file or module is edited for several unrelated reasons. -> split so each module changes for one reason.
- **Speculative Generality** - abstraction, parameters, or hooks added for needs the spec doesn't have. -> delete it; inline back until a real need shows.
- **Message Chains** - long `a.b().c().d()` navigation the caller shouldn't depend on. -> hide the walk behind one method on the first object.
- **Middle Man** - a class or function that mostly just delegates onward. -> cut it, call the real target direct.
- **Refused Bequest** - a subclass or implementer that ignores or overrides most of what it inherits. -> drop the inheritance, use composition.

## 8. Axis 3 - Resource utilization

**The question: does this still hold at a million songs? At ten million?**

That is the bar, not a stretch goal - see *The scale bar* and *Idle RAM* in [.github/skills/PROJECT-RULES.md](../PROJECT-RULES.md), and paste both into this agent's prompt. Give it the context pack plus this brief:

- **State the complexity.** For each changed loop, query, and pipeline, give the big-O in n = number of songs. Anything O(n^2) or worse on a song-scaled collection is a finding on its own. Watch for nested loops over songs, pairwise similarity computed in Python, `x in some_list` inside a per-song loop (a set is O(1)), sorting the whole catalogue to take the top 20.
- **Per-item round trips.** A DB query, a mediaserver API call, an embedding load, or a `task_status` write **inside a per-song or per-album loop** is an N+1 and fails at a million. It must be one batched query, or one whole-catalogue fetch per server.
- **Unbounded materialization.** `cursor.fetchall()`, `list(...)`, or a DataFrame over a song-scaled result holds the entire catalogue in RAM. It must be a server-side/named cursor, keyset pagination with `LIMIT`, or a generator that streams. Ten million rows times anything is the failure mode - say what the RAM cost is per row and multiply.
- **Idle RAM must go back. This is the rule this project is judged on.** When the worker, the Flask app, or the Postgres container is **idle**, each one must release as much RAM as it possibly can. Memory held after the work is done is a finding even when the peak is fine and the code is fast. Concretely:
  - every large index, map, tree, model, or embedding cache loads **lazily on first use** and **unloads after an idle timeout**; anything loaded at import or at process start is wrong;
  - results, DataFrames, and batch buffers are dropped as soon as the job ends, never parked on a module-level global or a long-lived singleton;
  - a worker returns to its baseline RSS after a job completes, and Flask returns to baseline after a request;
  - on the Postgres side, bulk writes clean up after themselves - temporary tables dropped, large result sets not held open, bloat reclaimed - so the container's memory does not stay high once the work is over.
  - **If a new functionality does not release its RAM at idle, say so explicitly in the review**, even if nothing else about it is wrong.
- **SQL.** A new `WHERE`, `JOIN`, or `ORDER BY` column on a song-scaled table with no index; `SELECT *` dragging embedding blobs the caller never reads; an `IN (...)` list built from a song-scaled Python list; a query with no `LIMIT` feeding a UI; a full-table scan or a `VACUUM`-shaped operation on a request path.
- **Worker and queue shape.** One queue job per song where a batch would do; progress or status writes per song hammering the task queue; work that blocks the Flask request thread instead of going to a worker.
- **Propose the fix, concretely** - the batched query, the cursor, the index, the streaming rewrite - not just "this is slow".
- **Do not report micro-optimizations.** A constant-cost inefficiency that does not grow with n is not a finding here. Only things that scale.

## 9. Aggregate

Open with a header stating what was actually compared, so the baseline is never in doubt:

```
Base:   origin/main @ a1b2c3d (fetched just now)
Target: working tree (staged + unstaged + 2 untracked)
Scope:  3 commits, 11 files
Spec:   issue #842 (fetched) / commit messages only
```

Flag here any guard from step 3 that fired: main moved ahead, target already merged, unstaged-only comparison, PR base branch.

Then present the three reports under `## Correctness`, `## Guideline`, and `## Resource utilization`, verbatim or lightly cleaned. Do **not** merge or rerank findings across axes - the separation is the point (see *Why three axes*).

End with a one-line summary: findings per axis, and the worst issue *within each axis*. Don't pick a single winner across axes - that's the reranking the separation exists to prevent.

If the user asked for fixes, apply them only after presenting the report, and never touch a file the diff didn't already touch without saying so.

## 10. Worked example

A real invocation:

> starting from this commit 0a8352c7151d24ea7f0de2f900403aa3987dca14 to the last one. They are Ram and Postgresql space cleaning. Check that they don't introduce bug or regression

resolves like this (the SHAs and counts are a snapshot from when this example was captured - always re-resolve them at run time):

- **Baseline.** The user named a starting commit and "starting from" is inclusive, so the base is its parent: `git rev-parse 0a8352c7^`. Cross-check that against `origin/main` after the mandatory fetch. When the two are the same commit, the readings agree and the header has nothing to flag - that was the case when this example was captured. When they differ, the step 3 guards decide and the header says which fired: main has moved ahead, or the work has since been merged into main (which is what happens to this very example once its PR lands).
- **Target.** "to the last one" = the working tree: every commit after the base **plus everything staged and unstaged right now**, plus untracked files.
- **Diff.** `git diff 7df158c0` (no end ref), then `git status --porcelain` for the untracked files to read in full. Both traps are live in this example: `git diff 0a8352c7..HEAD` shows 6 files, `git diff 0a8352c7^..HEAD` shows 9, and `git diff 0a8352c7^` shows 36 plus 4 untracked. Stopping at `HEAD` would have left 27 changed files and 4 new ones completely unreviewed.
- **Spec.** Source 1 - the user wrote it in the prompt: "RAM and PostgreSQL space cleaning". That is what the diff is measured against. The three commit messages (source 5) back it up, since no issue is referenced.
- **Emphasis.** "check they don't introduce bug or regression" is the Correctness axis. All three still run; Correctness leads the report and gets that sentence verbatim in its brief.
- **What the axes then do.** Correctness traces every caller of the touched helpers, checks the cleanup paths for half-committed transactions, and asks whether it holds in the frozen native builds as well as in the containers. Guideline checks `CONTRIBUTING.md` and the standing rules on point for this change - never tamper with the allocator for anything RAM-shaped, never add a table for anything Postgres-shaped - after checking the project memory in case either has moved. Resource utilization asks whether the cleanup itself scales - a vacuum or a blob reclaim that walks every row must not run per request at 10 million songs - and, since this change is *about* memory, whether the worker and Flask actually drop back to baseline RSS once it is idle.

## Why three axes

A change can pass one axis and fail another, and a single reviewer holding all three questions at once always drops two of them:

- Code that follows every rule and runs fast but implements the wrong thing -> **Guideline pass, Resource pass, Correctness fail.**
- Code that does exactly what the issue asked, cleanly, and loads all 1M embeddings into RAM to do it -> **Correctness pass, Guideline pass, Resource fail.**
- Code that is correct and fast but adds a second config default, a docstring inside a function, and a duplicate of an existing helper -> **Correctness pass, Resource pass, Guideline fail.**

Reporting them separately stops one axis from masking another.
