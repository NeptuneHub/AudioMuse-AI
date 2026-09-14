---
name: code-review
description: Fast, project-specific review of a target - the working tree (default), a commit, a range, a branch, or a PR - against the head of remote main. Hunts regressions in callers, task-queue and multi-server contract breaks, build-matrix breaks, 1M-song scale and idle-RAM defects, and breaches of this project's hard rules. Output is ONE table - change requested, priority, risk of regression. Use when the user wants a commit, branch, PR, or work in progress reviewed.
---

A review that is worth reading is short, verified, and about what really breaks in this project. Skip anything a linter, CI, or a generic style guide would say.

## 0. Ground rules

- **Read-only.** Never edit, stage, commit, stash, checkout, reset, or rebase. `git fetch` is the only write allowed. Scratch files go in the session scratchpad, never the repo.
- **Other agents may be editing the tree live.** Review what is there; never revert it.
- **Rules authority:** the project memory (if present) > [.github/skills/PROJECT-RULES.md](../PROJECT-RULES.md) > `CONTRIBUTING.md`. Read PROJECT-RULES.md once, before reviewing.

## 1. Resolve the diff (one pass, no deliberation)

```sh
git fetch --quiet origin main
BASE=$(git rev-parse origin/main)
```

`origin` is always `NeptuneHub/AudioMuse-AI`. Never pick another remote. If `origin/main` does not resolve, stop and ask.

| User gave | Diff |
| :---- | :---- |
| nothing | `git diff $BASE` (committed + staged + unstaged) + untracked files |
| "staged" | `git diff --cached $BASE` |
| "unstaged" | `git diff` (against the index - say so) |
| "from X" / "from X to the last one" | `git diff X^` - X is INCLUSIVE, the end is the working tree, not HEAD |
| "since X" | `git diff X` |
| a ref alone | `git diff $BASE...<ref>` |
| `A..B` | `git diff A..B` |
| PR number | `gh pr diff <N>` (base = the PR base branch) |
| "against HEAD" | `git diff HEAD` - only when explicitly asked |

Then, in the same shell call:

- `git status --porcelain` - read every `??` file in full; no diff shows them. Never `git add -N`.
- `git diff --stat <same args>` - size and file list. Empty diff: say so and stop.
- If main moved past the target (`git rev-list --count <target>..$BASE` > 0), use the three-dot form so main's new commits do not show up as deletions.
- **Intent** = what the user wrote > PR body / linked issue (`gh pr view`, `gh issue view`) > commit messages (`git log $BASE..<target> --oneline`). Do not go hunting further.

## 2. Pick the speed path

- **Up to ~1500 changed lines:** review inline, yourself. No sub-agents - they start cold and re-read everything.
- **Larger:** split the FILES (not the checklist) into at most 3 groups by area (e.g. `taskqueue/`+`tasks/`, `app*.py`+`templates/`+`static/`, the rest), spawn one parallel sub-agent per group in a single message, give each the diff command with resolved SHAs, its file list, the intent, PROJECT-RULES.md pasted verbatim, and sections 3-5 of this skill. Each returns rows for the table only.

In both paths: read the surrounding code of every hunk, not just the hunk.

## 3. What to check - in this order, highest yield first

**A. Regressions in callers (most real bugs live here)**
- For every changed function, method, route, SQL column, config key, JSON key, or return shape: grep ALL call sites - Python, `templates/`, `static/` JS, `plugin/`, `native-build/`, `scripts/`, `test/`. One caller left behind is a confirmed finding.
- A removed or renamed import/re-export: the unit suite never imports `app.py`, the worker entry points, or the native launchers, so it stays green while the container dies at boot.
- A status or state column gains or changes a writer: census EVERY writer of that column repo-wide, not just the one in the diff.

**B. Project contracts that are easy to break**
- **Task queue:** a task returns a summary dict (its `message` is the dashboard recap) or raises. `taskqueue.TaskFailed` = never retried, `taskqueue.TaskCancelled` = revoked, anything else = retried. The queue writes the terminal row - a task writing its own terminal status is a bug. Cancel is GLOBAL and only one batch task runs at a time (409): never scope either. Connectivity classification must never match the `OperationalError` base class (a `QueryCanceled` would requeue forever).
- **Multi-server:** every batch/cron task runs against ALL servers, sequentially. `track_server_map` is N:1 - several files of one song share one canonical id, so a dict keyed by canonical id silently drops duplicates. Never an `fp_` id in an API response. One whole-catalogue fetch per server, never per id.
- **Config:** every tunable + default once in `config.py`; `os.environ.get('X', default)` outside config.py is forbidden (`test_config_centralization.py`). Every persistable parameter is written to `app_config` on first start, so a changed default only reaches fresh installs, and a per-install path belongs in `SETUP_BOOTSTRAP_EXCLUDED_KEYS`. Only None/bool/int/float/str (or exact-JSON list/dict) values are persistable.
- **Backup/restore** carries everything - any filter, exclusion, or neutralization on either side is a finding.
- **UI:** restart = countdown then redirect (never polls); every page shows catalogue vs per-server scope; index coverage shows bands, never percentages; no traceback to the frontend.
- **AI:** prompts stay general (intelligence in tool schemas/enums), never overfit to a test query; naming never sends an output-token cap.

**C. Build matrix** - containers (CPU intel/arm, nvidia, nvidia-arm) and native Windows/macOS/Linux (PyInstaller).
- POSIX-only calls (`os.fork`, `fcntl`, `signal.SIGKILL`, `/tmp`, symlinks) unguarded. The fork decision is `hasattr(os, 'fork')` only - Windows runs jobs inline.
- Frozen-build hostility: dynamic imports, data paths from `__file__`, multiprocessing without a frozen-safe start.
- A dependency added to only some `requirements/` files (`test_requirements_alignment.py`). CUDA with no CPU fallback. Non-ASCII in `.py`.
- The noavx2 image (`Dockerfile-noavx2`, `*-noavx2.txt`) is frozen legacy: never request a change there.

**D. Scale (1M songs, must survive 10M) and idle RAM**
- Per-song/album DB query, API call, or `task_status` write inside a loop. O(n^2) on songs. `x in list` in a per-song loop. Sorting the catalogue for a top-k.
- `fetchall()` / `list()` / DataFrame over a song-scaled result; `SELECT *` pulling embedding blobs; no index on a new song-scaled `WHERE`/`JOIN`/`ORDER BY`; no `LIMIT` feeding a UI.
- Big index/map/model/cache loaded at import or kept on a module global: it must load lazily and unload on idle. Indexes store via `config.IVF_STORAGE_DTYPE` + `ivf_quant`; local index caches are never persistent.
- Give the fix concretely (batched query, named cursor, index). Constant-cost inefficiency is not a finding.

**E. Hard rules and tests**
- Standing rules from PROJECT-RULES.md that CI does NOT catch: docstring/comment inside a function or class, file header shape, `logger.exception` in handlers, loud failure logs (no warn-once), new DB table, debug route, tightened URL/IP restriction, `deployment/*.yaml` edits, allocator tuning, dataclass to cut a param count, imports not at top.
- Dead code the diff orphans (the old path left behind) or a duplicate of an existing helper - search before claiming.
- New/changed behaviour with no test, or a test that would still pass with the fix reverted. Integration tests must hit real Postgres; in CI all integration modules share ONE database, so a test asserting on state it does not own (e.g. absence of a `public.*` table) is flaky. A feature flag defaulting off means its tests must enable it.

**Never report:** anything flake8/ruff/mypy/codespell/LF/no-emoji CI catches, style opinions, generic code smells, micro-optimizations, pre-existing issues the diff does not touch, or any "fix" that breaks a rule above (SSRF blocking, noavx2 edits, debug routes, deployment quotas, `MALLOC_ARENA_MAX`).

## 4. Verify before reporting

Every row must survive one re-read of the actual code: name the trigger (input/state -> wrong result). If it cannot be named, drop it. Merge duplicates. No cap on count, but no filler - an empty table is a valid result.

## 5. Output - ONE table, nothing else

One scope line, then the table, most important first. No prose sections, no praise, no summary.

```
Base: origin/main @ a1b2c3d | Target: working tree (+2 untracked) | 11 files | Intent: issue #842

| # | Change requested | Priority | Regression risk |
| :-- | :---- | :---- | :--: |
| 1 | [app_helper.py:120](app_helper.py#L120) - `get_tracks` now returns a dict but `app_map.py:88` still indexes it as a list; update the caller | Critical | 2 |
```

- **Change requested:** clickable relative link, the defect in one clause, the concrete change to make. One or two sentences.
- **Priority:** `Critical` (data loss, crash, broken build flavour, hard-rule breach that ships a bug) - `High` (wrong behaviour on a real path, hard-rule breach, missing test for new behaviour) - `Medium` (edge-case bug, scale defect that bites at 1M, orphaned code) - `Low` (worth doing, no user impact today).
- **Regression risk (1-10):** how likely APPLYING the requested change breaks something else. 1 = local, fully covered by tests. 5 = touches shared code or several callers. 10 = changes a cross-process contract (queue, config persistence, DB schema, backup/restore) with thin test coverage.

If nothing survives verification, output the scope line and a single row: `| - | No changes requested | - | - |`.

If the user asks for fixes afterwards, apply only the rows they name, and never touch a file outside the diff without saying so.
