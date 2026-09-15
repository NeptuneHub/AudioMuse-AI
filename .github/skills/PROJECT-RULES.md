# Project rules and review context

Shared reference for the review skills in this directory - `code-review` and `arch-review`. Both read this file and paste it into every sub-agent prompt, so the two can never drift apart. Add a rule here once and both skills get it.

**This file is a snapshot. Check the project memory for updated information** - the memory is the authority, it changes as the user makes new decisions, it records *why* parts of the code look the way they do (so you don't "fix" something deliberate), and anything found there outranks what is written below.

## Standing rules

Hard rules, not suggestions. A breach is the most severe finding a review can report, and a proposal that requires one is not a proposal.

**Comments and file shape**
- No docstring and no comment inside any function or class. Behaviour is documented in test names. The only prose is the file header: every `.py` opens with the AGPLv3 `#` block, then a module docstring (one line plus `Main Features:` bullets). `config.py` is the exception where inline comments are expected.
- Imports at the top of the file; move one into a function only when it is genuinely necessary.
- No emoji or non-ASCII in code - it breaks the Windows build. HTML, templates, and frontend JS may use them in moderation. Never an em-dash anywhere; plain `-`.
- Never bundle parameters into a dataclass just to cut a parameter count. Extracting a helper to cut complexity is fine.

**Configuration**
- Each tunable and its default is defined once, in `config.py`, and read as `config.NAME`. Never re-specify a default at the call site.
- The database URL is derived inside `config.py`, never read from the environment; the user configures only the `POSTGRES_*` values.
- `docs/PARAMETERS.md` covers wizard parameters only, stays in sync with `config.py`, and is never reorganized or reordered.

**Database**
- Never add a new table. Solve it inside the existing tables and columns. New indexes are fine.

**Errors and logging**
- In an except handler use `logger.exception("msg")`, never `logger.error(f"...{e}")`.
- Never send a traceback or stack trace to the frontend - a generic message pointing at the container logs.
- Failure logs stay loud: no warn-once, no dedup, no rate-limiting on them.

**Routes and network**
- Never add a debug, test, or backdoor route.
- Never tighten outbound URL or IP restrictions. This is homelab software; LAN and loopback targets must keep working.

**Media servers, batch work, and data movement**
- Every batch or cron task runs against all servers, sequentially - never just the current one.
- One whole-catalogue fetch per server. Never a per-id fetch loop.
- Fix scale and orchestration problems at the migration/orchestration layer, not inside the per-provider fetch functions.
- Never return an internal `fp_` id from an API. Always the target server's provider id.
- Backup and restore carry everything. Never filter, exclude, or neutralize anything on either side.

**Memory and deployment**
- Any large in-RAM index, map, tree, model, or embedding cache loads lazily on first use and unloads after an idle timeout.
- Never tamper with the allocator (arena tuning and similar). Reclaim memory by freeing it properly.
- Never work around an application problem by editing `deployment/*.yaml`, and never touch CPU, RAM, or storage quotas there. Fix it in the code.

**UI**
- Every page carries a scope indicator: catalogue-wide versus per-server.
- Restart flows are a countdown and then a redirect. Never poll a status endpoint waiting for the app to come back.

**Tests and repo hygiene**
- LF line endings everywhere; CI fails on CRLF.
- Integration tests drive the real system - real Postgres, real transactions, real DB state - not mocks.
- New or changed major functionality ships with automated tests under `test/`.

## Documents that carry the rest

- `CONTRIBUTING.md` - especially *Linting, Code Style and test* and *PR Requirements*: snake_case naming, imports at the top, one config default per tunable, plain-ASCII `.py`, small focused changes, automated tests for new functionality. Its **Codebase Map** table says what each path is for.
- `docs/ARCHITECTURE.md` - component responsibilities (Flask container, worker container, `taskqueue/`, Postgres, media servers) and the data flows. This is the project's own vocabulary; name things the way it names them.
- `docs/PARAMETERS.md` - wizard parameters, must stay in sync with `config.py`.
- `docs/ERROR_CODES.md`, `docs/MULTI_SERVER.md`, `docs/PLUGIN.md`, and the rest of `docs/` for the area being touched.
- `.github/workflows/lint-*.yml` - what CI already enforces (flake8, ruff, codespell, mypy, LF endings, no-emoji). **Never spend a finding on these**; they come back as red CI and they bury the real findings.

## The build matrix

Every change and every proposal must work on all of it. These are separate CI jobs, and something that only works on one of them is broken:

- **Containers** - `Dockerfile` for CPU intel+arm (`requirements/cpu.txt` + `common.txt`), nvidia GPU (`gpu.txt`), nvidia ARM (`gpu-arm64.txt`); `Dockerfile-noavx2` on ubuntu 22.04 for CPUs without AVX2 (`cpu-noavx2.txt` + `common-noavx2.txt`).
- **Native apps** - `scripts/standalone/build.py --platform {windows,macos,linux}`, with `requirements/{windows,macos,linux}.txt` and the launchers under `native-build/{windows,macos,linux}/`.

What breaks it, concretely: a dependency or import added to only some `requirements/` files (`test/unit/test_requirements_alignment.py` pins this); POSIX-only calls (`os.fork`, `fcntl`, `signal.SIGKILL`, `/tmp`, symlinks) not guarded by `hasattr` or `sys.platform`; hardcoded path separators; PyInstaller-hostile code in a frozen build (dynamic imports, data paths built from `__file__`, multiprocessing without a frozen-safe start method); anything needing AVX2 or a recent CPU; a CUDA path with no CPU fallback; embedded-Postgres assumptions that only hold in the container.

## The scale bar

The catalogue is **1 million songs, and it must not fall over at 10 million**. Every operation whose cost grows with the number of songs, albums, artists, or embeddings must be implemented so it scales: no per-song round trips, no unbounded materialization, no O(n^2) on a song-scaled collection, indexes on song-scaled query columns.

## Idle RAM

When the worker, the Flask app, or the Postgres container is **idle**, each must release as much RAM as it possibly can. Memory held after the work is done is a defect even when the peak is fine and the code is fast: caches load lazily and unload on an idle timeout, buffers are dropped when the job ends rather than parked on a module-level global, a worker returns to baseline RSS after a job and Flask after a request, and bulk writes clean up after themselves on the Postgres side.

## The process split

Flask, the workers, and Postgres are separate processes and usually separate containers. Moving code across that line changes what runs where, what holds RAM, and what happens on restart. Anything that crosses it must say so explicitly.
