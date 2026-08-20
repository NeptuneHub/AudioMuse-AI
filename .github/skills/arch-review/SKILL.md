---
name: arch-review
description: Scan the codebase for architectural friction and return an action plan - a ranked list of deepening opportunities, each scored for regression risk from 1 to 10, ordered by the value it adds. Refactors that turn shallow modules into deep ones, for testability and AI-navigability. Output is a markdown table plus pointed lists, never HTML and never a written report file. Use when the user asks where the codebase is getting hard to change, for refactoring opportunities, or for an architecture review.
disable-model-invocation: true
---

Surface architectural friction and propose **deepening opportunities** - refactors that turn shallow modules into deep ones. The aim is testability and AI-navigability, and the deliverable is an **action plan you can work through in order**.

## 0. Ground rules

- **The output is a plan, not a change.** This skill reads, scores, and reports. It never edits, stages, commits, or refactors anything. Applying an item is a separate, later decision by the user.
- **No HTML. Ever.** No report file, no temp file, no browser. The plan is markdown - one table plus pointed lists - written directly in the reply.
- **Nothing lands in the repo.** If a scratch note is genuinely needed, it goes in the session scratchpad directory, never in the working tree.
- **Other agents may be editing this working tree live.** Read what is there, change nothing.
- **No new files, no new tables, no new services** in any proposal unless the constraints in step 3 actually allow it.

## 1. Vocabulary

Use these words exactly. Do not drift into "component", "service", "API", or "boundary" - they are vague enough to hide the very problem being described.

- **Module** - a unit with an interface and an implementation behind it. A file, a package, a class.
- **Interface** - everything a caller must know to use the module: signatures, required call order, error modes, side effects, the state it expects.
- **Depth** - the ratio of functionality to interface. A **deep** module hides a lot behind a small interface. A **shallow** module has an interface nearly as complex as its implementation, so it costs almost as much to use as to reimplement.
- **Seam** - a place where behaviour can be replaced without editing callers. Real seams have more than one implementation; a seam with one implementation is hypothetical.
- **Adapter** - the implementation that sits behind a seam and speaks to one specific outside thing (a media server backend, a database, a model runtime).
- **Locality** - how much of one behaviour lives in one place. High locality means a change to that behaviour touches one module.
- **Leverage** - how much future change one refactor makes cheaper. Leverage is why a hot spot beats a tidy backwater.

Three principles carry most of the weight:

- **The deletion test.** Delete the module in your head. Does the complexity **concentrate** somewhere sensible, or does it just **scatter** into the callers? "Concentrates" means the module was shallow and the refactor is real. "Scatters" means it was doing real work - leave it alone.
- **The interface is the test surface.** If a module can only be tested by reaching around its interface, the interface is wrong. Tests that need internal state are describing a design problem, not a testing problem.
- **One adapter is a hypothetical seam, two is a real one.** An abstraction with a single implementation is speculative generality until a second one exists.

## 2. Scope before you scan

YAGNI applies to reviews too. Deepening pays off where change keeps landing, so decide where to look **before** looking.

- **If the user named a direction** - a module, a subsystem, a pain point - take it and skip the inference.
- **Otherwise find the hot spots.** Walk back a good stretch of history and let the files that keep coming up pull your attention first:

```sh
git log --since="6 months ago" --pretty=format: --name-only -- '*.py' | grep . | sort | uniq -c | sort -rn | head -30
```

  If the changes are scattered with no clear hot spot, widen the net and fall back on size plus coupling (`git ls-files '*.py' | xargs wc -l | sort -rn | head -20`).

**Read the documents listed in [.github/skills/PROJECT-RULES.md](../PROJECT-RULES.md) first**, so the proposals speak the project's own language rather than inventing one. `docs/ARCHITECTURE.md` is the load-bearing one here: it names the components and the data flows, and a proposal should name modules the way it names them.

The layout at a glance: `app*.py` and `database.py` at the root are Flask and data access, `tasks/` is the core logic hub (with `tasks/mediaserver/` dispatching per backend and `tasks/ai/` holding the AI stack), `taskqueue/` is the Postgres-backed queue, `native-build/` and `scripts/standalone/` build the desktop apps, `test/` is unit plus integration.

## 3. Constraints that bound every proposal

A proposal that breaks one of these is not a candidate. **Drop it before it reaches the plan** - do not surface it as a trade-off.

Everything in [.github/skills/PROJECT-RULES.md](../PROJECT-RULES.md) binds a proposal exactly as it binds a code change - it is the same file the `code-review` skill uses, so the two skills can never disagree. Read it in full and paste it into every sub-agent prompt. In particular:

- **The standing rules.** Never a new database table, one config default defined once in `config.py`, no docstrings or comments inside functions, never an internal `fp_` id out of an API, batch work against all servers, caches lazy-loaded and idle-unloaded, never a fix in `deployment/*.yaml`. A refactor that needs one of these broken is not a candidate.
- **The build matrix.** Containers (CPU intel+arm, nvidia, nvidia-arm, noavx2) *and* the frozen native windows, macos, and linux apps. A refactor that assumes `os.fork`, a POSIX path, a live network, or an unfrozen import system is not portable here.
- **The scale bar.** 1 million songs, not falling over at 10 million. A proposal that buys elegance with a per-song round trip is a regression, not a candidate.
- **Idle RAM.** Whatever the new shape is, the worker, Flask, and Postgres must still drop back to baseline when idle.
- **The process split.** Moving code across the Flask / worker / Postgres line changes what runs where, what holds RAM, and what happens on restart. Never do it silently - a proposal that crosses it says so, in those words.

That file is a snapshot; **check the project memory for updated information**, and treat what you find there as the authority - it also records the decisions the user has already made and does not want re-litigated, which is what stops this skill from proposing the same rejected refactor twice.

## 4. Explore

Spawn `general-purpose` sub-agents to walk the code, one per area you scoped, **all in a single message** so they run in parallel. Give each the vocabulary from step 1, the constraints from step 3, and its area.

Do not follow rigid heuristics - explore organically and note **where you experience friction**:

- Where does understanding one concept require bouncing between many small modules?
- Where is a module shallow - interface nearly as complex as the implementation?
- Where were pure functions extracted just for testability, while the real bugs hide in how they are called (no locality)?
- Where do tightly-coupled modules leak across their seams - a caller that must know the callee's internal order, state, or error shape?
- What is untested, or hard to test through its current interface?

And the shapes this codebase produces in particular:

- The same rule implemented once in Flask and again in a worker, free to drift.
- Logic duplicated across the media server backends where the dispatcher should hold it once - or a seam with only one adapter behind it.
- A task reaching into `database.py` internals instead of calling one named operation.
- Analysis, clustering, and index-building each carrying their own copy of batching, progress reporting, or cancellation.
- A module that only differs per platform buried in an `if sys.platform` cascade rather than sitting behind a seam.

Apply the **deletion test** to everything you suspect is shallow, and report its result per candidate. "Concentrates" is the signal you want; "scatters" means drop the candidate.

## 5. Score every candidate

Two scores, both required, both justified by naming the factors that fired. No unscored item reaches the plan.

### Value added, 1 to 10 - this sets the order

Built from: how often the area changes (a hot spot has leverage, a backwater does not), how much friction the refactor removes, how much testability it buys, and whether it unblocks other work.

| Score | Anchor |
| :---- | :---- |
| 9-10 | A top hot spot. Removes a whole class of recurring bug, or makes a currently untestable area testable through its interface. Other work is waiting on it. |
| 7-8 | Frequently touched. Cuts real friction - fewer files to hold in your head for a routine change - and improves the test surface. |
| 5-6 | Real but local. One module gets clearer; nothing else changes. |
| 3-4 | Mostly tidiness. Nice to have, no leverage. |
| 1-2 | Speculative. Serves an imagined future need. Say so, and put it last. |

### Regression risk, 1 to 10 - this is the warning label

Raise the score for each factor that applies, and **list the factors** that produced it:

- number of call sites touched (count them, do not estimate)
- crosses the Flask / worker / Postgres process split, or changes what runs where
- touches SQL, a query on a song-scaled table, or anything schema-shaped
- touches the media server adapters, so it can break one backend and not the others
- touches packaging, `requirements/`, or the frozen native builds
- thin or absent test coverage in the area today
- user-visible behaviour: playlists, ids, API responses, the UI

| Score | Anchor |
| :---- | :---- |
| 1-2 | Rename or move inside one module, every call site in the same file, area well covered by tests. |
| 3-4 | One module plus fewer than ten call sites, all in Python, tests exist and would catch a mistake. |
| 5-6 | Several modules or a shared helper; partial coverage; a mistake shows up in one feature. |
| 7-8 | Crosses the process split, or touches SQL, the media server adapters, or the build path. Thin coverage. A mistake reaches users. |
| 9-10 | Schema, id semantics, packaging, or anything users can see immediately. Needs a migration, a staged rollout, or a rebuild of every flavour to verify. |

## 6. The action plan - the deliverable

Order **by value added, descending**. Break ties by lower regression risk first. Then output exactly this, inline in the reply:

**The plan table**, one row per item:

| # | Opportunity | Modules | Value | Risk | Confidence | First step |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | short imperative name | the files involved | 1-10 | 1-10 | Strong / Worth exploring / Speculative | the smallest safe first move |

**Then one block per item, in the same order**, as a pointed list - no prose paragraphs:

- **Files** - the modules involved, as clickable relative links.
- **Problem** - the friction, in the vocabulary from step 1. Where does the interface cost as much as the implementation?
- **Solution** - plain English. What moves where, what the deepened module's interface becomes, what sits behind the seam.
- **Benefit** - stated as locality and leverage, plus what the test surface becomes: which tests get simpler, which become possible at all.
- **Deletion test** - concentrates or scatters, and where.
- **Value <n>/10** - the factors behind the number.
- **Regression risk <n>/10** - the factors behind the number, with the call-site count.
- **Test plan** - what must be green before starting, what to add, and how to prove no regression across the build matrix.
- **Constraint check** - which standing rule, build flavour, or scale limit this brushes against, and why it still passes.

**Then close with:**

- **Sequencing** - which items unblock others, which pairs touch the same call sites and must not be done in parallel, and any item that should wait for a test to exist first.
- **Top recommendation** - the one item to do first and why, in two sentences. Usually the best value-to-risk ratio, not simply the highest value.

If a candidate contradicts a decision the user has already made, surface it **only** when the friction is real enough to justify reopening it, and mark it plainly: "contradicts the standing rule on X - worth reopening because...". Do not list every refactor the rules forbid.

## 7. After the plan

Ask which item the user wants to explore, then go deep on that one alone: the constraints it really has, the dependencies it drags in, the exact interface of the deepened module, what sits behind the seam, which existing tests survive the change and which have to be rewritten. Still no code changes unless the user asks for them.

If the user rejects an item for a **load-bearing** reason - one a future review would need in order not to re-suggest the same thing - offer to record it in the project memory: "Want me to save this so future architecture reviews don't propose it again?" Skip ephemeral reasons ("not now") and self-evident ones.
