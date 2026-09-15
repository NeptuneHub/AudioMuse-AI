---
name: sec-review
description: Security review of this codebase - reasons about the code like a security researcher, tracing untrusted input from entry point to sink rather than pattern-matching. Scans the whole tree at HEAD for hardcoded secrets (anything beyond the empty defaults config.py declares), then checks what the current change introduces against the head of remote main. Uses the OWASP Top Ten as the primary risk source, reports severity plus confidence in a markdown table, and proposes fixes without ever applying them. Use when asked whether the code is secure, to audit for vulnerabilities, to check for injection, auth, secrets, or dependency risk, or to review a change for security regressions.
---

Security review in four axes, each running as **its own sub-agent**, in parallel:

- **Secrets and exposure** - is any credential in the tree that should only ever come from the environment?
- **Injection and untrusted data flow** - can input a user or a model controls reach a dangerous sink?
- **Authentication and access control** - can something be reached, or done, by someone who should not?
- **Configuration, dependencies, and supply chain** - is the way this is built and shipped safe?

The primary source of risk is the **OWASP Top Ten**: <https://owasp.org/www-project-top-ten/>. Every finding carries its OWASP category.

## 0. Ground rules

- **Read-only, and nothing is auto-applied.** Fixes are *proposed* in the report. The user applies them. Say so explicitly at the end: "Nothing has been changed. Review each fix before applying."
- **No HTML.** The report is markdown - tables plus pointed lists - written directly in the reply. No report file, no temp file, no browser.
- **Never print a secret.** A report that quotes the token leaks it a second time, into the transcript and anywhere the report is pasted. Report the file, the line, the variable name, the length, and at most the first 4 characters. Never the value.
- **Proposed fixes obey the project's rules.** In particular: **no comment explaining the fix inside the code** - this project forbids comments and docstrings inside functions, so the explanation lives in the report and only in the report. The rest of the rules, the build matrix, and the scale bar are in [.github/skills/PROJECT-RULES.md](../PROJECT-RULES.md); paste it into every sub-agent prompt. A fix that breaks a standing rule is not a fix.
- **Nothing lands in the repo.** Scratch notes go in the session scratchpad directory, never the working tree.
- **Other agents may be editing this working tree live.** Read what is there, change nothing.

## 1. Scope - two of them, both required

**Scope A: the whole tree at HEAD.** Secrets, dependencies, and configuration are scanned across the **entire codebase as it stands right now**, including uncommitted and untracked work - not just the diff. A credential committed last year is exposed today.

```sh
git ls-files                       # tracked
git status --porcelain             # plus staged, unstaged, untracked
```

**Scope B: what this change introduces.** Resolve the baseline the same way the `code-review` skill does - `git fetch --quiet origin main`, then `BASE=$(git rev-parse origin/main)`, and `git diff $BASE` for the working tree. The remote is always `origin`, and `origin` is `NeptuneHub/AudioMuse-AI`; never another remote.

Every finding is tagged **introduced by this change** or **pre-existing**. Both get reported - the first is what blocks a merge, the second is what needs scheduling. If the user names a path, it narrows the code scan but **never** narrows the secrets scan.

## 2. Use the OWASP Top Ten as the checklist

Fetch <https://owasp.org/www-project-top-ten/> at run time and work from the list you actually get - the Top Ten is revised periodically and a memorised copy goes stale. Map every finding to its category id.

Use it for **coverage, not just labelling**: walk the categories one by one and state the outcome for each, so "we found nothing in access control" is visibly different from "we never looked at access control".

If the fetch fails, fall back on the categories below, say in the report that the live list was unreachable, and treat this copy as possibly outdated: broken access control; cryptographic failures; injection; insecure design; security misconfiguration; vulnerable and outdated components; identification and authentication failures; software and data integrity failures; security logging and monitoring failures; server-side request forgery.

## 3. Axis 1 - Secrets and exposure

**The rule for this project is exact, so use it exactly.** `config.py` declares every credential as an environment lookup with an **empty default**:

```python
JELLYFIN_TOKEN = os.environ.get("JELLYFIN_TOKEN", "")
AUDIOMUSE_PASSWORD = os.environ.get("AUDIOMUSE_PASSWORD", "")
JWT_SECRET = os.environ.get("JWT_SECRET", "")
```

That is the only acceptable shape. **Any credential-shaped name assigned a non-empty literal, anywhere in the tree, is a finding - `config.py` included.** A non-empty default in `config.py` is not "a default", it is a shipped credential and every deployment that never overrides it shares the same one.

- **Names that count as credential-shaped:** `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, `API_KEY`, `APIKEY`, `CREDENTIAL`, `AUTH`, `SALT`, `PRIVATE_KEY`, `BEARER`, `DSN`, connection strings - plus this project's own `JELLYFIN_TOKEN`, `EMBY_TOKEN`, `NAVIDROME_PASSWORD`, `NAVIDROME_API_KEY`, `PLEX_TOKEN`, `AUDIOMUSE_USER`, `AUDIOMUSE_PASSWORD`, `API_TOKEN`, `JWT_SECRET`.
- **Look everywhere, not just `.py`:** `templates/`, `static/` JS, `Dockerfile*`, `deployment/*.yaml`, `.github/workflows/*`, `scripts/`, `native-build/`, `docs/`, `screenshot/`, and `test/` - a real token in a fixture is still a leaked token.
- **Entropy heuristic** for values that have no obvious name: a literal of 20+ characters mixing case and digits, assigned to a credential-shaped name, put in an `Authorization`/`X-Emby-Token` style header, or passed to a login call.
- **Files that should never be tracked:** `.env*`, `*.pem`, `*.key`, `*.p12`, service-account JSON. Check whether any were ever committed, not just whether they are present now: `git log --all --diff-filter=A --name-only -- '.env*' '*.pem' '*.key'`.
- **Secrets that leak by output, not by storage:** a credential written to a log line, echoed in an API response, rendered into a template, or included in an error path. The project already forbids sending tracebacks to the frontend; a credential in a log is the same class of bug.
- **A secret in the tree is CRITICAL and the fix is rotation.** Deleting the line leaves the value in git history and in every clone. Say "rotate this credential, then remove it", never just "remove it".

**Enumerate files through git, never with a bare recursive grep.** `git ls-files` plus the untracked entries from `git status --porcelain` is the file list. A recursive walk of the working directory drowns in `.venv/`, `__pycache__/`, and vendored model blobs - all gitignored, none of them this project's code - and the first "finding" will be somebody else's library constant.

**These are not secrets.** A name-based match is a candidate, not a finding; open the line before reporting it. Real examples from this tree that must never be reported:

- `SECRET_PLACEHOLDER = '********'` - a masking placeholder, the opposite of a leak.
- `_DEFAULT_TOKENIZER_DIR = '/app/model/...'` - "TOKEN" matched inside "TOKENIZER". Watch for substring collisions.
- `_SHARED_TOKEN = "SELECT shared_token FROM task_status WHERE task_id = %s"` - a SQL constant named after the column it selects.
- Test fixtures that are self-evidently fake (`'integration-test-secret-do-not-use-in-prod'`). A fixture is only a finding when the value could actually authenticate somewhere.

## 4. Axis 2 - Injection and untrusted data flow

Trace from **source** to **sink** across files. Do not stop at the file where the sink lives.

The sources in this codebase:

- HTTP request data on the Flask routes in `app.py` and `app_*.py` - query params, JSON bodies, form fields, headers, path segments.
- **Arguments produced by the AI tool-calling layer** (`tasks/ai/tools.py`, `tasks/ai/tool_impl.py`, `tasks/ai/planner.py`). A model's output is untrusted input, and a plan that reaches a SQL builder or a filesystem path is an injection source like any other.
- Responses from media servers - a track title, an artist name, or a path from Jellyfin, Navidrome, Emby, Plex, or Lyrion is attacker-influenced data if the server is not yours.
- Plugin code loaded under `plugin/`, cron parameters, uploaded or scanned file paths, and anything read back out of the database that arrived from any of the above (second-order injection).

The sinks to reach for:

- SQL - string-built queries anywhere near a source. Parameterized queries are the standard here; an f-string in a `WHERE` clause is a finding even when today's input looks safe.
- Command execution, subprocess calls, and anything that shells out to a build or an external binary.
- Filesystem paths - `..` traversal and absolute-path injection when a source names a file.
- HTML and templates - unescaped rendering into `templates/` or DOM injection in `static/` JS.
- Deserialization - `pickle`, `yaml.load`, and anything reconstructing an object from stored or fetched bytes.
- XML parsing that resolves external entities.

## 5. Axis 3 - Authentication and access control

- **Every new route is a new attack surface.** Auth is controlled by `AUTH_ENABLED`, `AUDIOMUSE_USER`, `AUDIOMUSE_PASSWORD`, `API_TOKEN`, and `JWT_SECRET`, enforced in `app.py` and `app_helper.py`. For every route the diff adds or changes, confirm it goes through the same enforcement as its neighbours. A route that quietly skips the decorator is the single most likely real finding in this codebase.
- **Empty-secret behaviour.** `JWT_SECRET` and `API_TOKEN` default to empty. Check what the code does when they are empty: signing with an empty key, accepting any token, or failing open is CRITICAL.
- Token handling: tokens in query strings or URLs (they land in logs and history), missing expiry validation, algorithm confusion, tokens logged on the way in.
- Object-level authorization: can one user's request act on another user's or another server's data by changing an id? Check the multi-server paths especially, where a `server_id` often arrives from the client.
- State-changing endpoints reachable by `GET`, or without CSRF protection when the UI drives them from a browser session.
- **Exposure of internal identifiers:** an internal `fp_` id in any API response is a rule breach and an information leak - report it.
- Do **not** propose adding a debug, diagnostic, or test route as part of a fix. That is forbidden here.

## 6. Axis 4 - Configuration, dependencies, and supply chain

- `requirements/*.txt` - unpinned versions, a dependency pulled from a git URL or an alternate index, a package with a known CVE, an abandoned crypto or parsing library. CI already runs pip-audit and a bandit gate, so **do not restate what CI already reports**; look for what it misses.
- `Dockerfile` and `Dockerfile-noavx2` - build-time secrets in `ARG`/`ENV`, `ADD` from a URL, running as root when it need not, unpinned base images.
- `.github/workflows/*` - overly broad `permissions:`, `pull_request_target` combined with a checkout of untrusted code, and script injection through `${{ github.event.* }}` interpolated into a `run:` block.
- `deployment/*.yaml` - **read-only**. Report a risk if one exists; never propose editing these files, and never touch resource quotas.
- Native and desktop packaging under `native-build/` and `scripts/standalone/` - where the embedded Postgres data directory lives, what its trust settings are, what ports bind, and whether anything writes a credential to disk in the clear.

## 7. Known false positives - do not spend findings on these

This project has already triaged these. Reporting them again is noise, and one of them is an explicitly rejected change:

- **Never propose restricting outbound URLs or IP ranges, and never propose blocking LAN or loopback targets.** This is homelab software: users' media servers live on private addresses. "The URL can reach a private IP" is not a finding here. An SSRF finding is only real when the risk is something else - for example a user-controlled URL that reaches an internal endpoint *with credentials attached*, or one whose response is reflected back with secrets in it.
- Bandit `B608` (SQL built as a string in queries that are in fact parameterized), `B104` (binding `0.0.0.0` in a container), `B310` (urllib to localhost), and `B615` are triaged false positives in this repo.
- Anything the lint or security workflows already fail on. It comes back as red CI; a finding spent there buries a real one.
- Do not propose sending more diagnostic detail to the frontend - tracebacks stay in the container logs.

## 8. Self-verification pass

Before anything reaches the report, re-read each finding with fresh eyes:

- Is it genuinely reachable? Name the entry point and the path to the sink. If you cannot name one, it is not a finding, it is a suspicion - say so or drop it.
- Is it already handled upstream - by the auth decorator, by `sanitization.py`, by `ssrf_guard.py`, by a parameterized query, by a framework default?
- Is the input actually attacker-controlled, or is it a constant?
- Then assign **severity** and **confidence** separately. A high-severity finding you are unsure about is reported as HIGH severity, LOW confidence - not downgraded to hide the doubt.

| Severity | Meaning | Example in this codebase |
| :---- | :---- | :---- |
| CRITICAL | Immediate exploitation, data or account loss | A real media-server token in the tree; auth bypass on a route; SQL injection reachable from a request |
| HIGH | Serious, exploit path exists | A route missing auth enforcement; a credential written to logs; empty `JWT_SECRET` accepted as valid |
| MEDIUM | Exploitable with conditions or chaining | CSRF on a state-changing endpoint; weak hashing; internal id exposed by an API |
| LOW | Best-practice violation, low direct risk | Verbose error message; missing security header; unpinned dependency with no known CVE |
| INFO | Worth knowing, not a vulnerability | Outdated dependency, no CVE; a hardening opportunity |

## 9. The report

Markdown only. Open with what was actually scanned, so the scope is never in doubt:

```
Tree:   HEAD @ a1b2c3d, 341 python files + templates, static, workflows, deployment
Change: vs origin/main @ e4f5g6h - 11 files
OWASP:  Top Ten fetched from owasp.org
```

Then, in order:

1. **Summary table** - counts by severity, split into *introduced by this change* and *pre-existing*.
2. **OWASP coverage** - one line per category with its outcome, so silence is visible.
3. **Findings table**, ordered by severity then confidence: `# | Severity | Confidence | OWASP | Where | Introduced? | Risk in one line`.
4. **One block per finding**, grouped by category rather than by file:
   - **Where** - clickable relative link with the line number.
   - **What** - the vulnerable code, secrets redacted to name, length, and first 4 characters.
   - **Why it matters** - plain English: what an attacker does with this, and what they get.
   - **Reachability** - the entry point, the path to the sink, and what has to be true for it to fire.
   - **Fix** - before and after, in the project's existing style, with no comment added to the code. Explain the change here in the report instead.
   - **Introduced by this change** - yes or no.
5. **Close with:** "Nothing has been changed. Review each fix before applying."

If nothing is found, say so plainly and list what was scanned and which categories were checked. "No vulnerabilities found" with no scope attached is not a result.
