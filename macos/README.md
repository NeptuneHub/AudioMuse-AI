# AudioMuse-AI — Standalone macOS App

This folder builds AudioMuse-AI into a single double-clickable **`AudioMuse-AI.app`**
with **no Docker, no separately-installed PostgreSQL or Redis**. The app runs as a
menu-bar agent that starts everything for you:

- **Embedded PostgreSQL** via [`pgserver`](https://github.com/orm011/pgserver) (the same
  embedded server already used in this repo's integration tests).
- **Embedded Redis** via a bundled `redis-server` binary.
- The Flask web UI (served by `waitress`) on `http://127.0.0.1:8000`.
- The two RQ workers, the janitor, and the config-restart listener.

The ~2.5 GB of ONNX models (MusiCNN, CLAP, Whisper, gte-multilingual, Silero VAD)
are bundled inside the app, so analysis works fully offline.

All writable state lives in your Library, never inside the (read-only, signed) app:

- Database / Redis / scratch: `~/Library/Application Support/AudioMuse-AI/`
- Logs: `~/Library/Logs/AudioMuse-AI/audiomuse.log` (rotated, capped at ~40 MB total)

## Menu-bar items

- **Open in Browser** — opens the web UI at `http://127.0.0.1:8000`.
- **Pause Server / Start Server** — stops or restarts all embedded services and workers.
- **Open Log** — opens the rotating log in Console.app.
- **Quit** — cleanly shuts down PostgreSQL, Redis and every worker (no orphans).

After first launch, open the UI and configure your media server (Jellyfin, Navidrome,
Lyrion, Emby or MPD) exactly as you would for the container version.

---

## Building (developer machine)

> Build **on the target architecture**: build on Apple Silicon for an arm64 app and
> on an Intel Mac for an x86_64 app. (universal2 is intentionally not used — several
> ML wheels are not universal2.)

### Prerequisites

1. macOS with **Xcode Command Line Tools** (`xcode-select --install`) — provides
   `codesign`, `sips`, `iconutil`, `ditto`.
2. **Python 3.12** (matching the project venv).
3. A **`redis-server`** binary for your architecture, vendored at:
   ```
   macos/vendor/redis/$(uname -m)/redis-server
   ```
   Get one from a Homebrew install (`brew install redis` then copy
   `$(brew --prefix redis)/bin/redis-server`) or build it from source. Make it
   executable: `chmod +x macos/vendor/redis/$(uname -m)/redis-server`.

### Steps

```bash
cd <repo root>
python3.12 -m venv .venv-macos
source .venv-macos/bin/activate
pip install -r requirements/macos.txt

bash macos/build.sh
```

`build.sh` will:
1. Generate `AudioMuse-AI.icns` + the menu-bar icon from `screenshot/audiomuseai.png`.
2. Run PyInstaller against `macos/AudioMuse-AI.spec`.
3. **Ad-hoc sign** every nested binary (Postgres, Redis, dylibs) and then the bundle.
4. Produce `dist/AudioMuse-AI-<arch>.zip`.

The build is **not notarized and not Developer-ID signed** — we have no Apple
Developer account. That is expected; see the next section for how users open it.

---

## Installing & authorizing (end users)

Because the app is **unsigned** (built without a paid Apple Developer account),
macOS Gatekeeper will refuse to open it on first try, and on macOS Sequoia (15+)
the old right-click → Open shortcut no longer works. Use **one** of these:

### Option A — Terminal (recommended, one command)

1. Unzip and move `AudioMuse-AI.app` to `/Applications`.
2. Run:
   ```bash
   xattr -dr com.apple.quarantine /Applications/AudioMuse-AI.app
   ```
   This removes the download "quarantine" flag from the whole app (including the
   bundled Postgres/Redis binaries), so Gatekeeper stops blocking it.
3. Double-click the app. The AudioMuse-AI icon appears in your menu bar.

### Option B — System Settings (no Terminal)

1. Move the app to `/Applications` and double-click it. macOS shows a warning;
   dismiss it.
2. Open **System Settings → Privacy & Security**, scroll to the **Security**
   section, and click **Open Anyway** next to AudioMuse-AI. Authenticate.
3. Launch the app again and confirm.

> The `xattr` command is safe and expected for unsigned open-source apps; it only
> removes the "downloaded from the internet" marker. If you prefer not to run it,
> use Option B.

### First launch notes

- First boot initializes the embedded PostgreSQL data directory and can take a
  little longer; the menu-bar status shows **Starting…** then **Running**.
- If something looks stuck, use **Open Log** to inspect
  `~/Library/Logs/AudioMuse-AI/audiomuse.log`.

---

## Note to the AI (architecture & debugging handoff)

> Read this first if you are an AI assistant picking up this work in a fresh session
> or on another machine. It explains *what* was changed and *why* so you don't have
> to re-derive it. Verify any file/line reference still exists before acting on it.

### The core problem and the chosen strategy

AudioMuse-AI was container-only: a Flask app + RQ workers needing an **external
PostgreSQL** and **external Redis**. The goal was a single double-clickable macOS
`.app` with neither. The blockers were that database access (~897 raw-SQL sites
across ~37 files, leaning on Postgres-only features: `ON CONFLICT`, `RETURNING`,
`JSONB`, `pg_trgm`/`unaccent`, PL/pgSQL triggers, advisory locks,
`information_schema`) and Redis/RQ usage (across ~51 files) are spread everywhere.

**Decision: embed real PostgreSQL + real Redis rather than port to SQLite / a
non-Redis queue.** This keeps the SQL dialect and all RQ semantics (Lua `EVAL`,
pub/sub, job registries, `job.meta`, `Job.fetch`, `send_stop_job_command`)
**byte-for-byte unchanged**. No call site was rewritten — that is the whole point,
and it is how the "no miss anyone" requirement is satisfied structurally.

### The seam (this is the key idea)

Connection/queue construction was already centralized in **one** place
(`app_helper.py`) and config is read from env in **one** place (`config.py`). So the
abstraction is config-driven dispatch in the spirit of `tasks/mediaserver.py`:

- **`database.py`** (repo root) owns `get_db()`/`close_db()`, dispatched on
  `config.DATABASE_TYPE`. It also exposes `start_embedded(data_dir)` /
  `stop_embedded()` (pgserver) used **only by the macOS supervisor**.
- **`taskqueue.py`** (repo root) owns `redis_conn`, `rq_queue_high`,
  `rq_queue_default`, dispatched on `config.QUEUE_TYPE`. It re-exports the RQ
  symbols the codebase uses and exposes `build_embedded_redis_argv(...)`.
- **`app_helper.py`** no longer constructs anything; it does
  `from database import get_db, close_db` and
  `from taskqueue import redis_conn, rq_queue_high, rq_queue_default, Job, NoSuchJobError, send_stop_job_command`.
  Every existing `from app_helper import …` keeps working unchanged.

Both `postgres`/`embedded` (DB) and `redis`/`embedded` (queue) connect identically —
they read `config.DATABASE_URL` / `config.REDIS_URL`. The **only** difference in
embedded mode is *who starts the server* (the supervisor) and *what URL the env
points at* (the embedded socket). Connections are still plain `psycopg2.connect` /
`Redis.from_url`.

### Config switches (defined once in `config.py`, defaults preserve cloud behavior)

`DATABASE_TYPE` (`postgres`), `QUEUE_TYPE` (`redis`), `APP_DATA_DIR` (`""`),
`AUDIOMUSE_PLATFORM` (`""`), `AUDIOMUSE_CONTROL_SOCKET` (`""`). The macOS supervisor
sets `DATABASE_TYPE=embedded`, `QUEUE_TYPE=embedded`, `AUDIOMUSE_PLATFORM=macos` and
the embedded `DATABASE_URL`/`REDIS_URL` **into each child's environment**. Because
`config.py` reads env at import, each child imports `config` already pointed at the
embedded services. With defaults unset, Docker/K8s behavior is unchanged.

### macOS process model

The frozen binary is **one** executable that behaves two ways:
- No args → runs the **rumps menu-bar agent** (`macos/launcher.py::_run_menubar`),
  which constructs a `ProcessSupervisor` and auto-starts it.
- `--role=<x>` → runs **one child service** (`_run_role`). Workers/janitor/listener
  are re-run via `runpy.run_module(<name>, run_name="__main__")` (they have no
  reusable `main()` except `restart_listener`); the web server is `waitress.serve`
  on `app.app`. Children are spawned by the supervisor as
  `[sys.executable, "--role=…"]`.

**`ProcessSupervisor` (`macos/supervisor.py`)** boots in order: embedded Postgres
(`database.start_embedded`) → embedded Redis (bundled binary, args from
`taskqueue.build_embedded_redis_argv`) → `flask` (gated on an HTTP readiness probe;
this is what runs `init_db()` and creates the schema) → `rq-worker-high` →
`rq-worker-default` → `rq-janitor` → `restart-listener`. Children spawn with
`start_new_session=True`; shutdown is reverse-order `killpg(SIGTERM)` → `SIGKILL`
backstop, then `database.stop_embedded()`. A health thread restarts children that
exit while `RUNNING` (mirrors supervisord `autorestart`; note RQ workers
intentionally exit after `max_jobs` and are meant to be respawned). Orphans from a
crashed previous run are reaped on startup via a PID file + `psutil`.

**Control plane.** The container uses supervisord; macOS has none. The web UI's
"save config → restart workers" flow publishes to Redis → `restart_listener`
(a supervised child) → `restart_manager`. On macOS (`config.AUDIOMUSE_PLATFORM ==
"macos"`) `restart_manager._run_supervisorctl` / `_spawn_supervisorctl` instead send
one JSON line `{"action", "services"}` to `config.AUDIOMUSE_CONTROL_SOCKET`, served
by `macos/control_ipc.ControlServer`, which calls
`ProcessSupervisor.dispatch_control`. The service names (`flask`,
`rq-worker-default`, `rq-worker-high`, `rq-janitor`) map 1:1 to supervisor children.

### Surgical edits outside `/macos` (and why)

- `config.py:40` — `TEMP_DIR` was a hardcoded `/app/temp_audio` constant (unwritable
  inside a read-only bundle); now `os.environ.get(...)`.
- `flask_app.py` — `template_folder`/`static_folder` resolve via `sys._MEIPASS` when
  `getattr(sys, "frozen", False)`; identical to before in dev.
- `restart_manager.py` — the macOS control-socket branch described above.
- Model paths were **already** env-driven (`EMBEDDING_MODEL_PATH`,
  `PREDICTION_MODEL_PATH`, `CLAP_AUDIO_MODEL_PATH`, `CLAP_TEXT_MODEL_PATH`,
  `LYRICS_MODEL_DIR`); `macos/env.py` just repoints them into the bundled `model/`
  dir. No code change there.

### Gotchas / where bugs will hide (ranked)

1. **Unsigned nested binaries** (Postgres, Redis, dylibs) get killed by
   Gatekeeper/quarantine. Mitigated by ad-hoc signing everything in `build.sh` +
   the `xattr -dr com.apple.quarantine` user step. No hardened runtime, no
   notarization (we have no Apple Developer account).
2. **RQ job funcs not importable when frozen** — RQ imports `tasks.foo.bar` by
   string at run time; static analysis misses them. Fixed by
   `macos/hooks/hook-tasks.py` (`collect_submodules("tasks")`). If jobs fail with
   ModuleNotFoundError only at run time, this hook (or the spec hiddenimports) is
   the place to look.
3. **pgserver binaries under PyInstaller** — collected via
   `collect_data_files("pgserver")`. If `pgserver.get_server` can't find
   initdb/postgres, set `pgserver.POSTGRES_BIN_PATH` to the bundled path before
   calling it. (pgserver API used here: `get_server(data_dir)` → `.get_uri()` →
   `.cleanup()`, mirroring `test/test_provider_migration_integration.py`.)
4. **numba "cannot cache function"** in frozen apps via librosa — `macos/env.py`
   sets `NUMBA_CACHE_DIR` to a writable dir before any import.
5. **LSUIElement may be ignored** by the PyInstaller bootloader, showing a Dock
   icon. Belt-and-braces: Info.plist key in the spec **and** a runtime
   `NSApp().setActivationPolicy_(Accessory)` in `_run_menubar`.
6. **Per-arch, not universal2** — onnxruntime/PyAV/voyager wheels aren't reliably
   universal2. Build once on arm64, once on x86_64.
7. **waitress, not gunicorn**, for the macOS web server — avoids fork-in-frozen-app
   fragility. The RQ workers still fork per job (that's separate and unchanged).

### Debugging entry points

- **Logs:** `~/Library/Logs/AudioMuse-AI/audiomuse.log` (rotating). Every child's
  stdout/stderr is pumped here, tagged `[flask]`, `[rq-worker-default]`, etc., by
  `ProcessSupervisor._pump`.
- **State dir:** `~/Library/Application Support/AudioMuse-AI/` — `pgdata/`,
  `redis/redis.sock`, `temp_audio/`, `numba_cache/`, `control.sock`,
  `supervisor_pids.json`.
- **Run a single role by hand** (after building): `dist/AudioMuse-AI.app/Contents/MacOS/AudioMuse-AI --role=flask`
  (or `--role=worker-default`, etc.) — but it expects the supervisor's env vars, so
  reproduce them (or export `DATABASE_URL`/`REDIS_URL`/model paths) first.
- **Verified on Linux/WSL (no Mac needed):** the whole non-GUI chain imports; the
  full unit suite (1069 passed, 1 skipped) and the real-Postgres integration suite
  (18 passed, via `pgserver`) both pass with defaults — i.e. the seam refactor is
  regression-clean. Only the bundle build, the menu bar (rumps/AppKit), waitress,
  the bundled `redis-server`, icon generation and signing are Mac-only.
- **If cloud/container behavior regressed:** check that `DATABASE_TYPE`/`QUEUE_TYPE`
  default to `postgres`/`redis` and that `app_helper` still re-exports the handles —
  those two facts are what keep the non-macOS path identical.
