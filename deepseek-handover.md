# Windows Standalone Build — DeepSeek Handover

## What this is

A complete Windows native build for **AudioMuse-AI** — a standalone `.exe` (PyInstaller) that bundles:
- Embedded **PostgreSQL** (pgserver 0.1.4, PostgreSQL 16.2 + pgvector)
- Embedded **Redis** (tporadowski/redis 5.0.14.1 — variadic HSET required by Python redis 7.x)
- All **ML models** (ONNX: MusiCNN, CLAP audio/text, Whisper-small, Silero VAD, GTE)
- **Flask** web UI served via `waitress`
- **RQ workers** (SimpleWorker on Windows — see below)
- **MSI installer** scaffolding (WiX Toolset v4)

Built with PyInstaller 6.20.0 on Python 3.11.9, targeting `windows-2022` GitHub Actions runner.

---

## Source files created/modified

### New files (`windows/`)

| File | Purpose |
|------|---------|
| `windows/launcher.py` | Entry point. Multiprocessing `fork`→`spawn` monkey-patch, POSIX `os` function stubs (`wait4`, `WIFEXITED`, `WIFSIGNALED`, `WTERMSIG`, `WEXITSTATUS`), OTEL context fix. CLI: `start`/`stop`/`status`/`open`. Single-instance lock via Windows named mutex. |
| `windows/supervisor.py` | Process supervisor. Boots PostgreSQL → Redis → Flask → RQ workers → janitor → restart-listener. TCP control server (no Unix sockets on Windows). Health loop, auto-restart children. |
| `windows/env.py` | Builds environment for child processes. Sets `AUDIOMUSE_PLATFORM=macos` (tricks `restart_manager.py` into using the control-server path). TCP control host/port, all model paths, `AUDIOMUSE_ROLE=worker` for worker children. |
| `windows/paths.py` | Windows filesystem paths: `%LOCALAPPDATA%\AudioMuse-AI` for data, TCP ports for Redis (6379)/PG (5432)/control (8001). |
| `windows/db_backend.py` | DB backend selector (pgserver preferred, fallback `embedded_pg`). Auto-cleans stale pgdata on crash + retry once. |
| `windows/control_server.py` | TCP control server (replaces Unix socket). JSON-line protocol + HTTP `/status`/`/stop`. |
| `windows/embedded_pg.py` | Fallback PostgreSQL manager using bundled `pg_ctl`/`initdb`/`pg_isready`. Uses `pg_isready` for health checks (not `pg_ctl poll()` which exits immediately). |
| `windows/AudioMuse-AI.spec` | PyInstaller spec. Correct pg-contrib path (`share/postgresql/extension/`), icon, flasgger hidden import. |
| `windows/build.bat` | Build script: PyInstaller → zip (MSI if WiX available). UpgradeCode from `.wxs`. **No `::error::` inside `()` blocks** — batch parser chokes on `::` in parenthesized blocks. |
| `windows/download_models.bat` | Downloads all ONNX models + HuggingFace cache (~2.4 GB compressed). Error checks after every `tar` extraction. |
| `windows/vendor/build-redis.bat` | Downloads Redis 5.0.14.1 from tporadowski/redis, extracts `redis-server.exe`. **No `::error::` inside `()` blocks.** |
| `windows/vendor/pg-contrib/amd64/extension/` | PostgreSQL extension SQL/control files for `unaccent` and `pg_trgm` (text files). DLLs are empty placeholders — need cross-compilation from Linux. |
| `windows/vendor/pg-contrib/amd64/tsearch_data/unaccent.rules` | Accent-removal character mapping for the `unaccent` extension. |
| `windows/packaging/AudioMuse-AI.wxs` | Full WiX v4 MSI authoring with UpgradeCode `A1B2C3D4-E5F6-7890-ABCD-EF1234567890`. |
| `.github/workflows/build-windows.yml` | Full CI: checkout → Python 3.12 → install deps → build redis → assemble models → PyInstaller → upload artifacts (.msi + .zip). Uses `Join-Path $env:RUNNER_TEMP` (NOT `New-TemporaryFile` pipeline which is buggy). |
| `requirements/windows.txt` | Extends common.txt with `onnxruntime`, `pgserver`, `waitress`, `pywin32`, `pyinstaller`, `pyinstaller-hooks-contrib`. |

### Modified shared files (kept diff minimal)

| File | Change |
|------|--------|
| `app_helper.py` | Added `import sys`. Extension creation and `immutable_unaccent` blocks wrapped in `if sys.platform == 'win32': try/except` — non-Windows path unchanged. |
| `restart_manager.py` | `_send_control()` now supports TCP (`AUDIOMUSE_CONTROL_HOST`/`PORT`) alongside Unix socket. Unix path unchanged. |
| `rq_worker.py` | `WorkerClass = SimpleWorker if sys.platform == 'win32' else Worker`. `SimpleWorker` runs jobs in-process — no `os.spawnv()`, no `-c` flag, works in frozen exe. |
| `rq_worker_high_priority.py` | Same `SimpleWorker` switch. Removed 🚀 emoji from `print()` (cp1252 encoding crash). |
| `lyrics/lyrics_transcriber.py` | `SIGALRM`/`signal.alarm()` guarded by `hasattr(signal, 'SIGALRM')` — POSIX-only, skips timeout on Windows. |
| `.gitignore` | Added `/windows/vendor/redis/` (downloaded binary, not committed). |
| `.dockerignore` | Added `windows/vendor/redis/`. |
| `.gitattributes` | Added `*.yml`, `*.yaml`, `*.txt`, `*.spec`, `*.bat`, `*.ps1`, `*.wxs`, `*.md` → `text eol=lf`. |

---

## Key design decisions

### SimpleWorker (not SpawnWorker/Fork)
RQ's `SpawnWorker` uses `os.spawnv()` with `sys.executable -c "..."` — a PyInstaller-frozen `.exe` can't execute `-c` strings. `SimpleWorker` runs jobs directly in the worker process. If a job crashes the worker, the supervisor auto-restarts it. This is the single most important fix — without it, workers enqueue jobs but never execute them.

### POSIX monkey-patches
Windows lacks `os.wait4`, `os.WIFEXITED`, `os.WIFSIGNALED`, `os.WTERMSIG`, `os.WEXITSTATUS`. These are monkey-patched in `launcher.py` before any imports. RQ's `SpawnWorker` references them even though we use `SimpleWorker` — but other RQ internals and the multiprocessing patch need them.

### `AUDIOMUSE_PLATFORM=macos` on Windows
`restart_manager.py` keys on `AUDIOMUSE_PLATFORM == 'macos'` to use the control-server restart path (vs `supervisorctl`). Our Windows supervisor implements the same control protocol over TCP. Setting this reports "macos" to shared code without modifying it.

### TCP control server (no Unix sockets)
Windows has no `AF_UNIX`. The supervisor listens on TCP `127.0.0.1:8001` with a JSON-line protocol (same payload as the Unix socket). `restart_manager.py` was updated to try TCP first (`AUDIOMUSE_CONTROL_HOST`/`PORT`), fall back to Unix socket.

### `::error::` in batch parenthesized blocks
`cmd.exe` treats `::` as a label comment. Inside `( )` blocks, `::error::` breaks the parser with `. was unexpected at this time.` All such occurrences in `.bat` files replaced with `[ERROR]`/`[WARNING]`.

### `New-TemporaryFile` PowerShell pipeline bug
`New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item ... }` can produce a broken path when `$_` references a deleted file. Replaced with `Join-Path $env:RUNNER_TEMP "dirname"` in the CI workflow.

---

## Known limitations

1. **No accent-insensitive search**: `unaccent.dll` and `pg_trgm.dll` are empty placeholders. They need cross-compilation from Linux (`mingw-w64`) against pgserver's PostgreSQL headers. Extension SQL/control files and `unaccent.rules` are in the repo — only the compiled `.dll` files are missing.
2. **No MSI locally**: WiX Toolset v4 must be installed and on PATH. The CI workflow has it via `choco install wixtoolset`.
3. **Redis binary not in repo**: `windows/vendor/redis/` is gitignored. `build-redis.bat` downloads it on demand.
4. **Analyzer thread count**: On Windows, `SimpleWorker` runs in-process and CPU thread caps applied. Large libraries may be slower than Linux/macOS where `fork()` workers each get independent CPU limits.

---

## Local build commands

```powershell
# From repo root, inside the venv:
.venv-windows\Scripts\activate
set PKG_VERSION=1.0.0
windows\build.bat

# Or directly:
.venv-windows\Scripts\python.exe -m PyInstaller windows\AudioMuse-AI.spec --noconfirm

# Run:
.\dist\AudioMuse-AI\AudioMuse-AI.exe

# Stop:
.\dist\AudioMuse-AI\AudioMuse-AI.exe stop
```

## CI

Triggered on PR (opened/reopened/synchronize) and version tags (`v*.*.*`). Builds on `windows-2022`, uploads `.msi` and `.zip` artifacts. The `pr-test-link.yml` workflow posts download links to the PR description.
