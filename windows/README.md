# AudioMuse-AI — Standalone Windows App

Windows counterpart of the [`macos/`](../macos) and [`linux/`](../linux) builds.
Packages the entire AudioMuse-AI stack (Python, ONNX models, embedded PostgreSQL
via pgserver, embedded Redis, the Flask web UI, the RQ workers) into a single
MSI installer — no Docker, no external database, no manual setup.

> **x86_64 only** initially. ARM64 Windows support may be added later once
> onnxruntime/pgserver/redis-server Windows arm64 builds are validated.

## Quick start (developer build)

**Prerequisites:**
* Windows 10/11 (x86_64)
* Python 3.12
* Visual Studio Build Tools (for compiling pg-contrib; or use pre-built vendor binaries)
* [WiX Toolset v5](https://wixtoolset.org/) (optional; for `.msi` packaging)

**Steps:**
```powershell
# 1. Create and activate a virtual environment
python -m venv .venv-windows
.venv-windows\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements\windows.txt

# 3. Build or download vendor binaries (Redis + PostgreSQL contrib)
windows\vendor\build-redis.bat
# For pg-contrib, either cross-compile on Linux (see vendor README)
# or use pre-built binaries committed to the repo.

# 4. Assemble ./model (same models as Docker/macOS/Linux)
#    In CI this is done by the workflow. For local dev:
#    - Download models from GitHub releases (see build-windows.yml for URLs)

# 5. Build the app bundle
set PKG_VERSION=0.0.0
windows\build.bat

# 6. Output
#    dist\AudioMuse-AI\                — one-dir PyInstaller bundle
#    dist\AudioMuse-AI-amd64-windows.msi — MSI installer (if WiX is installed)
#    dist\AudioMuse-AI-amd64-windows.zip — ZIP archive (always produced)
```

## What the build produces

* `dist/AudioMuse-AI/` — unzipped, runnable folder. Double-click `AudioMuse-AI.exe` or run from a terminal.
* `dist/AudioMuse-AI-amd64-windows.msi` — MSI installer. Installs to `C:\Program Files\AudioMuse-AI\`, creates Start Menu shortcuts.
* `dist/AudioMuse-AI-amd64-windows.zip` — portable zip archive (same content as the one-dir bundle).

## Runtime layout

When installed via MSI:
```
C:\Program Files\AudioMuse-AI\
├── AudioMuse-AI.exe       # Launcher (start/stop/status/open)
├── _internal\             # PyInstaller bundle
│   ├── python3.dll
│   ├── model\             # ONNX models + HuggingFace cache
│   ├── templates\
│   ├── static\
│   ├── redis-server.exe   # Embedded Redis
│   └── pgsql\             # Embedded PostgreSQL (if pgserver not used)
└── uninstall.exe
```

Writable data lives in `%LOCALAPPDATA%\AudioMuse-AI\`:
```
C:\Users\<user>\AppData\Local\AudioMuse-AI\
├── pgdata\                # PostgreSQL cluster
├── redis\                 # Redis working directory
├── temp_audio\            # Transcoding scratch
├── numba_cache\
├── backup\
├── logs\
│   └── audiomuse.log      # Newest lines first (same as macOS)
├── supervisor.lock        # Single-instance mutex
└── supervisor_pids.json
```

## CLI commands

```
AudioMuse-AI.exe              # Start the full stack + open browser
AudioMuse-AI.exe start        # Same as above
AudioMuse-AI.exe stop         # Gracefully shut down
AudioMuse-AI.exe status       # Print running/stopped
AudioMuse-AI.exe open         # Open web UI (auto-starts if stopped)
```

## Platform notes

* **No Unix sockets** — Windows uses TCP on 127.0.0.1 for Redis (6379), PostgreSQL (5432), and the control server (8001). The `restart_manager.py` shared code uses `AUDIOMUSE_PLATFORM=macos` (same as Linux) to select the socket-based restart path; `windows/control_server.py` provides the same JSON-line protocol over TCP.
* **No `flock`** — single-instance enforcement uses a Windows named mutex (`CreateMutexW`).
* **No `rumps`/`AppKit`** — the launcher is a console application, not a menu-bar agent.
* **`CTRL_BREAK_EVENT`** instead of `SIGTERM` for child process termination.
* **`waitress`** (not `gunicorn`) serves the Flask app — same as macOS and Linux builds.

## Shared-code impact

**Zero.** The Windows build follows the same pattern as Linux:

* Reports `AUDIOMUSE_PLATFORM=macos` so `restart_manager.py` uses the control-server path (no change).
* Uses the macOS hooks (`macos/hooks/hook-tasks.py`), `macos.reverse_log`, which are platform-agnostic.
* `database.py` is called as-is via `windows/db_backend.py`; the fallback `windows/embedded_pg.py` only activates when pgserver isn't available.
* All path overrides go through `windows/env.py` → `config` environment variables.
