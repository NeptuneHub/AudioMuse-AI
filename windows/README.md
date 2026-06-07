# AudioMuse-AI — Standalone Windows App

Windows counterpart of the [`macos/`](../macos) and [`linux/`](../linux) builds.
Packages the entire AudioMuse-AI stack (Python, ONNX models, embedded PostgreSQL
via pgserver, embedded Redis, the Flask web UI, the RQ workers) into a single
self-contained folder, distributed as a zip — no Docker, no external database,
no manual setup.

> **x86_64 only** initially. ARM64 Windows support may be added later once
> onnxruntime/pgserver/redis-server Windows arm64 builds are validated.

## Quick start (developer build)

**Prerequisites:**
* Windows 10/11 (x86_64)
* Python 3.12
* Visual Studio Build Tools (for compiling pg-contrib; or use pre-built vendor binaries)

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
#    dist\AudioMuse-AI-amd64-windows.zip — ZIP archive (the shipped artifact)
```

## What the build produces

* `dist/AudioMuse-AI/` — runnable folder. Double-click `AudioMuse-AI.exe` to launch the tray app, or run from a terminal.
* `dist/AudioMuse-AI-amd64-windows.zip` — the shipped artifact: a portable zip of the one-dir bundle. Unzip anywhere and run `AudioMuse-AI.exe`.

## Runtime layout

The unzipped bundle (run it from wherever you extract it):
```
AudioMuse-AI\
├── AudioMuse-AI.exe       # Launcher (tray app / start/stop/status/open)
└── _internal\             # PyInstaller bundle
    ├── python3.dll
    ├── model\             # ONNX models + HuggingFace cache
    ├── templates\
    ├── static\
    ├── redis-server.exe   # Embedded Redis
    └── pgserver\          # Embedded PostgreSQL (pgserver wheel)
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

## Tray app

Launching with no arguments (double-click, or the Start Menu / desktop shortcut)
opens a notification-area (system tray) icon — the Windows counterpart of the
macOS menu-bar agent. Right-click the icon for the menu:

* **Status** — Running / Starting… / Stopped
* **Open in Browser** — open the web UI (also the left-click action)
* **Start** / **Stop** — boot or shut down the embedded stack
* **Open Log** — open `audiomuse.log` in the default editor
* **Quit** — stop everything and exit

The console window is hidden automatically when launched by double-click (it stays
visible when you run a command from an existing terminal).

## CLI commands

```
AudioMuse-AI.exe              # Open the tray app (default)
AudioMuse-AI.exe tray         # Same as above
AudioMuse-AI.exe start        # Run the supervisor in the foreground console (logs to the terminal)
AudioMuse-AI.exe stop         # Gracefully shut down a running instance
AudioMuse-AI.exe status       # Print running/stopped
AudioMuse-AI.exe open         # Open web UI (auto-starts if stopped)
```

## Platform notes

* **No Unix sockets** — Windows uses TCP on 127.0.0.1 for Redis (6379), PostgreSQL (5432), and the control server (8001). The `restart_manager.py` shared code uses `AUDIOMUSE_PLATFORM=macos` (same as Linux) to select the socket-based restart path; `windows/control_server.py` provides the same JSON-line protocol over TCP.
* **No `flock`** — single-instance enforcement uses a Windows named mutex (`CreateMutexW`).
* **Tray app via `pystray`** (not `rumps`/`AppKit`) — the Windows counterpart of the macOS menu-bar agent: a notification-area icon with Start / Stop / Open Log / Open in Browser / Quit. The exe stays a console app (so the CLI subcommands and `CTRL_BREAK_EVENT` shutdown keep working); the console window is just hidden when launched by double-click.
* **`CTRL_BREAK_EVENT`** instead of `SIGTERM` for child process termination.
* **`waitress`** (not `gunicorn`) serves the Flask app — same as macOS and Linux builds.

## Shared-code impact

**Zero.** The Windows build follows the same pattern as Linux:

* Reports `AUDIOMUSE_PLATFORM=macos` so `restart_manager.py` uses the control-server path (no change).
* Uses the macOS hooks (`macos/hooks/hook-tasks.py`), `macos.reverse_log`, which are platform-agnostic.
* `database.py` is called as-is via `windows/db_backend.py`; the fallback `windows/embedded_pg.py` only activates when pgserver isn't available.
* All path overrides go through `windows/env.py` → `config` environment variables.
