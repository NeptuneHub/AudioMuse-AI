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
