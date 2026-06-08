# AudioMuse-AI — Standalone Linux build (`.deb` / `.rpm`)

This folder builds AudioMuse-AI into native Linux packages with **no Docker and
no separately-installed PostgreSQL or Redis**. Installing the `.deb` or `.rpm`
drops a fully self-contained app under `/opt/AudioMuse-AI` and a launcher in your
application menu. It is the Linux counterpart of the [`macos/`](../macos) build.

The package bundles:

- **Embedded PostgreSQL** with the `unaccent` / `pg_trgm` extensions the schema
  needs. On **x86_64** this is [`pgserver`](https://github.com/orm011/pgserver)
  (+ the two contrib modules we vendor); on **aarch64** (no pgserver wheel) it's
  a relocatable PostgreSQL built from source in CI and driven by `initdb` /
  `pg_ctl` (`linux/embedded_pg.py`). Both look identical to the app.
- **Embedded Redis** via a bundled `redis-server` binary.
- The Flask web UI (served by `waitress`) on `http://127.0.0.1:8000`.
- The two RQ workers, the janitor, and the config-restart listener.
- The ~5 GB of ONNX models (MusiCNN, CLAP audio+text, the RoBERTa tokenizer
  cache, Whisper-small, gte-multilingual, Silero VAD) — so analysis **and**
  lyrics transcription work fully offline.

A single `audiomuse-ai` process supervises all of the above (start, health-check,
auto-restart, clean shutdown), exactly like the macOS menu-bar agent and the
container's supervisord.

## Using the package

```bash
# Debian / Ubuntu
sudo apt install ./AudioMuse-AI-x86_64.deb

# Fedora / RHEL / openSUSE
sudo dnf install ./AudioMuse-AI-x86_64.rpm
```

> Use `apt install ./file.deb` (not `sudo dpkg -i file.deb`). `dpkg -i` does
> **not** pull the two declared dependencies (`libgomp1`, `xdg-utils`) and leaves
> the package half-configured if they are absent — its post-install step (which
> refreshes the menu/icon caches) then never runs, so the launcher may not show
> up. If you already used `dpkg -i`, finish it with `sudo apt-get -f install`.

### Starting it

**The app is started on demand — it is not a background daemon, so nothing
listens on `http://127.0.0.1:8000` until you start it** (just like the macOS
menu-bar app). Launch **AudioMuse-AI** from your application menu, or from a
terminal:

```bash
audiomuse-ai start     # start everything + open the browser (foreground)
audiomuse-ai stop      # cleanly stop PostgreSQL, Redis and all workers
audiomuse-ai status    # is it running?
audiomuse-ai open      # open the web UI (starting the stack first if needed)
```

The web UI is at `http://127.0.0.1:8000`. After first launch, configure your
media server (Jellyfin, Navidrome, Lyrion, Emby or MPD) exactly as for the
container version.

If the **AudioMuse-AI** launcher does not appear in your menu right after
install, log out and back in (some desktops only rescan `/usr/share/applications`
on session start).

### Optional: start automatically at login

The package ships a **systemd user service** (disabled by default). Enable it to
have AudioMuse-AI start for your user on login and stay supervised in the
background:

```bash
systemctl --user enable --now audiomuse-ai      # start now + on every login
loginctl enable-linger "$USER"                  # also keep it running after logout
systemctl --user status audiomuse-ai            # check it
systemctl --user disable --now audiomuse-ai     # turn it back off
```

It runs as your user over the same per-user data dir (no root, no system
service), and starts without opening a browser (`AUDIOMUSE_OPEN_BROWSER=0`).

### Where state lives

All **writable** state lives under your home (never inside the read-only
`/opt/AudioMuse-AI`), following the XDG spec:

- Database / Redis / scratch / backups: `~/.local/share/AudioMuse-AI/`
- Logs: `~/.local/state/AudioMuse-AI/logs/audiomuse.log` — written **newest line
  first**, bounded to the most recent ~40k lines (see `macos/reverse_log.py`).

Uninstalling the package leaves this directory in place (so your analysis
database survives a reinstall); delete it by hand if you want a clean slate.

## Why PostgreSQL and Redis are **bundled**, not system dependencies

The goal was to prefer hard package dependencies on system PostgreSQL/Redis (so
`apt`/`dnf` pull and auto-start them). We bundle them instead, because a single
portable `.deb` + `.rpm` cannot reliably depend on the database the app needs:

- **pgvector + exact PostgreSQL minor.** The schema needs `pgvector` *and* the
  `unaccent` / `pg_trgm` contrib extensions. PostgreSQL loadable modules are not
  ABI-stable across minor releases, and `pgvector` is packaged inconsistently
  across distros (a separate `postgresql-NN-pgvector`, or only via the PGDG
  repo, or absent). Pinning the exact server version the app was tested against
  is exactly what `pgserver` already gives us — and it is the same mechanism the
  repo's own integration tests use.
- **Per-distro version skew.** A hard `Depends: postgresql-16` would fail to
  install on distros that ship a different major (Ubuntu 22.04 → 14, Debian 12 →
  15, etc.). One package cannot name one "precise version" that exists
  everywhere.
- **No root, no auth dance.** The bundled servers run per-user over unix sockets
  under `~/.local/share`, with the supervisor owning their lifecycle
  (start / health-check / auto-restart / shutdown). There is no system cluster
  to provision, no role/password to create, and nothing runs as root.

This satisfies the goal's fallback ("if NOT possible, bundle everything … in a
similar approach to the macOS one"): the macOS build bundles them the same way,
and the supervisor here gives the same autostart/restart guarantees the goal
asked for — just without depending on the host's database.

The only genuine system dependencies declared by the package are `libgomp`
(OpenMP runtime onnxruntime links against) and `xdg-utils` (for `xdg-open`).

## Building (CI)

`.github/workflows/build-linux.yml` builds **x86_64** (on `ubuntu-22.04`) and
**aarch64** (on `ubuntu-22.04-arm`) on every `v*.*.*` tag and on PRs, and
attaches the `.deb`/`.rpm` to the release. For each arch it:

1. installs the build toolchain + Python deps (`requirements/linux.txt`;
   `pgserver` is x86_64-only via a platform marker),
2. builds the vendored `redis-server` (`linux/vendor/build-redis.sh`),
3. provides embedded PostgreSQL per arch:
   - **x86_64** — builds `unaccent`/`pg_trgm` against the pgserver wheel's
     PostgreSQL (`linux/vendor/pg-contrib/build-pg-contrib.sh`),
   - **aarch64** — builds a relocatable PostgreSQL + those contrib modules from
     source (`linux/vendor/postgres/build-postgres.sh`),
4. assembles `./model` from the model releases (trimming the HF cache to the
   roberta tokenizer so the assets stay under GitHub's 2 GB limit),
5. runs `scripts/standalone/build.py --platform linux` → PyInstaller (the shared
   `AudioMuse-AI.spec`) → `nfpm`.

## Building (developer machine)

```bash
python3.12 -m venv .venv-linux
source .venv-linux/bin/activate
pip install -r requirements/linux.txt

# Native build inputs (need build-essential, bison, flex, zlib1g-dev, rpm)
bash linux/vendor/build-redis.sh
# Embedded PostgreSQL — pick the one for your arch:
bash linux/vendor/pg-contrib/build-pg-contrib.sh   # x86_64 (against pgserver)
bash linux/vendor/postgres/build-postgres.sh       # aarch64 (from source)

# Models: assemble ./model exactly as the workflow does (see build-linux.yml),
# or copy an existing ./model tree into the repo root.

# Package (needs nfpm on PATH: https://nfpm.goreleaser.com)
PKG_VERSION=1.0.0 python scripts/standalone/build.py --platform linux
# -> dist/AudioMuse-AI-<arch>-linux.deb  and  dist/AudioMuse-AI-<arch>-linux.rpm
```

## Layout of this folder

| File | Purpose |
| --- | --- |
| `launcher.py` | PyInstaller entry point: `start`/`stop`/`status`/`open` and the `--role=` child entry points. |
| `supervisor.py` | Process supervisor (embedded PG/Redis + Flask + workers); port of `macos/supervisor.py`. |
| `db_backend.py` | Selects the embedded-Postgres backend by arch (pgserver on x86_64, `embedded_pg` on aarch64). |
| `embedded_pg.py` | `initdb`/`pg_ctl` manager for the from-source PostgreSQL bundled on aarch64. |
| `paths.py` | XDG-based writable dirs + bundled-resource locations (incl. per-arch Postgres paths). |
| `env.py` | The environment handed to each child (embedded DB/queue, model paths). |
| `packaging/` | `nfpm` config template, `.desktop` entries, the systemd **user** service, the square app icons (`icons/`), post-install/-remove scripts. |
| `vendor/` | Helper scripts that build `redis-server`, the x86_64 PG contrib modules, and the aarch64 from-source PostgreSQL in CI. |

> **No shared code is modified by this build.** The `linux/` package only *adds*
> helpers. It reuses the platform-agnostic `macos.control_ipc` /
> `macos.reverse_log` helpers, and it reports `AUDIOMUSE_PLATFORM=macos` to the
> shared `restart_manager.py` so the UI's "restart workers" flow uses the
> control-socket path (the only platform-keyed branch there) — see
> `linux/env.py` for the full rationale.
