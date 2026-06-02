# AudioMuse-AI — Standalone Linux build (`.deb` / `.rpm`)

This folder builds AudioMuse-AI into native Linux packages with **no Docker and
no separately-installed PostgreSQL or Redis**. Installing the `.deb` or `.rpm`
drops a fully self-contained app under `/opt/AudioMuse-AI` and a launcher in your
application menu. It is the Linux counterpart of the [`macos/`](../macos) build.

The package bundles:

- **Embedded PostgreSQL** via [`pgserver`](https://github.com/orm011/pgserver)
  (with `pgvector`, plus the `unaccent` / `pg_trgm` contrib extensions we vendor).
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

Then launch **AudioMuse-AI** from your application menu, or from a terminal:

```bash
audiomuse-ai start     # start everything + open the browser (foreground)
audiomuse-ai stop      # cleanly stop PostgreSQL, Redis and all workers
audiomuse-ai status    # is it running?
audiomuse-ai open      # open the web UI (starting the stack first if needed)
```

The web UI is at `http://127.0.0.1:8000`. After first launch, configure your
media server (Jellyfin, Navidrome, Lyrion, Emby or MPD) exactly as for the
container version.

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

`.github/workflows/build-linux.yml` builds both architectures
(`x86_64` on `ubuntu-22.04`, `aarch64` on `ubuntu-22.04-arm`) on every `v*.*.*`
tag and on PRs, and attaches the `.deb`/`.rpm` to the release. For each arch it:

1. installs the build toolchain + Python deps (`requirements/linux.txt`),
2. builds the vendored `redis-server` (`linux/vendor/build-redis.sh`),
3. builds the vendored `unaccent`/`pg_trgm` against pgserver's PostgreSQL
   (`linux/vendor/pg-contrib/build-pg-contrib.sh`),
4. assembles `./model` from the model releases (trimming the HF cache to the
   roberta tokenizer so the assets stay under GitHub's 2 GB limit),
5. runs `linux/build.sh` → PyInstaller (`linux/AudioMuse-AI.spec`) → `nfpm`.

## Building (developer machine, x86_64 or aarch64 Linux)

```bash
python3.12 -m venv .venv-linux
source .venv-linux/bin/activate
pip install -r requirements/linux.txt

# Native build inputs (need build-essential, libreadline-dev, zlib1g-dev, rpm)
bash linux/vendor/build-redis.sh
bash linux/vendor/pg-contrib/build-pg-contrib.sh

# Models: assemble ./model exactly as the workflow does (see build-linux.yml),
# or copy an existing ./model tree into the repo root.

# Package (needs nfpm on PATH: https://nfpm.goreleaser.com)
PKG_VERSION=1.0.0 bash linux/build.sh
# -> dist/AudioMuse-AI-<arch>.deb  and  dist/AudioMuse-AI-<arch>.rpm
```

## Layout of this folder

| File | Purpose |
| --- | --- |
| `launcher.py` | PyInstaller entry point: `start`/`stop`/`status`/`open` and the `--role=` child entry points. |
| `supervisor.py` | Process supervisor (embedded PG/Redis + Flask + workers); port of `macos/supervisor.py`. |
| `paths.py` | XDG-based writable dirs + bundled-resource locations. |
| `env.py` | The environment handed to each child (embedded DB/queue, model paths). |
| `AudioMuse-AI.spec` | PyInstaller one-dir spec. |
| `build.sh` | PyInstaller build + `nfpm` packaging into `.deb`/`.rpm`. |
| `packaging/` | `nfpm` config template, `.desktop` entries, post-install/-remove scripts. |
| `vendor/` | Helper scripts that build `redis-server` and the PG contrib modules in CI. |

> **No shared code is modified by this build.** The `linux/` package only *adds*
> helpers. It reuses the platform-agnostic `macos.control_ipc` /
> `macos.reverse_log` helpers, and it reports `AUDIOMUSE_PLATFORM=macos` to the
> shared `restart_manager.py` so the UI's "restart workers" flow uses the
> control-socket path (the only platform-keyed branch there) — see
> `linux/env.py` for the full rationale.
