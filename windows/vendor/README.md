# Windows vendor binaries

The standalone Windows app embeds two native services:

* **Redis** — the RQ broker and pub/sub bus
* **PostgreSQL contrib modules** — `unaccent` and `pg_trgm` extensions

These are NOT regenerated at build time; they are committed (or built by CI) so
the bundle is reproducible.

## Directory layout

```
windows/vendor/
├── README.md
├── redis/
│   └── amd64/
│       └── redis-server.exe       # Redis 7.x for Windows (Microsoft Archive)
├── pg-contrib/
│   └── amd64/
│       ├── lib/
│       │   ├── unaccent.dll        # compiled against pgserver's PostgreSQL 16.2
│       │   └── pg_trgm.dll
│       ├── extension/
│       │   ├── unaccent.control
│       │   └── pg_trgm.control
│       └── tsearch_data/
│           └── unaccent.rules
└── postgres/                       # fallback: full PostgreSQL tree (if pgserver unavailable)
    └── amd64/
        ├── bin/                    # postgres.exe, initdb.exe, pg_ctl.exe, etc.
        ├── lib/                    # .dll files
        └── share/                  # extension .sql and .control files
```

## Building the vendor binaries

### redis-server.exe

Redis does not officially support Windows. The recommended approach is to use
the **Microsoft Archive Redis for Windows** build:

1. Download the latest release from https://github.com/microsoftarchive/redis/releases
2. Extract `redis-server.exe` to `windows/vendor/redis/amd64/`

Alternatively, build from source with MSYS2/MinGW or use the Windows Subsystem for Linux.

**Version target:** Redis 7.x (matching what the container deployments use).

### PostgreSQL contrib modules (unaccent, pg_trgm)

These must be compiled against the EXACT PostgreSQL version that pgserver bundles
(16.2). The .dll files are ABI-specific to this PG minor.

**Option A — cross-compile on Linux (recommended for CI):**
```bash
# On an x86_64 Linux runner with mingw-w64 installed:
bash windows/vendor/pg-contrib/build-pg-contrib-cross.sh
```

**Option B — build on Windows:**
```powershell
# Requires Visual Studio Build Tools + PostgreSQL 16.2 dev headers
.\windows\vendor\pg-contrib\build-pg-contrib.ps1
```

### Full PostgreSQL tree (fallback, only if pgserver has no Windows wheel)

If pgserver does not provide a Windows wheel, the `windows/embedded_pg.py`
fallback requires a full relocatable PostgreSQL installation.

**Build steps:**
1. Download PostgreSQL 16.2 source from https://www.postgresql.org/ftp/source/
2. Compile with MSVC or MinGW, targeting a relocatable layout
3. Place the resulting `bin/`, `lib/`, `share/` trees under `windows/vendor/postgres/amd64/`

This is the same approach used by the Linux aarch64 build.
