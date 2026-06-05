# Vendored PostgreSQL (Linux aarch64)

`pgserver` (used on x86_64) ships **no Linux/aarch64 wheel**, so the arm64 build
bundles its own PostgreSQL. `build-postgres.sh` compiles a relocatable
PostgreSQL — server, client tools, and the `unaccent` / `pg_trgm` contrib
extensions the schema needs — from source and installs it into
`linux/vendor/postgres/<arch>/`. The PyInstaller spec then bundles that tree as
`pgsql/`, and `linux/embedded_pg.py` drives it at runtime with
`initdb` + `pg_ctl`.

> **pgvector is intentionally not built.** The app does vector similarity
> in-process via the `voyager` library and never runs `CREATE EXTENSION vector`
> (it only creates `unaccent` and `pg_trgm`), so the arm64 server doesn't need
> pgvector. (The x86_64 `pgserver` wheel happens to include it; it's unused.)

## How to (re)generate

Run on the target arch (the CI workflow runs it on the aarch64 runner), from the
repo root. Needs `build-essential`, `bison`, `flex`, `zlib1g-dev`:

```bash
bash linux/vendor/postgres/build-postgres.sh        # PG_VERSION=16.9 by default
```

Output (built fresh in CI, **not** committed — see `../README.md`):

```
<arch>/bin/{postgres,initdb,pg_ctl,psql,pg_dump,pg_restore,...}
<arch>/lib/...                          # server libs
<arch>/lib/postgresql/{unaccent,pg_trgm}.so
<arch>/share/postgresql/...             # incl. extension/*.{control,sql}, tsearch_data/unaccent.rules
```

## Why from source (not a prebuilt binary)

Building the `unaccent` / `pg_trgm` contrib modules requires the server headers
and `pgxs`, which runtime-only binary distributions (e.g. zonky's
embedded-postgres-binaries) do not ship. Building from source also lets us pass
`--without-icu` (no libicu runtime dependency; initdb uses the libc locale
provider) and `--without-readline`, keeping the bundle self-contained.
PostgreSQL is relocatable — it derives its support-file paths from the running
executable — so the installed tree works from wherever the package is installed.
