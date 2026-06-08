# Vendored PostgreSQL contrib extensions (`unaccent`, `pg_trgm`)

`pgserver` ships a **minimal** PostgreSQL 16.2 — only `plpgsql` and `pgvector`.
The AudioMuse-AI schema (`app_helper.py::init_db`) requires two contrib
extensions that pgserver does **not** include:

```sql
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

So we vendor those two modules here and the shared PyInstaller spec
(`AudioMuse-AI.spec`) grafts them into the bundled `pgserver/pginstall`
tree (`.dylib` → `lib/postgresql/`, `*.control`/`*--*.sql` →
`share/postgresql/extension/`, `unaccent.rules` →
`share/postgresql/tsearch_data/`).

## Why these must be built from 16.2 source (not copied from Homebrew)

PostgreSQL loadable modules are **not** ABI-stable across minor releases. A
module copied from Homebrew's `postgresql@16` (currently 16.14) fails to load in
pgserver's 16.2 server with e.g. `Symbol not found: _pg_mblen_cstr` — a symbol
added after 16.2. The modules must be compiled against the **exact** server
version pgserver ships.

pgserver conveniently bundles `pg_config`, the server headers and `pgxs`, so the
modules can be compiled directly against its own install.

## How to regenerate (per architecture)

Run on the target arch (arm64 build host for `arm64/`, Intel for `x86_64/`):

```bash
# 1. pgserver's pg_config (in the build venv)
PGC="$(python -c 'import pgserver,os;print(os.path.join(os.path.dirname(pgserver.__file__),"pginstall","bin","pg_config"))')"
PGVER="$("$PGC" --version | awk '{print $2}')"   # e.g. 16.2

# 2. matching PostgreSQL source
curl -fsSL "https://ftp.postgresql.org/pub/source/v${PGVER}/postgresql-${PGVER}.tar.bz2" | tar xj
cd "postgresql-${PGVER}"

# 3. build against pgserver's ABI. PG_SYSROOT overrides the (often stale) SDK
#    path baked into pgserver's pg_config on its CI builder.
SDK="$(xcrun --show-sdk-path)"
for m in unaccent pg_trgm; do
  make -C "contrib/$m" USE_PGXS=1 PG_CONFIG="$PGC" PG_SYSROOT="$SDK"
done

# 4. drop the artifacts here (ARCH = arm64 | x86_64)
ARCH="$(uname -m)"
DEST="macos/vendor/pg-contrib/$ARCH"
mkdir -p "$DEST/lib" "$DEST/extension" "$DEST/tsearch_data"
cp contrib/unaccent/unaccent.dylib contrib/pg_trgm/pg_trgm.dylib "$DEST/lib/"
cp contrib/unaccent/unaccent.control contrib/unaccent/unaccent--*.sql "$DEST/extension/"
cp contrib/pg_trgm/pg_trgm.control  contrib/pg_trgm/pg_trgm--*.sql   "$DEST/extension/"
cp contrib/unaccent/unaccent.rules "$DEST/tsearch_data/"
```

Each `.dylib` should depend only on `/usr/lib/libSystem.B.dylib` (`otool -L`);
server symbols resolve at load time via `-bundle_loader postgres`.
`scripts/standalone/platforms/macos.py` ad-hoc signs them along with every other nested binary.
