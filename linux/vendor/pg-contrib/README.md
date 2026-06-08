# Vendored PostgreSQL contrib extensions (`unaccent`, `pg_trgm`) — Linux

`pgserver` ships a **minimal** PostgreSQL (only `plpgsql` and `pgvector`).
The AudioMuse-AI schema (`app_helper.py::init_db`) requires two contrib
extensions that pgserver does **not** include:

```sql
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

So we build those two modules from PostgreSQL source and the shared PyInstaller spec
(`AudioMuse-AI.spec`) grafts them into the bundled `pgserver/pginstall`
tree (`.so` → `lib/postgresql/`, `*.control`/`*--*.sql` →
`share/postgresql/extension/`, `unaccent.rules` →
`share/postgresql/tsearch_data/`).

## Why these must be built from the exact pgserver PG version (not the distro's)

PostgreSQL loadable modules are **not** ABI-stable across minor releases. A
module copied from the distro's `postgresql-16` package fails to load in
pgserver's bundled server if the minor versions differ. The modules must be
compiled against the **exact** server version pgserver ships
(`pgserver==0.1.4` → PostgreSQL 16.2). pgserver conveniently bundles
`pg_config`, the server headers and `pgxs`, so the modules compile directly
against its own install.

## How to (re)generate

Run on the target arch (the CI workflow runs it on x86_64 and aarch64 runners),
from the repo root inside the build venv:

```bash
source .venv-linux/bin/activate     # so pgserver is importable
bash linux/vendor/pg-contrib/build-pg-contrib.sh
```

Output lands in `linux/vendor/pg-contrib/<arch>/` (`x86_64` or `aarch64`):

```
<arch>/lib/unaccent.so
<arch>/lib/pg_trgm.so
<arch>/extension/{unaccent,pg_trgm}.control
<arch>/extension/{unaccent,pg_trgm}--*.sql
<arch>/tsearch_data/unaccent.rules
```

These are built fresh in CI and are **not** committed to git (see
`../README.md`).
