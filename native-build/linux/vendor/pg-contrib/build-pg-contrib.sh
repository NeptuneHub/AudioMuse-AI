#!/usr/bin/env bash
#
# Compile the PostgreSQL ``unaccent`` and ``pg_trgm`` contrib extensions against
# the EXACT PostgreSQL version that ``pgserver`` bundles, and drop the artifacts
# under ``native-build/linux/vendor/pg-contrib/<arch>/`` where the PyInstaller spec grafts
# them into the bundled ``pgserver/pginstall`` tree.
#
# Run from the repo root, inside the build venv (so ``pgserver`` is importable):
#     source .venv-linux/bin/activate
#     bash native-build/linux/vendor/pg-contrib/build-pg-contrib.sh
#
# Why this is necessary: ``pgserver`` ships a MINIMAL PostgreSQL (only plpgsql +
# pgvector), but the AudioMuse-AI schema (app_helper.py::init_db) runs
#     CREATE EXTENSION IF NOT EXISTS unaccent;
#     CREATE EXTENSION IF NOT EXISTS pg_trgm;
# PostgreSQL loadable modules are NOT ABI-stable across minor releases, so the
# modules must be compiled against pgserver's own server version/headers (it
# conveniently bundles pg_config + the server headers + pgxs).
set -euo pipefail

ARCH="$(uname -m)"   # x86_64 | aarch64
DEST="native-build/linux/vendor/pg-contrib/${ARCH}"

# pgserver's own pg_config (in the active venv).
PGC="$(python -c 'import pgserver,os;print(os.path.join(os.path.dirname(pgserver.__file__),"pginstall","bin","pg_config"))')"
[ -x "$PGC" ] || { echo "pg_config not found at $PGC (is pgserver installed?)" >&2; exit 1; }
PGVER="$("$PGC" --version | awk '{print $2}')"   # e.g. 16.2
echo "==> pgserver bundles PostgreSQL ${PGVER}; building unaccent + pg_trgm against it"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "==> Fetching PostgreSQL ${PGVER} source"
curl -fsSL "https://ftp.postgresql.org/pub/source/v${PGVER}/postgresql-${PGVER}.tar.bz2" \
  | tar xj -C "$work"
srcdir="$work/postgresql-${PGVER}"

for m in unaccent pg_trgm; do
  echo "==> Building contrib/$m"
  make -C "$srcdir/contrib/$m" USE_PGXS=1 PG_CONFIG="$PGC" -j"$(nproc)"
done

echo "==> Collecting artifacts into $DEST"
mkdir -p "$DEST/lib" "$DEST/extension" "$DEST/tsearch_data"
cp "$srcdir/contrib/unaccent/unaccent.so" "$DEST/lib/"
cp "$srcdir/contrib/pg_trgm/pg_trgm.so"   "$DEST/lib/"
cp "$srcdir/contrib/unaccent/unaccent.control" "$srcdir"/contrib/unaccent/unaccent--*.sql "$DEST/extension/"
cp "$srcdir/contrib/pg_trgm/pg_trgm.control"   "$srcdir"/contrib/pg_trgm/pg_trgm--*.sql   "$DEST/extension/"
cp "$srcdir/contrib/unaccent/unaccent.rules"   "$DEST/tsearch_data/"

echo "==> Done. Artifacts:"
find "$DEST" -type f
echo "==> .so dependencies (server symbols resolve at load time inside postgres):"
ldd "$DEST/lib/unaccent.so" || true
ldd "$DEST/lib/pg_trgm.so" || true
