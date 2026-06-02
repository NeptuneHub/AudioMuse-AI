#!/usr/bin/env bash
#
# Build a relocatable PostgreSQL (server + client tools + the ``unaccent`` and
# ``pg_trgm`` contrib extensions) from source and install it under
# ``linux/vendor/postgres/<arch>/`` where the PyInstaller spec bundles it as
# ``pgsql/``. Used for the **aarch64** Linux build, where ``pgserver`` has no
# wheel.
#
# Run from the repo root on the target architecture (the CI workflow runs it on
# the aarch64 runner):
#     bash linux/vendor/postgres/build-postgres.sh
#
# Why from source (not a prebuilt binary like zonky's embedded-postgres):
#   * we must compile the unaccent/pg_trgm contrib modules, which need the
#     server headers + pgxs that runtime-only binary distributions omit;
#   * --without-icu keeps the bundle free of a libicu runtime dependency
#     (initdb defaults to the libc locale provider);
#   * PostgreSQL is relocatable -- it derives its support-file paths from the
#     running executable -- so the installed tree works from wherever the
#     package lands (``/opt/AudioMuse-AI/_internal/pgsql``).
set -euo pipefail

PG_VERSION="${PG_VERSION:-16.9}"
ARCH="$(uname -m)"   # aarch64
PREFIX="$(pwd)/linux/vendor/postgres/${ARCH}"

echo "==> Building PostgreSQL ${PG_VERSION} (${ARCH}) -> ${PREFIX}"
rm -rf "$PREFIX"
mkdir -p "$PREFIX"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "==> Fetching PostgreSQL ${PG_VERSION} source"
curl -fsSL "https://ftp.postgresql.org/pub/source/v${PG_VERSION}/postgresql-${PG_VERSION}.tar.bz2" \
  | tar xj -C "$work"
src="$work/postgresql-${PG_VERSION}"

echo "==> Configuring (relocatable, ICU-free, no readline)"
( cd "$src" && ./configure \
    --prefix="$PREFIX" \
    --without-icu \
    --without-readline \
    --with-zlib \
    >/dev/null )

echo "==> Building + installing server"
make -C "$src" -j"$(nproc)" >/dev/null
make -C "$src" install >/dev/null

echo "==> Building + installing contrib (unaccent, pg_trgm)"
for m in unaccent pg_trgm; do
  make -C "$src/contrib/$m" -j"$(nproc)" >/dev/null
  make -C "$src/contrib/$m" install >/dev/null
done

echo "==> Pruning build-only artifacts (headers, pgxs, static libs, docs)"
rm -rf "$PREFIX/include" "$PREFIX/lib/pgxs" "$PREFIX/share/doc" "$PREFIX/share/man"
find "$PREFIX/lib" -maxdepth 1 -name '*.a' -delete 2>/dev/null || true

echo "==> Installed tree:"
ls -1 "$PREFIX"
echo "==> Sanity: versions + contrib present"
"$PREFIX/bin/postgres" --version
"$PREFIX/bin/initdb"   --version
ls "$PREFIX/lib/postgresql/unaccent.so" "$PREFIX/lib/postgresql/pg_trgm.so"
ls "$PREFIX/share/postgresql/extension/unaccent.control" \
   "$PREFIX/share/postgresql/extension/pg_trgm.control"
echo "==> Done."
