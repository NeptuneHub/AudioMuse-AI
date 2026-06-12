#!/usr/bin/env bash
#
# Build a relocatable PostgreSQL (server + client tools + the ``unaccent`` and
# ``pg_trgm`` contrib extensions) from source and install it under
# ``native-build/linux/vendor/postgres/<arch>/`` where the PyInstaller spec bundles it as
# ``pgsql/``. Used for the **aarch64** Linux build, where ``pgserver`` has no
# wheel.
#
# Run from the repo root on the target architecture (the CI workflow runs it on
# the aarch64 runner):
#     bash native-build/linux/vendor/postgres/build-postgres.sh
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
PREFIX="$(pwd)/native-build/linux/vendor/postgres/${ARCH}"

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

echo "==> Stripping debug symbols from binaries/libraries"
# strip --strip-unneeded is safe for executables and shared objects (keeps the
# dynamic symbols needed at load time, drops debug/symbol tables).
find "$PREFIX/bin" -type f -exec strip --strip-unneeded {} + 2>/dev/null || true
find "$PREFIX/lib" -type f \( -name '*.so' -o -name '*.so.*' \) -exec strip --strip-unneeded {} + 2>/dev/null || true

echo "==> Installed tree:"
ls -1 "$PREFIX"
echo "==> Sanity: versions + contrib present"
"$PREFIX/bin/postgres" --version
"$PREFIX/bin/initdb"   --version
# A from-source --prefix install uses the plain layout (pkglibdir=$PREFIX/lib,
# sharedir=$PREFIX/share), not the Debian-style lib/postgresql + share/postgresql,
# and the exact subdir can vary, so locate the artifacts rather than hardcoding.
fail=0
for f in unaccent.so pg_trgm.so unaccent.control pg_trgm.control; do
  hit="$(find "$PREFIX" -name "$f" -print -quit)"
  if [ -n "$hit" ]; then
    echo "    found $f -> $hit"
  else
    echo "::error::contrib artifact not found after install: $f"; fail=1
  fi
done
[ "$fail" -eq 0 ] || { echo "::error::PostgreSQL contrib build incomplete"; exit 1; }
echo "==> Done."
