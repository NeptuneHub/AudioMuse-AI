#!/usr/bin/env bash
# ============================================================================
# Cross-compile unaccent and pg_trgm PostgreSQL contrib extensions for Windows
# from a Linux CI runner (mingw-w64).
#
# These are the same extensions that macOS/Linux build, just compiled for
# Windows against the PostgreSQL 16.2 ABI that pgserver bundles.
#
# Prerequisites: mingw-w64, postgresql-server-dev-16 (for headers)
# Output: windows/vendor/pg-contrib/amd64/
# ============================================================================
set -euo pipefail

ARCH="amd64"
PG_MAJOR="16"
DEST="windows/vendor/pg-contrib/${ARCH}"

echo "==> Installing build dependencies"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    gcc-mingw-w64-x86-64 \
    postgresql-server-dev-${PG_MAJOR} \
    make

# PG's pg_config tells us where the headers live
PG_INCLUDEDIR=$(pg_config --includedir-server)
PG_PKGLIBDIR=$(pg_config --pkglibdir)
PG_SHAREDIR=$(pg_config --sharedir)

echo "PG includedir-server: ${PG_INCLUDEDIR}"
echo "PG pkglibdir:        ${PG_PKGLIBDIR}"
echo "PG sharedir:          ${PG_SHAREDIR}"

rm -rf "${DEST}"
mkdir -p "${DEST}/lib" "${DEST}/extension" "${DEST}/tsearch_data"

# Cross-compiler prefix for x86_64 Windows
CC=x86_64-w64-mingw32-gcc
CFLAGS="-O2 -Wall -fPIC -I${PG_INCLUDEDIR}"

# We need to compile the contrib modules without their Makefiles.
# The simplest approach: compile the single .c files directly.
# unaccent.c is part of the PostgreSQL contrib source.

# Download PostgreSQL 16.2 source (for the contrib .c files only)
PG_SRC_DIR="/tmp/postgresql-windows-contrib"
if [ ! -d "${PG_SRC_DIR}" ]; then
    echo "==> Downloading PostgreSQL 16.2 source (contrib only)"
    mkdir -p "${PG_SRC_DIR}"
    curl -fsSL "https://ftp.postgresql.org/pub/source/v16.2/postgresql-16.2.tar.gz" \
        | tar xz -C "${PG_SRC_DIR}" --strip-components=1
fi

# Build unaccent.dll
echo "==> Building unaccent.dll"
${CC} ${CFLAGS} -shared \
    -o "${DEST}/lib/unaccent.dll" \
    "${PG_SRC_DIR}/contrib/unaccent/unaccent.c" \
    -lws2_32

# Build pg_trgm.dll
echo "==> Building pg_trgm.dll"
${CC} ${CFLAGS} -shared \
    -o "${DEST}/lib/pg_trgm.dll" \
    "${PG_SRC_DIR}/contrib/pg_trgm/trgm_op.c" \
    "${PG_SRC_DIR}/contrib/pg_trgm/trgm_gist.c" \
    "${PG_SRC_DIR}/contrib/pg_trgm/trgm_gin.c" \
    "${PG_SRC_DIR}/contrib/pg_trgm/trgm_regexp.c" \
    -lws2_32

# Copy control files and data
cp "${PG_SRC_DIR}/contrib/unaccent/unaccent.control"      "${DEST}/extension/"
cp "${PG_SRC_DIR}/contrib/unaccent/unaccent--1.1.sql"     "${DEST}/extension/"
cp "${PG_SRC_DIR}/contrib/unaccent/unaccent.rules"        "${DEST}/tsearch_data/"
cp "${PG_SRC_DIR}/contrib/pg_trgm/pg_trgm.control"        "${DEST}/extension/"
cp "${PG_SRC_DIR}/contrib/pg_trgm/pg_trgm--1.6.sql"       "${DEST}/extension/"

echo "==> Verifying outputs"
for f in \
    "${DEST}/lib/unaccent.dll" \
    "${DEST}/lib/pg_trgm.dll" \
    "${DEST}/extension/unaccent.control" \
    "${DEST}/extension/pg_trgm.control" \
    "${DEST}/tsearch_data/unaccent.rules"
do
    if [ ! -s "$f" ]; then
        echo "::error::Missing output: $f"
        exit 1
    fi
    file "$f" 2>/dev/null || true
done

echo "==> Done. Vendor files in ${DEST}/"
du -sh "${DEST}"
