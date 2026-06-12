#!/usr/bin/env bash
# Regenerate native-build/windows/vendor/pg-contrib/<arch>/ : run on Linux with gcc-mingw-w64-x86-64 + python3 + curl.
set -euo pipefail

ARCH="amd64"
PG_VERSION="16.2"
PGSERVER_VERSION="0.1.4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DEST="${DEST:-${REPO_ROOT}/native-build/windows/vendor/pg-contrib/${ARCH}}"

command -v x86_64-w64-mingw32-gcc >/dev/null || { echo "Need gcc-mingw-w64-x86-64 (apt install gcc-mingw-w64-x86-64)"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

echo "==> Fetching pgserver ${PGSERVER_VERSION} Windows wheel (for its exact PG ${PG_VERSION} headers + libpostgres.a)"
python3 -m pip download "pgserver==${PGSERVER_VERSION}" --no-deps \
    --platform win_amd64 --python-version 312 --only-binary=:all: -d "${WORK}/wheel"
python3 -m zipfile -e "${WORK}"/wheel/pgserver-*.whl "${WORK}/pgs"
PGI="${WORK}/pgs/pgserver/pginstall"

echo "==> Fetching PostgreSQL ${PG_VERSION} contrib source (unaccent + pg_trgm)"
curl -fsSL "https://ftp.postgresql.org/pub/source/v${PG_VERSION}/postgresql-${PG_VERSION}.tar.gz" \
    | tar xz -C "${WORK}" "postgresql-${PG_VERSION}/contrib/unaccent" "postgresql-${PG_VERSION}/contrib/pg_trgm"
SRC="${WORK}/postgresql-${PG_VERSION}/contrib"

mkdir -p "${DEST}/lib" "${DEST}/extension" "${DEST}/tsearch_data"

CC=x86_64-w64-mingw32-gcc
INC="-I${PGI}/include/postgresql/server -I${PGI}/include/postgresql/server/port/win32 -I${PGI}/include/postgresql/internal"
LIBS="-L${PGI}/lib -lpostgres -lpgcommon -lpgport -lws2_32"

echo "==> Compiling unaccent.dll"
"${CC}" -O2 -Wall -shared ${INC} -o "${DEST}/lib/unaccent.dll" "${SRC}/unaccent/unaccent.c" ${LIBS}

echo "==> Compiling pg_trgm.dll"
"${CC}" -O2 -Wall -shared ${INC} -I"${SRC}/pg_trgm" -o "${DEST}/lib/pg_trgm.dll" \
    "${SRC}/pg_trgm/trgm_op.c" "${SRC}/pg_trgm/trgm_gist.c" \
    "${SRC}/pg_trgm/trgm_gin.c" "${SRC}/pg_trgm/trgm_regexp.c" ${LIBS}

cp "${SRC}/unaccent/unaccent.control" "${SRC}/unaccent/unaccent--1.1.sql" "${SRC}/unaccent/unaccent--1.0--1.1.sql" "${DEST}/extension/"
cp "${SRC}/unaccent/unaccent.rules" "${DEST}/tsearch_data/"
cp "${SRC}/pg_trgm/pg_trgm.control" "${SRC}"/pg_trgm/pg_trgm--*.sql "${DEST}/extension/"

for f in lib/unaccent.dll lib/pg_trgm.dll extension/unaccent.control extension/pg_trgm.control tsearch_data/unaccent.rules; do
    test -s "${DEST}/${f}" || { echo "ERROR: missing/empty ${DEST}/${f}"; exit 1; }
done

echo "==> Done. Built into ${DEST}"
ls -la "${DEST}/lib"
