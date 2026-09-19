# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The disposable Postgres behind the end-to-end stack.

Mirrors test/integration/conftest.py: AUDIOMUSE_TEST_DATABASE_URL names a
disposable database (CI's service container) and is a hard failure when it is
unreachable, otherwise an ephemeral pgserver cluster is started in the run's
data directory. The app never reads a DSN (config.py derives DATABASE_URL from
the five POSTGRES_* parts), so the DSN is split into those parts here, the same
way native-build/native_common/child_env.py does for the embedded server.

Main Features:
* resolve returns a PgHandle with the DSN, the POSTGRES_* parts and, for
  pgserver, the bin directory holding pg_dump and psql for the backup page
* split_dsn understands pgserver's socket-directory DSN (host in the query)
* reset_public_schema empties the database so every run starts from a fresh
  catalogue and init_db rebuilds the schema
"""

import os
from urllib.parse import parse_qs, unquote, urlsplit

import psycopg2

from .errors import StackError

ENV_DSN = 'AUDIOMUSE_TEST_DATABASE_URL'
REQUIRED_EXTENSIONS = ('unaccent', 'pg_trgm')

INSTALL_HINT = (
    'No end-to-end database. Set AUDIOMUSE_TEST_DATABASE_URL to a disposable '
    'database, or `pip install pgserver==0.1.4` for an ephemeral local cluster.'
)


class PgHandle:
    def __init__(self, dsn, parts, bin_dir=None, server=None):
        self.dsn = dsn
        self.parts = parts
        self.bin_dir = bin_dir
        self.server = server

    def cleanup(self):
        if self.server is not None:
            try:
                self.server.cleanup()
            finally:
                self.server = None


def split_dsn(dsn):
    url = urlsplit(dsn)
    query = parse_qs(url.query)
    host = (query.get('host') or [url.hostname or 'localhost'])[0]
    port = str(url.port or (query.get('port') or ['5432'])[0])
    dbname = (url.path or '').lstrip('/') or (query.get('dbname') or ['postgres'])[0]
    return {
        'POSTGRES_USER': unquote(url.username or ''),
        'POSTGRES_PASSWORD': unquote(url.password or ''),
        'POSTGRES_HOST': host,
        'POSTGRES_PORT': port,
        'POSTGRES_DB': dbname,
    }


def connect(dsn, autocommit=True):
    conn = psycopg2.connect(dsn)
    conn.autocommit = autocommit
    return conn


def resolve(data_root):
    dsn = os.environ.get(ENV_DSN, '').strip()
    if dsn:
        try:
            psycopg2.connect(dsn).close()
        except Exception as exc:
            raise StackError(
                f'{ENV_DSN} is set but not reachable, refusing to skip: {exc}'
            ) from exc
        return PgHandle(dsn, split_dsn(dsn))
    try:
        import pgserver
    except ImportError:
        return None
    install_dir = os.path.join(os.path.dirname(pgserver.__file__), 'pginstall')
    missing = [
        ext for ext in REQUIRED_EXTENSIONS
        if not os.path.isfile(os.path.join(install_dir, 'share', 'postgresql', 'extension', f'{ext}.control'))
    ]
    if missing:
        raise StackError(
            f'pgserver at {install_dir} lacks the {missing} extension(s) that init_db creates; '
            'run test/e2e/run_local.sh once (it builds them with '
            'native-build/linux/vendor/pg-contrib/build-pg-contrib.sh) or set '
            f'{ENV_DSN} to a full PostgreSQL such as `docker run -d -p 5432:5432 '
            '-e POSTGRES_PASSWORD=postgres postgres:15-alpine`'
        )
    data_dir = os.path.join(data_root, 'pg')
    server = pgserver.get_server(data_dir)
    dsn = server.get_uri()
    bin_dir = os.path.join(os.path.dirname(pgserver.__file__), 'pginstall', 'bin')
    return PgHandle(dsn, split_dsn(dsn), bin_dir=bin_dir, server=server)


def reset_public_schema(dsn):
    conn = connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute('DROP SCHEMA public CASCADE')
            cur.execute('CREATE SCHEMA public')
    finally:
        conn.close()


def scalar(dsn, sql, params=None):
    conn = connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return None if row is None else row[0]
    finally:
        conn.close()
