# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Plugin schema self-heal tests against a real Postgres database.

Reproduces the worker boot crash where the RQ worker reads the plugins registry
before the web process has run init_db, then verifies that ensure_plugins_table
creates the table so the read no longer raises UndefinedTable.

Main Features:
* Reading plugins before creation raises UndefinedTable (the reported failure).
* ensure_plugins_table creates the table, is idempotent, and list_plugins returns [].
"""

import os
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None

pytestmark = pytest.mark.integration


@pytest.fixture(scope='session')
def pg_dsn():
    if psycopg2 is None:
        pytest.skip("psycopg2 not importable")
    dsn = os.environ.get('AUDIOMUSE_TEST_DATABASE_URL')
    if dsn:
        try:
            psycopg2.connect(dsn).close()
        except Exception as e:
            pytest.skip(f"AUDIOMUSE_TEST_DATABASE_URL not reachable: {e}")
        yield dsn
        return
    try:
        import pgserver
    except Exception:
        pytest.skip(
            "No test database. Set AUDIOMUSE_TEST_DATABASE_URL to a disposable "
            "DB, or `pip install pgserver` for an ephemeral local instance."
        )
    data_dir = tempfile.mkdtemp(prefix='audiomuse_pg_')
    server = pgserver.get_server(data_dir)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture
def fresh_db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS plugins CASCADE")
    conn.commit()
    yield conn
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS plugins CASCADE")
    conn.commit()
    conn.close()


def _table_exists(conn, name):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (name,))
        return cur.fetchone()[0] is not None


class TestEnsurePluginsTable:
    def test_reading_registry_before_creation_raises(self, fresh_db):
        import database

        assert not _table_exists(fresh_db, 'plugins')
        with pytest.raises(psycopg2.errors.UndefinedTable):
            database.list_plugins(fresh_db)
        fresh_db.rollback()

    def test_ensure_creates_table_and_registry_reads_empty(self, fresh_db):
        import database

        database.ensure_plugins_table(fresh_db)
        assert _table_exists(fresh_db, 'plugins')
        assert database.list_plugins(fresh_db) == []

    def test_ensure_is_idempotent(self, fresh_db):
        import database

        database.ensure_plugins_table(fresh_db)
        database.ensure_plugins_table(fresh_db)
        assert database.list_plugins(fresh_db) == []
