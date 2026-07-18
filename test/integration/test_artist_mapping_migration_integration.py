# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Drive the one-time artist_mapping -> artist_server_map fold+drop against real PG.

The legacy default-only ``artist_mapping`` table is folded into
``artist_server_map`` (keyed by the default server) and then DROPPED at startup, so
``artist_server_map`` becomes the single source of truth. This proves the real DDL:
the rows land under the default server, the table is gone afterwards, a second run
is a crash-free no-op, and NULL-id / conflicting / no-default edge cases behave.

Main Features:
* Legacy rows fold into artist_server_map under the default server, table dropped.
* ON CONFLICT keeps an existing artist_server_map row; NULL artist_id is skipped.
* Idempotent: once dropped, re-running is an instant no-op (never errors).
* No default server: an empty legacy table is dropped, a non-empty one is kept.
"""

import os

import pytest

try:
    import psycopg2
except Exception:  # pragma: no cover
    psycopg2 = None

pytestmark = pytest.mark.integration

_SCHEMA = [
    "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, "
    "is_default BOOLEAN NOT NULL DEFAULT FALSE)",
    "CREATE TABLE artist_server_map ("
    "artist_name TEXT NOT NULL, "
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_artist_id TEXT NOT NULL, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "PRIMARY KEY (artist_name, server_id), UNIQUE (server_id, provider_artist_id))",
    "CREATE TABLE artist_mapping (artist_name TEXT PRIMARY KEY, artist_id TEXT)",
]


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
        pytest.skip("neither AUDIOMUSE_TEST_DATABASE_URL nor pgserver is available")
    import tempfile

    with tempfile.TemporaryDirectory() as data_dir:
        server = pgserver.get_server(data_dir)
        try:
            yield server.get_uri()
        finally:
            server.cleanup()


@pytest.fixture
def db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    with conn.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS artist_server_map, artist_mapping, music_servers CASCADE"
        )
        for ddl in _SCHEMA:
            cur.execute(ddl)
    conn.commit()
    yield conn
    conn.close()


def _run(conn):
    from database import _migrate_artist_mapping_to_server_map

    with conn.cursor() as cur:
        _migrate_artist_mapping_to_server_map(cur)
    conn.commit()


def _table_exists(conn, name):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f'public.{name}',))
        return cur.fetchone()[0] is not None


def _server_map(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT artist_name, server_id, provider_artist_id "
            "FROM artist_server_map ORDER BY artist_name"
        )
        return cur.fetchall()


class TestArtistMappingMigration:
    def test_folds_into_default_server_then_drops_and_is_idempotent(self, db):
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO music_servers (server_id, name, is_default) "
                "VALUES ('srv', 'Default', TRUE)"
            )
            cur.executemany(
                "INSERT INTO artist_mapping (artist_name, artist_id) VALUES (%s, %s)",
                [('Daft Punk', 'a1'), ('Air', 'a2'), ('NoId', None)],
            )
        db.commit()

        _run(db)

        # a1/a2 folded under the default server; the NULL-id row is skipped.
        assert _server_map(db) == [
            ('Air', 'srv', 'a2'),
            ('Daft Punk', 'srv', 'a1'),
        ]
        # The legacy table is gone.
        assert not _table_exists(db, 'artist_mapping')
        # Second run is a crash-free no-op (table already dropped).
        _run(db)
        assert not _table_exists(db, 'artist_mapping')

    def test_existing_server_map_row_wins_over_legacy(self, db):
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO music_servers (server_id, name, is_default) "
                "VALUES ('srv', 'Default', TRUE)"
            )
            cur.execute(
                "INSERT INTO artist_server_map (artist_name, server_id, provider_artist_id) "
                "VALUES ('Daft Punk', 'srv', 'current')"
            )
            cur.execute(
                "INSERT INTO artist_mapping (artist_name, artist_id) VALUES ('Daft Punk', 'stale')"
            )
        db.commit()

        _run(db)

        assert _server_map(db) == [('Daft Punk', 'srv', 'current')]
        assert not _table_exists(db, 'artist_mapping')

    def test_no_default_server_drops_empty_but_keeps_non_empty(self, db):
        # No default server: an empty legacy table is safe to drop.
        _run(db)
        assert not _table_exists(db, 'artist_mapping')

    def test_no_default_server_keeps_non_empty_for_next_boot(self, db):
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO artist_mapping (artist_name, artist_id) VALUES ('Air', 'a2')"
            )
        db.commit()

        _run(db)

        # Nothing to attribute the rows to yet: keep the table for a later boot.
        assert _table_exists(db, 'artist_mapping')
        assert _server_map(db) == []
