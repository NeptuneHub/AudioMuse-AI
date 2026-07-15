# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Real-Postgres coverage of the playlist server_id migration and scoped writes.

Drives database._migrate_playlist_server_column, update_playlist_table and
prune_playlist_rows_for_missing_servers against an ephemeral Postgres,
asserting the playlist table stays a last-run-per-server snapshot.

Main Features:
* The migration backfills legacy rows to the default server, replaces the old
  two-column unique constraint with the three-column index, reruns cleanly,
  and survives NULL rows colliding with the default server's rows
* The same bare playlist name coexists on two servers after migration
* A scoped write replaces only its own server's rows; repeated runs never grow
  the table; pruning drops rows of servers no longer configured; a server
  deleted mid-run gets no ghost rows re-inserted
"""

import os

import pytest

try:
    import psycopg2
except Exception:
    psycopg2 = None

pytestmark = pytest.mark.integration


_LEGACY_SCHEMA = [
    "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, "
    "server_type TEXT, creds JSONB DEFAULT '{}', music_libraries TEXT DEFAULT '', "
    "is_default BOOLEAN NOT NULL DEFAULT FALSE, track_count INTEGER, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE playlist (id SERIAL PRIMARY KEY, playlist_name TEXT, "
    "item_id TEXT, title TEXT, author TEXT, UNIQUE (playlist_name, item_id))",
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
    yield conn
    conn.close()


def _build(conn, servers, legacy_rows=()):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS playlist, music_servers CASCADE")
        for ddl in _LEGACY_SCHEMA:
            cur.execute(ddl)
        for server_id, name, is_default in servers:
            cur.execute(
                "INSERT INTO music_servers (server_id, name, server_type, is_default) "
                "VALUES (%s, %s, 'jellyfin', %s)",
                (server_id, name, is_default),
            )
        for name, item_id in legacy_rows:
            cur.execute(
                "INSERT INTO playlist (playlist_name, item_id, title, author) "
                "VALUES (%s, %s, %s, 'Artist')",
                (name, item_id, f'Title {item_id}'),
            )
    conn.commit()


def _migrate(conn):
    from database import _migrate_playlist_server_column

    with conn.cursor() as cur:
        _migrate_playlist_server_column(cur)
    conn.commit()


def _rows(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT playlist_name, item_id, server_id FROM playlist "
            "ORDER BY playlist_name, item_id, server_id"
        )
        return cur.fetchall()


def _use_conn(monkeypatch, conn):
    import database

    monkeypatch.setattr(database, 'get_db', lambda: conn)


class TestPlaylistServerColumnMigration:
    def test_migration_backfills_legacy_rows_to_the_default_server(self, db):
        _build(
            db,
            servers=[('s1', 'One', True), ('s2', 'Two', False)],
            legacy_rows=[('Rock_automatic', 'i1'), ('Jazz_automatic', 'i2')],
        )
        _migrate(db)
        assert _rows(db) == [
            ('Jazz_automatic', 'i2', 's1'),
            ('Rock_automatic', 'i1', 's1'),
        ]

    def test_migration_lets_the_same_bare_name_exist_on_two_servers(self, db):
        _build(db, servers=[('s1', 'One', True), ('s2', 'Two', False)])
        _migrate(db)
        with db.cursor() as cur:
            for server_id in ('s1', 's2'):
                cur.execute(
                    "INSERT INTO playlist (playlist_name, item_id, title, author, server_id) "
                    "VALUES ('Rock_automatic', 'i1', 'Title i1', 'Artist', %s)",
                    (server_id,),
                )
        db.commit()
        assert _rows(db) == [
            ('Rock_automatic', 'i1', 's1'),
            ('Rock_automatic', 'i1', 's2'),
        ]

    def test_migration_rejects_duplicate_rows_for_the_same_server(self, db):
        _build(db, servers=[('s1', 'One', True)])
        _migrate(db)
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO playlist (playlist_name, item_id, title, author, server_id) "
                "VALUES ('Rock_automatic', 'i1', 'Title i1', 'Artist', 's1')"
            )
            with pytest.raises(psycopg2.errors.UniqueViolation):
                cur.execute(
                    "INSERT INTO playlist (playlist_name, item_id, title, author, server_id) "
                    "VALUES ('Rock_automatic', 'i1', 'Title i1', 'Artist', 's1')"
                )
        db.rollback()

    def test_migration_runs_twice_without_error(self, db):
        _build(
            db,
            servers=[('s1', 'One', True)],
            legacy_rows=[('Rock_automatic', 'i1')],
        )
        _migrate(db)
        _migrate(db)
        assert _rows(db) == [('Rock_automatic', 'i1', 's1')]

    def test_migration_survives_null_rows_colliding_with_default_rows(self, db):
        _build(db, servers=[('s1', 'One', True)])
        _migrate(db)
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO playlist (playlist_name, item_id, title, author, server_id) "
                "VALUES ('Rock_automatic', 'i1', 'Title i1', 'Artist', 's1')"
            )
            cur.execute(
                "INSERT INTO playlist (playlist_name, item_id, title, author) "
                "VALUES ('Rock_automatic', 'i1', 'Title i1', 'Artist'), "
                "('Rock_automatic', 'i1', 'Title i1', 'Artist'), "
                "('Jazz_automatic', 'i2', 'Title i2', 'Artist')"
            )
        db.commit()
        _migrate(db)
        assert _rows(db) == [
            ('Jazz_automatic', 'i2', 's1'),
            ('Rock_automatic', 'i1', 's1'),
        ]


class TestScopedPlaylistWrites:
    def test_scoped_write_leaves_the_other_servers_rows_untouched(self, db, monkeypatch):
        from database import update_playlist_table

        _build(db, servers=[('s1', 'One', True), ('s2', 'Two', False)])
        _migrate(db)
        _use_conn(monkeypatch, db)
        update_playlist_table({'Rock_automatic': [('i1', 'Title i1', 'Artist')]}, 's1')
        update_playlist_table({'Jazz_automatic': [('i2', 'Title i2', 'Artist')]}, 's2')
        update_playlist_table({'Pop_automatic': [('i3', 'Title i3', 'Artist')]}, 's1')
        assert _rows(db) == [
            ('Jazz_automatic', 'i2', 's2'),
            ('Pop_automatic', 'i3', 's1'),
        ]

    def test_repeated_runs_never_grow_the_table_beyond_the_last_run_per_server(
        self, db, monkeypatch
    ):
        from database import update_playlist_table

        _build(db, servers=[('s1', 'One', True), ('s2', 'Two', False)])
        _migrate(db)
        _use_conn(monkeypatch, db)
        playlists = {
            'Rock_automatic': [('i1', 'Title i1', 'Artist'), ('i2', 'Title i2', 'Artist')],
            'Jazz_automatic': [('i3', 'Title i3', 'Artist')],
        }
        for _run in range(3):
            update_playlist_table(playlists, 's1')
            update_playlist_table(playlists, 's2')
        assert len(_rows(db)) == 6

    def test_prune_drops_rows_of_servers_no_longer_configured(self, db, monkeypatch):
        from database import update_playlist_table, prune_playlist_rows_for_missing_servers

        _build(db, servers=[('s1', 'One', True), ('s2', 'Two', False)])
        _migrate(db)
        _use_conn(monkeypatch, db)
        update_playlist_table({'Rock_automatic': [('i1', 'Title i1', 'Artist')]}, 's1')
        update_playlist_table({'Jazz_automatic': [('i2', 'Title i2', 'Artist')]}, 's2')
        prune_playlist_rows_for_missing_servers(['s1'])
        assert _rows(db) == [('Rock_automatic', 'i1', 's1')]

    def test_write_for_a_server_deleted_mid_run_inserts_no_ghost_rows(self, db, monkeypatch):
        from database import update_playlist_table

        _build(db, servers=[('s1', 'One', True), ('s2', 'Two', False)])
        _migrate(db)
        _use_conn(monkeypatch, db)
        update_playlist_table({'Jazz_automatic': [('i2', 'Title i2', 'Artist')]}, 's2')
        with db.cursor() as cur:
            cur.execute("DELETE FROM playlist WHERE server_id = 's2'")
            cur.execute("DELETE FROM music_servers WHERE server_id = 's2'")
        db.commit()
        update_playlist_table({'Jazz_automatic': [('i2', 'Title i2', 'Artist')]}, 's2')
        assert _rows(db) == []
