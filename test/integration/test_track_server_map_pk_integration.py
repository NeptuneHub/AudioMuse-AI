# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""track_server_map primary-key relaxation, against a real Postgres database.

Proves the schema migration that lets N provider files map to one canonical song
per server, and that the resulting many-to-one map is read deterministically.

Main Features:
* The old (item_id, server_id) PK is swapped to (server_id, provider_track_id),
  the item-leading index is created, and the reverse index is dropped.
* Migration is idempotent and never accumulates a duplicate unique index.
* Two provider files for one item_id on one server can coexist after migration.
* translate_ids picks one provider id deterministically (strongest tier wins).
* The real provider-migration transaction collapses the duplicate default-server
  rows of one item before it stamps the new provider id, so the relaxed PK is
  never violated, and it leaves every non-default server untouched.
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
            pytest.fail(f"AUDIOMUSE_TEST_DATABASE_URL is set but not reachable, refusing to skip: {e}")
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


def _drop_all(conn):
    with conn.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS track_server_map, artist_server_map, "
            "music_servers, score CASCADE"
        )
    conn.commit()


@pytest.fixture
def old_schema_db(pg_dsn):
    """A database in the PRE-migration shape: track_server_map with the old
    (item_id, server_id) PK plus the (server_id, provider_track_id) unique index,
    one score row, one server, one mapping."""
    conn = psycopg2.connect(pg_dsn)
    _drop_all(conn)
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT)")
        cur.execute(
            "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, "
            "server_type TEXT, creds JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "music_libraries TEXT NOT NULL DEFAULT '', "
            "is_default BOOLEAN NOT NULL DEFAULT FALSE, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cur.execute(
            "CREATE TABLE track_server_map ("
            "item_id TEXT NOT NULL REFERENCES score (item_id) "
            "ON UPDATE CASCADE ON DELETE CASCADE, "
            "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
            "provider_track_id TEXT NOT NULL, match_tier TEXT, file_path TEXT, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "PRIMARY KEY (item_id, server_id))"
        )
        cur.execute(
            "CREATE UNIQUE INDEX idx_track_server_map_provider_unique "
            "ON track_server_map (server_id, provider_track_id)"
        )
        cur.execute(
            "CREATE INDEX idx_track_server_map_reverse "
            "ON track_server_map (server_id, provider_track_id)"
        )
        cur.execute("INSERT INTO score (item_id) VALUES ('X')")
        cur.execute(
            "INSERT INTO music_servers (server_id, name, server_type, is_default) "
            "VALUES ('srv', 'Srv', 'jellyfin', TRUE)"
        )
        cur.execute(
            "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier) "
            "VALUES ('X', 'srv', 'provA', 'fingerprint')"
        )
    conn.commit()
    yield conn
    _drop_all(conn)
    conn.close()


def _pk_columns(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT array_agg(a.attname ORDER BY a.attname) FROM pg_constraint c "
            "JOIN unnest(c.conkey) k ON TRUE "
            "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k "
            "WHERE c.conrelid = 'track_server_map'::regclass AND c.contype = 'p'"
        )
        return sorted(cur.fetchone()[0])


def _index_exists(conn, name):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (name,))
        return cur.fetchone()[0] is not None


def _unique_indexes_on_provider(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_index i "
            "WHERE i.indrelid = 'track_server_map'::regclass AND i.indisunique "
            "AND i.indkey::text = ("
            "  SELECT string_agg(a.attnum::text, ' ' ORDER BY ord) FROM ("
            "    SELECT attnum, ord FROM pg_attribute a2 "
            "    JOIN (VALUES ('server_id',1),('provider_track_id',2)) v(cn,ord) "
            "      ON a2.attname = v.cn "
            "    WHERE a2.attrelid = 'track_server_map'::regclass) a)"
        )
        return cur.fetchone()[0]


def _prepare_migration_session(conn):
    import database

    with conn.cursor() as cur:
        database.relax_track_server_map_pk(cur)
        cur.execute("ALTER TABLE music_servers ADD COLUMN IF NOT EXISTS track_count INTEGER")
        cur.execute("DROP TABLE IF EXISTS migration_target_meta, migration_session CASCADE")
        cur.execute(
            "CREATE TABLE migration_session (id SERIAL PRIMARY KEY, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, completed_at TIMESTAMP, "
            "status TEXT NOT NULL DEFAULT 'in_progress', source_type TEXT NOT NULL, "
            "target_type TEXT NOT NULL, target_creds TEXT NOT NULL, "
            "state JSONB NOT NULL DEFAULT '{}')"
        )
        cur.execute(
            "INSERT INTO migration_session (source_type, target_type, target_creds) "
            "VALUES ('jellyfin', 'navidrome', '{}') RETURNING id"
        )
        session_id = cur.fetchone()[0]
    conn.commit()
    return session_id


class TestRelaxTrackServerMapPk:
    def test_swaps_pk_and_indexes(self, old_schema_db):
        import database

        assert _pk_columns(old_schema_db) == ['item_id', 'server_id']
        with old_schema_db.cursor() as cur:
            changed = database.relax_track_server_map_pk(cur)
        old_schema_db.commit()

        assert changed is True
        assert _pk_columns(old_schema_db) == ['provider_track_id', 'server_id']
        assert _index_exists(old_schema_db, 'idx_track_server_map_item')
        assert not _index_exists(old_schema_db, 'idx_track_server_map_reverse')

    def test_second_run_is_a_noop_without_duplicate_index(self, old_schema_db):
        import database

        with old_schema_db.cursor() as cur:
            assert database.relax_track_server_map_pk(cur) is True
        old_schema_db.commit()
        with old_schema_db.cursor() as cur:
            assert database.relax_track_server_map_pk(cur) is False
        old_schema_db.commit()

        assert _unique_indexes_on_provider(old_schema_db) == 1

    def test_two_providers_for_one_item_coexist_after_migration(self, old_schema_db):
        import database

        with old_schema_db.cursor() as cur:
            database.relax_track_server_map_pk(cur)
        old_schema_db.commit()

        with old_schema_db.cursor() as cur:
            cur.execute(
                "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier) "
                "VALUES ('X', 'srv', 'provB', 'fingerprint')"
            )
        old_schema_db.commit()

        with old_schema_db.cursor() as cur:
            cur.execute(
                "SELECT count(*), count(DISTINCT item_id) FROM track_server_map "
                "WHERE server_id = 'srv'"
            )
            rows_total, unique_songs = cur.fetchone()
        assert rows_total == 2
        assert unique_songs == 1

    def test_one_song_many_servers_many_files_each_keeps_its_own_path(self, old_schema_db):
        """The shape of the catalogue after migration, end to end.

        ONE AudioMuse id in `score`. Every FILE that carries that audio - on any
        server, including two duplicate copies on the SAME server - is its own
        track_server_map row, with its own provider id and its OWN path. The shared
        score row carries no path at all.
        """
        import database
        from tasks.mediaserver import registry

        with old_schema_db.cursor() as cur:
            database.relax_track_server_map_pk(cur)
            cur.execute("DELETE FROM track_server_map")
            cur.execute(
                "INSERT INTO music_servers (server_id, name, server_type, is_default) "
                "VALUES ('plex', 'Plex', 'plex', FALSE)"
            )
        old_schema_db.commit()

        # Jellyfin holds TWO copies of the same audio; Plex holds one, elsewhere.
        registry.upsert_track_maps(
            'srv',
            {
                'jf-1': ('X', 'fingerprint', '/music/Duran Duran/Rio/03 Rio.flac'),
                'jf-2': ('X', 'fingerprint', '/music/Compilations/80s Hits/07 Rio.flac'),
            },
            conn=old_schema_db,
        )
        registry.upsert_track_maps(
            'plex',
            {'plex-9': ('X', 'fingerprint', '/data/media/Duran Duran/Rio/03 Rio.flac')},
            conn=old_schema_db,
        )
        old_schema_db.commit()

        with old_schema_db.cursor() as cur:
            cur.execute("SELECT count(*) FROM score")
            assert cur.fetchone()[0] == 1, "one audio = one AudioMuse id, always"

            cur.execute(
                "SELECT server_id, provider_track_id, file_path FROM track_server_map "
                "WHERE item_id = 'X' ORDER BY server_id, provider_track_id"
            )
            rows = cur.fetchall()

        assert rows == [
            ('plex', 'plex-9', '/data/media/Duran Duran/Rio/03 Rio.flac'),
            ('srv', 'jf-1', '/music/Duran Duran/Rio/03 Rio.flac'),
            ('srv', 'jf-2', '/music/Compilations/80s Hits/07 Rio.flac'),
        ]

        # Three files, three paths, one song. This is the evidence the sweep matcher
        # now offers a NEW server: every path the catalogue knows, not the default's.
        with old_schema_db.cursor() as cur:
            cur.execute(
                "SELECT ARRAY(SELECT DISTINCT p.file_path FROM track_server_map p "
                "WHERE p.item_id = s.item_id AND p.file_path IS NOT NULL) "
                "FROM score s WHERE s.item_id = 'X'"
            )
            known_paths = cur.fetchone()[0]
        assert sorted(known_paths) == [
            '/data/media/Duran Duran/Rio/03 Rio.flac',
            '/music/Compilations/80s Hits/07 Rio.flac',
            '/music/Duran Duran/Rio/03 Rio.flac',
        ]

    def test_a_path_less_writer_never_erases_a_path_a_sweep_recorded(self, old_schema_db):
        """Analysis may not know the path; a sweep does. The COALESCE keeps it."""
        import database
        from tasks.mediaserver import registry

        with old_schema_db.cursor() as cur:
            database.relax_track_server_map_pk(cur)
            cur.execute("DELETE FROM track_server_map")
        old_schema_db.commit()

        registry.upsert_track_maps(
            'srv', {'p1': ('X', 'path', '/music/song.flac')}, conn=old_schema_db
        )
        registry.upsert_track_maps(
            'srv', {'p1': ('X', 'fingerprint')}, conn=old_schema_db
        )
        old_schema_db.commit()

        with old_schema_db.cursor() as cur:
            cur.execute("SELECT match_tier, file_path FROM track_server_map")
            assert cur.fetchone() == ('fingerprint', '/music/song.flac')

    def test_translate_ids_picks_strongest_tier_deterministically(self, old_schema_db):
        import database
        from tasks.mediaserver import registry

        with old_schema_db.cursor() as cur:
            database.relax_track_server_map_pk(cur)
            # A weaker-tier row whose provider id sorts BEFORE the fingerprint
            # row, so only tier priority (not id ordering) can select 'provA'.
            cur.execute("UPDATE track_server_map SET provider_track_id = 'zzz' WHERE item_id = 'X'")
            cur.execute(
                "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier) "
                "VALUES ('X', 'srv', 'aaa', 'analysis')"
            )
        old_schema_db.commit()

        first = registry.translate_ids(['X'], 'srv', conn=old_schema_db)
        second = registry.translate_ids(['X'], 'srv', conn=old_schema_db)
        assert first == {'X': 'zzz'}
        assert second == first

    def test_upsert_self_heals_when_key_is_missing(self, old_schema_db):
        """A DB that never got the (server_id, provider_track_id) key (a worker
        wrote before the startup migration, or a restore of an older schema)
        must not crash the album: upsert_track_maps ensures the schema and
        retries, and both provider ids for one song end up mapped."""
        from tasks.mediaserver import registry

        with old_schema_db.cursor() as cur:
            cur.execute("DROP INDEX idx_track_server_map_provider_unique")
        old_schema_db.commit()

        registry.upsert_track_maps('srv', {'provNEW': ('X', 'fingerprint')}, conn=old_schema_db)

        assert _pk_columns(old_schema_db) == ['provider_track_id', 'server_id']
        with old_schema_db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM track_server_map WHERE server_id = 'srv' AND item_id = 'X'"
            )
            assert cur.fetchone()[0] == 2

    def test_run_migration_transaction_collapses_duplicate_default_rows_before_repointing_ids(
        self, old_schema_db
    ):
        from tasks import provider_migration_tasks as mig

        session_id = _prepare_migration_session(old_schema_db)
        with old_schema_db.cursor() as cur:
            cur.execute(
                "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier) "
                "VALUES ('X', 'srv', 'provB', 'fingerprint')"
            )
            cur.execute("SELECT count(*) FROM track_server_map WHERE item_id = 'X'")
            assert cur.fetchone()[0] == 2
        old_schema_db.commit()

        with old_schema_db.cursor() as cur:
            mig._run_migration_transaction(
                cur, {'X': 'navi-1'}, {}, 'navidrome', {}, session_id
            )
            cur.execute(
                "SELECT item_id, server_id, provider_track_id, match_tier "
                "FROM track_server_map ORDER BY provider_track_id"
            )
            rows = cur.fetchall()
        old_schema_db.commit()

        assert rows == [('X', 'srv', 'navi-1', 'default')]

    def test_run_migration_transaction_repoints_every_server_only_on_the_default_one(
        self, old_schema_db
    ):
        from tasks import provider_migration_tasks as mig

        session_id = _prepare_migration_session(old_schema_db)
        with old_schema_db.cursor() as cur:
            cur.execute(
                "INSERT INTO music_servers (server_id, name, server_type, is_default) "
                "VALUES ('plex', 'Plex', 'plex', FALSE)"
            )
            cur.execute(
                "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier) "
                "VALUES ('X', 'plex', 'plex-9', 'fingerprint')"
            )
        old_schema_db.commit()

        with old_schema_db.cursor() as cur:
            mig._run_migration_transaction(
                cur, {'X': 'navi-1'}, {}, 'navidrome', {}, session_id
            )
            cur.execute(
                "SELECT server_id, provider_track_id, match_tier "
                "FROM track_server_map ORDER BY server_id"
            )
            rows = cur.fetchall()
        old_schema_db.commit()

        assert rows == [
            ('plex', 'plex-9', 'fingerprint'),
            ('srv', 'navi-1', 'default'),
        ]
