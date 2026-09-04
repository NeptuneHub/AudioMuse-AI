# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Drive the Chromaprint DB path against a real PostgreSQL.

Proves the SQL the unit tests mock: persist_chromaprint upserts the compressed
blob and the NULL retry-stop sentinel, _fetch_row_fingerprint maps a canonical
id back to any file's fingerprint via track_server_map, and upsert_track_maps
hands a swept mapping the Chromaprint already stored for its canonical track in
the same transaction.

Main Features:
* persist_chromaprint / get_chromaprint round-trip and the NULL retry-stop
  sentinel reading back as abstain.
* _fetch_row_fingerprint JOIN from a canonical id to any mapped file's blob.
* A sweep match carries the fingerprint with the mapping in the SAME transaction
  that writes it.
* The inherit rides a SAVEPOINT, so a chromaprint table that is missing or
  mid-migration can never roll back the mapping write it travels with.
* A track whose own fpcalc run failed carries the NULL retry-stop sentinel, and
  that row must not lock it out of the hand-over: with the backfill gone this is
  the only path that would ever give it a real print.
* The SOURCE side refuses an ambiguous fingerprint exactly like the target side.
  A source server holding two files for one canonical id cannot say which one the
  stored print belongs to, so handing either on is a guess the gate then treats
  as fact.
"""

import os
import zlib

import numpy as np
import pytest

try:
    import psycopg2
except Exception:  # pragma: no cover
    psycopg2 = None

pytestmark = pytest.mark.integration

_SCHEMA = [
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, album TEXT, "
    "duration DOUBLE PRECISION)",
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA)",
    "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, server_type TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE track_server_map ("
    "item_id TEXT NOT NULL REFERENCES score (item_id) ON DELETE CASCADE, "
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_track_id TEXT NOT NULL, match_tier TEXT, file_path TEXT, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "PRIMARY KEY (server_id, provider_track_id))",
    "CREATE INDEX idx_track_server_map_item ON track_server_map (item_id, server_id)",
    "CREATE TABLE chromaprint ("
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_track_id TEXT NOT NULL, fingerprint BYTEA, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "PRIMARY KEY (server_id, provider_track_id))",
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
            pytest.fail(f"AUDIOMUSE_TEST_DATABASE_URL is set but not reachable, refusing to skip: {e}")
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
            "DROP TABLE IF EXISTS chromaprint, track_server_map, music_servers, "
            "embedding, score CASCADE"
        )
        for ddl in _SCHEMA:
            cur.execute(ddl)
        cur.execute(
            "INSERT INTO music_servers (server_id, name, server_type) "
            "VALUES ('srv', 'Nav', 'navidrome')"
        )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def use_test_db(db, monkeypatch):
    import database
    from tasks.analysis import helper

    monkeypatch.setattr(database, 'get_db', lambda: db)
    monkeypatch.setattr(helper, 'get_db', lambda: db)
    return db


def _add_server(cur, server_id):
    cur.execute(
        "INSERT INTO music_servers (server_id, name, server_type) "
        "VALUES (%s, %s, 'navidrome') ON CONFLICT (server_id) DO NOTHING",
        (server_id, server_id),
    )


def _seed(cur, item_id, provider_id, album, file_path, server_id='srv',
          match_tier='fingerprint'):
    cur.execute(
        "INSERT INTO score (item_id, title, album, duration) VALUES (%s, %s, %s, 200.0) "
        "ON CONFLICT (item_id) DO NOTHING",
        (item_id, item_id, album),
    )
    cur.execute(
        "INSERT INTO embedding (item_id, embedding) VALUES (%s, %s) "
        "ON CONFLICT (item_id) DO NOTHING",
        (item_id, b'\x00\x00'),
    )
    cur.execute(
        "INSERT INTO track_server_map (item_id, server_id, provider_track_id, "
        "match_tier, file_path) VALUES (%s, %s, %s, %s, %s)",
        (item_id, server_id, provider_id, match_tier, file_path),
    )


def _blob(seed):
    return zlib.compress(np.arange(seed, seed + 60, dtype=np.uint32).tobytes())


class TestChromaprintDbPath:
    def test_persist_and_fetch_round_trip(self, db, use_test_db):
        from database import persist_chromaprint
        from tasks.analysis.helper import _fetch_row_fingerprint

        with db.cursor() as cur:
            _seed(cur, 'fp_x', 'p1', 'Alb', '/m/p1.flac')
        db.commit()

        blob = _blob(1)
        persist_chromaprint('srv', 'p1', blob)
        assert _fetch_row_fingerprint('fp_x') == blob

    def test_get_chromaprint_returns_blob_or_none(self, db, use_test_db):
        from database import persist_chromaprint, get_chromaprint

        with db.cursor() as cur:
            _seed(cur, 'fp_g', 'pg', 'Alb', '/m/pg.flac')
        db.commit()

        assert get_chromaprint('srv', 'pg') is None
        assert get_chromaprint('srv', 'missing') is None
        blob = _blob(3)
        persist_chromaprint('srv', 'pg', blob)
        assert get_chromaprint('srv', 'pg') == blob
        persist_chromaprint('srv', 'pg', None)
        assert get_chromaprint('srv', 'pg') is None

    def test_null_sentinel_reads_as_abstain(self, db, use_test_db):
        from database import persist_chromaprint
        from tasks.analysis.helper import _fetch_row_fingerprint

        with db.cursor() as cur:
            _seed(cur, 'fp_y', 'p2', 'Alb', '/m/p2.flac')
        db.commit()

        persist_chromaprint('srv', 'p2', None)
        assert _fetch_row_fingerprint('fp_y') is None

    def test_upsert_overwrites_prior_fingerprint(self, db, use_test_db):
        from database import persist_chromaprint
        from tasks.analysis.helper import _fetch_row_fingerprint

        with db.cursor() as cur:
            _seed(cur, 'fp_z', 'p3', 'Alb', '/m/p3.flac')
        db.commit()

        persist_chromaprint('srv', 'p3', _blob(1))
        persist_chromaprint('srv', 'p3', _blob(9))
        assert _fetch_row_fingerprint('fp_z') == _blob(9)

    def test_a_sweep_match_carries_the_fingerprint_with_the_mapping(
        self, db, use_test_db
    ):
        from database import persist_chromaprint, get_chromaprint
        from tasks.mediaserver import registry

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_s', 'old', 'A-album', '/one/old.flac')
        db.commit()
        persist_chromaprint('srv', 'old', _blob(1))

        registry.upsert_track_maps(
            'srv2', {'new': ('fp_s', 'path', '/two/new.flac')}, conn=db
        )

        assert get_chromaprint('srv2', 'new') == _blob(1)

    def test_a_second_file_on_one_id_never_gets_handed_a_borrowed_fingerprint(
        self, db, use_test_db
    ):
        from database import persist_chromaprint, get_chromaprint
        from tasks.mediaserver import registry

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_p', 'old', 'A-album', '/one/old.flac')
        db.commit()
        persist_chromaprint('srv', 'old', _blob(1))

        registry.upsert_track_maps(
            'srv2',
            {
                'newA': ('fp_p', 'path', '/two/a.flac'),
                'newB': ('fp_p', 'path', '/two/b.flac'),
            },
            conn=db,
        )

        assert get_chromaprint('srv2', 'newA') is None
        assert get_chromaprint('srv2', 'newB') is None

    def test_a_failed_fpcalc_sentinel_still_inherits_a_sibling_print(
        self, db, use_test_db
    ):
        from database import persist_chromaprint, get_chromaprint
        from tasks.mediaserver import registry

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_n', 'old', 'A-album', '/one/old.flac')
        db.commit()
        persist_chromaprint('srv', 'old', _blob(1))
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO track_server_map "
                "(item_id, server_id, provider_track_id, match_tier, file_path) "
                "VALUES ('fp_n', 'srv2', 'new', 'path', '/two/new.flac')"
            )
        db.commit()
        persist_chromaprint('srv2', 'new', None)

        registry.upsert_track_maps(
            'srv2', {'new': ('fp_n', 'path', '/two/new.flac')}, conn=db
        )

        assert get_chromaprint('srv2', 'new') == _blob(1), (
            'the NULL row says fpcalc failed on this file, not that the track is '
            'finished with: matching on c.provider_track_id IS NULL treated the '
            'sentinel as a print and locked the track out of every future sweep'
        )

    def test_an_ambiguous_source_never_hands_on_a_guessed_fingerprint(
        self, db, use_test_db
    ):
        from database import persist_chromaprint, get_chromaprint
        from tasks.mediaserver import registry

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_a', 'oldA', 'A-album', '/one/a.flac')
            cur.execute(
                "INSERT INTO track_server_map "
                "(item_id, server_id, provider_track_id, match_tier, file_path) "
                "VALUES ('fp_a', 'srv', 'oldB', 'path', '/one/b.flac')"
            )
        db.commit()
        persist_chromaprint('srv', 'oldA', _blob(1))
        persist_chromaprint('srv', 'oldB', _blob(7))

        registry.upsert_track_maps(
            'srv2', {'new': ('fp_a', 'path', '/two/new.flac')}, conn=db
        )

        assert get_chromaprint('srv2', 'new') is None, (
            'two files on the source server share this canonical id, so neither '
            'print is known to be the right one; DISTINCT ON just took whichever '
            'sorted first and the gate then compared against a guess'
        )

    def test_a_broken_chromaprint_table_never_fails_the_mapping_write(
        self, db, use_test_db
    ):
        from tasks.mediaserver import registry

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_r', 'old', 'A-album', '/one/old.flac')
            cur.execute("DROP TABLE chromaprint")
        db.commit()

        written = registry.upsert_track_maps(
            'srv2', {'new': ('fp_r', 'path', '/two/new.flac')}, conn=db
        )

        assert written == 1
        with db.cursor() as cur:
            cur.execute(
                "SELECT item_id FROM track_server_map "
                "WHERE server_id = 'srv2' AND provider_track_id = 'new'"
            )
            assert cur.fetchone()[0] == 'fp_r'

