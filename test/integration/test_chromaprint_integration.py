# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Drive the Chromaprint DB path against a real PostgreSQL.

Proves the SQL the unit tests mock: persist_chromaprint upserts the compressed
blob and the NULL retry-stop sentinel, the _fetch_row_fingerprint JOIN maps a
canonical id back to any file's fingerprint via track_server_map, and the
backfill target query picks whole albums that still lack a fingerprint while
skipping both already-fingerprinted tracks and the failed-once sentinel rows.
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
    "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, server_type TEXT)",
    "CREATE TABLE track_server_map ("
    "item_id TEXT NOT NULL REFERENCES score (item_id) ON DELETE CASCADE, "
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_track_id TEXT NOT NULL, match_tier TEXT, file_path TEXT, "
    "PRIMARY KEY (server_id, provider_track_id))",
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
    from tasks.analysis import helper, main

    monkeypatch.setattr(database, 'get_db', lambda: db)
    monkeypatch.setattr(helper, 'get_db', lambda: db)
    monkeypatch.setattr(main, 'get_db', lambda: db)
    return db


def _seed(cur, item_id, provider_id, album, file_path):
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
        "match_tier, file_path) VALUES (%s, 'srv', %s, 'fingerprint', %s)",
        (item_id, provider_id, file_path),
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

    def test_backfill_targets_skip_present_and_sentinel_rows(self, db, use_test_db):
        from database import persist_chromaprint
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _seed(cur, 'fp_a', 'pa', 'A-album', '/m/pa.flac')
            _seed(cur, 'fp_b', 'pb', 'B-album', '/m/pb.flac')
            _seed(cur, 'fp_c', 'pc', 'C-album', '/m/pc.flac')
        db.commit()

        persist_chromaprint('srv', 'pa', _blob(1))
        persist_chromaprint('srv', 'pb', None)

        targets = _chromaprint_backfill_targets('srv', 5)
        picked = {provider_id for provider_id, _path in targets}
        assert picked == {'pc'}

    def test_backfill_album_limit_bounds_the_work(self, db, use_test_db):
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _seed(cur, 'fp_1', 'p1', 'A-album', '/m/p1.flac')
            _seed(cur, 'fp_2', 'p2', 'B-album', '/m/p2.flac')
            _seed(cur, 'fp_3', 'p3', 'C-album', '/m/p3.flac')
        db.commit()

        targets = _chromaprint_backfill_targets('srv', 2)
        picked = {provider_id for provider_id, _path in targets}
        assert picked == {'p1', 'p2'}
