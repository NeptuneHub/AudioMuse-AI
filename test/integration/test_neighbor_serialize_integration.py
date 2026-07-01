# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Neighbor-serialization tests against a real Postgres database.

Seeds the score table and runs app_helper.serialize_neighbor_results over
real rows to confirm field mapping, missing-album handling, and injection
safety through the actual SELECT path.

Main Features:
* Missing-album sentinel vs None handling and title/author from rows.
* Unknown ids skipped, album-artist flag, and injection-style id safety.
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


_SCORE_DDL = (
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, "
    "album TEXT, album_artist TEXT, tempo REAL, key TEXT, scale TEXT, "
    "mood_vector TEXT, energy REAL, other_features TEXT, year INTEGER, "
    "rating INTEGER, file_path TEXT)"
)

_INJECTION_ID = "x'; DROP TABLE score; --"


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
def serialize_db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS score CASCADE")
        cur.execute(_SCORE_DDL)
        rows = [
            ('null-album', 'No Album Track', 'Artist A', None, 'AA', 'rock:0.9'),
            ('empty-album', 'Empty Album Track', 'Artist B', '', 'BB', 'pop:0.8'),
            ('normal-1', 'Real Title One', 'Real Author One', 'Real Album One', 'RA1', 'jazz:0.7'),
            ('normal-2', 'Real Title Two', 'Real Author Two', 'Real Album Two', 'RA2', 'metal:0.6'),
        ]
        for item_id, title, author, album, album_artist, mood in rows:
            cur.execute(
                "INSERT INTO score (item_id, title, author, album, album_artist, mood_vector) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (item_id, title, author, album, album_artist, mood),
            )
    yield conn
    conn.close()


def _bind_real_db(monkeypatch, conn):
    import database

    monkeypatch.setattr(database, 'get_db', lambda: conn)


def _by_id(out):
    return {row['item_id']: row for row in out}


def _table_exists(conn, name):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f'public.{name}',))
        return cur.fetchone()[0] is not None


def _score_count(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM score")
        return cur.fetchone()[0]


@pytest.mark.integration
class TestSerializeNeighborResultsRealDb:
    def test_missing_album_sentinel_substitutes_null_and_empty(self, serialize_db, monkeypatch):
        import app_helper

        _bind_real_db(monkeypatch, serialize_db)
        neighbors = [
            {'item_id': 'null-album', 'distance': 0.10},
            {'item_id': 'empty-album', 'distance': 0.20},
            {'item_id': 'normal-1', 'distance': 0.30},
        ]
        out = _by_id(app_helper.serialize_neighbor_results(neighbors, missing_album='unknown'))
        assert out['null-album']['album'] == 'unknown'
        assert out['empty-album']['album'] == 'unknown'
        assert out['normal-1']['album'] == 'Real Album One'

    def test_missing_album_none_preserves_raw_album(self, serialize_db, monkeypatch):
        import app_helper

        _bind_real_db(monkeypatch, serialize_db)
        neighbors = [
            {'item_id': 'null-album', 'distance': 0.10},
            {'item_id': 'empty-album', 'distance': 0.20},
            {'item_id': 'normal-1', 'distance': 0.30},
        ]
        out = _by_id(app_helper.serialize_neighbor_results(neighbors, missing_album=None))
        assert out['null-album']['album'] is None
        assert out['empty-album']['album'] == ''
        assert out['normal-1']['album'] == 'Real Album One'

    def test_title_author_come_from_real_rows(self, serialize_db, monkeypatch):
        import app_helper

        _bind_real_db(monkeypatch, serialize_db)
        neighbors = [
            {'item_id': 'normal-1', 'distance': 0.30},
            {'item_id': 'normal-2', 'distance': 0.40},
        ]
        out = _by_id(app_helper.serialize_neighbor_results(neighbors))
        assert out['normal-1']['title'] == 'Real Title One'
        assert out['normal-1']['author'] == 'Real Author One'
        assert out['normal-1']['album_artist'] == 'RA1'
        assert out['normal-1']['distance'] == pytest.approx(0.30)
        assert out['normal-2']['title'] == 'Real Title Two'
        assert out['normal-2']['author'] == 'Real Author Two'

    def test_unknown_id_absent_from_db_is_skipped(self, serialize_db, monkeypatch):
        import app_helper

        _bind_real_db(monkeypatch, serialize_db)
        neighbors = [
            {'item_id': 'normal-1', 'distance': 0.30},
            {'item_id': 'ghost-not-in-db', 'distance': 0.99},
        ]
        out = app_helper.serialize_neighbor_results(neighbors)
        ids = {row['item_id'] for row in out}
        assert 'normal-1' in ids
        assert 'ghost-not-in-db' not in ids
        assert len(out) == 1

    def test_include_album_artist_flag_omits_field(self, serialize_db, monkeypatch):
        import app_helper

        _bind_real_db(monkeypatch, serialize_db)
        neighbors = [{'item_id': 'normal-1', 'distance': 0.30}]
        out = app_helper.serialize_neighbor_results(neighbors, include_album_artist=False)
        assert out
        assert 'album_artist' not in out[0]

    def test_injection_style_id_through_real_select_is_safe(self, serialize_db, monkeypatch):
        import app_helper

        _bind_real_db(monkeypatch, serialize_db)
        before = _score_count(serialize_db)
        neighbors = [
            {'item_id': _INJECTION_ID, 'distance': 0.50},
            {'item_id': 'normal-1', 'distance': 0.30},
        ]
        out = app_helper.serialize_neighbor_results(neighbors)
        ids = {row['item_id'] for row in out}
        assert _INJECTION_ID not in ids
        assert 'normal-1' in ids
        assert _table_exists(serialize_db, 'score')
        assert _score_count(serialize_db) == before
