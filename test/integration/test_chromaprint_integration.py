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

Main Features:
* persist_chromaprint / get_chromaprint round-trip and the NULL retry-stop
  sentinel reading back as abstain.
* _fetch_row_fingerprint JOIN from a canonical id to any mapped file's blob.
* Backfill target query picks whole missing albums and skips present and
  sentinel rows, bounded by the album limit.
* The target query skips a mapping the canonical track is already covered for,
  and spends its album limit only on albums that survive that skip.
* A sweep match carries the fingerprint with the mapping in the SAME
  transaction that writes it, so the backfill has nothing left to download; a
  mapping written before its source was fingerprinted catches up set-based.
* What is never inherited: a failed-once NULL sentinel, a file that already has
  its own fingerprint, and the same-server duplicate groups the false-merge
  splitter compares, which must stay individually measured.
* The inherit rides a SAVEPOINT, so a chromaprint table that is missing or
  mid-migration can never roll back the mapping write it travels with.
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
    from tasks.analysis import helper, main

    monkeypatch.setattr(database, 'get_db', lambda: db)
    monkeypatch.setattr(helper, 'get_db', lambda: db)
    monkeypatch.setattr(main, 'get_db', lambda: db)
    return db


def _add_server(cur, server_id):
    cur.execute(
        "INSERT INTO music_servers (server_id, name, server_type) "
        "VALUES (%s, %s, 'navidrome') ON CONFLICT (server_id) DO NOTHING",
        (server_id, server_id),
    )


def _seed(cur, item_id, provider_id, album, file_path, server_id='srv'):
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
        "match_tier, file_path) VALUES (%s, %s, %s, 'fingerprint', %s)",
        (item_id, server_id, provider_id, file_path),
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

    def test_a_swept_mapping_is_not_refingerprinted_when_another_server_has_one(
        self, db, use_test_db
    ):
        from database import persist_chromaprint
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_s', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_s', 'new', 'A-album', '/two/new.flac', server_id='srv2')
        db.commit()

        persist_chromaprint('srv', 'old', _blob(1))

        assert _chromaprint_backfill_targets('srv2', 5) == []

    def test_a_swept_mapping_is_fingerprinted_when_no_server_has_one(
        self, db, use_test_db
    ):
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_n', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_n', 'new', 'A-album', '/two/new.flac', server_id='srv2')
        db.commit()

        targets = _chromaprint_backfill_targets('srv2', 5)
        assert {provider_id for provider_id, _path in targets} == {'new'}

    def test_a_failed_fingerprint_elsewhere_does_not_count_as_covered(
        self, db, use_test_db
    ):
        from database import persist_chromaprint
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_f', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_f', 'new', 'A-album', '/two/new.flac', server_id='srv2')
        db.commit()

        persist_chromaprint('srv', 'old', None)

        targets = _chromaprint_backfill_targets('srv2', 5)
        assert {provider_id for provider_id, _path in targets} == {'new'}

    def test_same_server_duplicates_are_still_fingerprinted_for_the_splitter(
        self, db, use_test_db
    ):
        from database import persist_chromaprint
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_d', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_d', 'newA', 'A-album', '/two/a.flac', server_id='srv2')
            _seed(cur, 'fp_d', 'newB', 'A-album', '/two/b.flac', server_id='srv2')
        db.commit()

        persist_chromaprint('srv', 'old', _blob(1))

        targets = _chromaprint_backfill_targets('srv2', 5)
        assert {provider_id for provider_id, _path in targets} == {'newA', 'newB'}

    def test_the_album_limit_is_spent_on_albums_that_still_need_work(
        self, db, use_test_db
    ):
        from database import persist_chromaprint
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_a', 'old-a', 'A-album', '/one/a.flac')
            _seed(cur, 'fp_a', 'new-a', 'A-album', '/two/a.flac', server_id='srv2')
            _seed(cur, 'fp_b', 'old-b', 'B-album', '/one/b.flac')
            _seed(cur, 'fp_b', 'new-b', 'B-album', '/two/b.flac', server_id='srv2')
            _seed(cur, 'fp_solo', 'solo', 'C-album', '/two/solo.flac', server_id='srv2')
        db.commit()

        persist_chromaprint('srv', 'old-a', _blob(1))
        persist_chromaprint('srv', 'old-b', _blob(2))

        targets = _chromaprint_backfill_targets('srv2', 1)
        assert {provider_id for provider_id, _path in targets} == {'solo'}

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

    def test_a_carried_fingerprint_leaves_the_backfill_nothing_to_download(
        self, db, use_test_db
    ):
        from database import persist_chromaprint
        from tasks.analysis.main import _chromaprint_backfill_targets
        from tasks.mediaserver import registry

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_s', 'old', 'A-album', '/one/old.flac')
        db.commit()
        persist_chromaprint('srv', 'old', _blob(1))

        registry.upsert_track_maps(
            'srv2', {'new': ('fp_s', 'path', '/two/new.flac')}, conn=db
        )

        assert _chromaprint_backfill_targets('srv2', 5) == []

    def test_a_mapping_written_before_the_source_was_fingerprinted_catches_up(
        self, db, use_test_db
    ):
        from database import (
            persist_chromaprint,
            get_chromaprint,
            inherit_chromaprints_for_mapped_tracks,
        )
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_l', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_l', 'new', 'A-album', '/two/new.flac', server_id='srv2')
        db.commit()
        persist_chromaprint('srv', 'old', _blob(1))

        assert inherit_chromaprints_for_mapped_tracks(conn=db) == 1
        assert get_chromaprint('srv2', 'new') == _blob(1)
        assert _chromaprint_backfill_targets('srv2', 5) == []

    def test_nothing_is_inherited_when_no_server_has_a_fingerprint_yet(
        self, db, use_test_db
    ):
        from database import inherit_chromaprints_for_mapped_tracks
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_n', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_n', 'new', 'A-album', '/two/new.flac', server_id='srv2')
        db.commit()

        assert inherit_chromaprints_for_mapped_tracks(conn=db) == 0
        targets = _chromaprint_backfill_targets('srv2', 5)
        assert {provider_id for provider_id, _path in targets} == {'new'}

    def test_a_failed_fingerprint_is_never_inherited_as_if_it_were_one(
        self, db, use_test_db
    ):
        from database import persist_chromaprint, inherit_chromaprints_for_mapped_tracks
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_f', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_f', 'new', 'A-album', '/two/new.flac', server_id='srv2')
        db.commit()
        persist_chromaprint('srv', 'old', None)

        assert inherit_chromaprints_for_mapped_tracks(conn=db) == 0
        targets = _chromaprint_backfill_targets('srv2', 5)
        assert {provider_id for provider_id, _path in targets} == {'new'}

    def test_inheriting_never_overwrites_a_fingerprint_a_file_already_has(
        self, db, use_test_db
    ):
        from database import (
            persist_chromaprint,
            get_chromaprint,
            inherit_chromaprints_for_mapped_tracks,
        )

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_k', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_k', 'new', 'A-album', '/two/new.flac', server_id='srv2')
        db.commit()
        persist_chromaprint('srv', 'old', _blob(1))
        persist_chromaprint('srv2', 'new', _blob(9))

        assert inherit_chromaprints_for_mapped_tracks(conn=db) == 0
        assert get_chromaprint('srv2', 'new') == _blob(9)

    def test_same_server_duplicates_are_measured_not_inherited(self, db, use_test_db):
        from database import (
            persist_chromaprint,
            get_chromaprint,
            inherit_chromaprints_for_mapped_tracks,
        )
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_d', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_d', 'newA', 'A-album', '/two/a.flac', server_id='srv2')
            _seed(cur, 'fp_d', 'newB', 'A-album', '/two/b.flac', server_id='srv2')
        db.commit()
        persist_chromaprint('srv', 'old', _blob(1))

        assert inherit_chromaprints_for_mapped_tracks(conn=db) == 0
        assert get_chromaprint('srv2', 'newA') is None
        assert get_chromaprint('srv2', 'newB') is None
        targets = _chromaprint_backfill_targets('srv2', 5)
        assert {provider_id for provider_id, _path in targets} == {'newA', 'newB'}

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

    def test_a_single_server_library_still_backfills_everything(self, db, use_test_db):
        from database import inherit_chromaprints_for_mapped_tracks
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _seed(cur, 'fp_o1', 'q1', 'A-album', '/m/q1.flac')
            _seed(cur, 'fp_o2', 'q2', 'B-album', '/m/q2.flac')
        db.commit()

        assert inherit_chromaprints_for_mapped_tracks(conn=db) == 0
        targets = _chromaprint_backfill_targets('srv', 5)
        assert {provider_id for provider_id, _path in targets} == {'q1', 'q2'}
