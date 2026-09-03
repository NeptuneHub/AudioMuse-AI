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
* Adding a server can never start a backfill: every tier a sweep writes is
  skipped outright, even when NO server has a fingerprint yet and even when the
  swept rows form a local duplicate group. What the server analysed itself, and
  legacy rows with no tier, are still backfilled.
* A swept track still ENDS UP with a Chromaprint: driving the real
  _run_chromaprint_backfill downloads nothing for it, and the moment any server
  measures that canonical track the swept mapping is handed the result, in
  either server order.
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
* It is chunked, so it completes past a batch boundary and a failure part way
  through keeps every batch already committed and reports that count honestly -
  an unbounded copy of real kilobyte fingerprints hit the statement timeout.
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


class _DropsOnExecute:
    def __init__(self, real, fail_on):
        self._real = real
        self._fail_on = fail_on
        self._executes = 0

    def cursor(self):
        outer = self

        class _Cursor:
            def __init__(self):
                self._cur = outer._real.cursor()

            def execute(self, *args, **kwargs):
                outer._executes += 1
                if outer._executes == outer._fail_on:
                    raise RuntimeError("connection lost")
                return self._cur.execute(*args, **kwargs)

            def fetchall(self):
                return self._cur.fetchall()

            @property
            def rowcount(self):
                return self._cur.rowcount

            def close(self):
                return self._cur.close()

        return _Cursor()

    def commit(self):
        return self._real.commit()

    def rollback(self):
        return self._real.rollback()


def _run_backfill_over(server_ids, measures=None):
    import contextlib
    from unittest import mock
    import tasks.analysis.main as analysis
    from tasks.mediaserver import context as server_context
    from database import persist_chromaprint

    downloaded = []

    def _fake_download(server_id, provider_id, _path):
        downloaded.append((server_id, provider_id))
        if measures is not None:
            persist_chromaprint(server_id, provider_id, measures)
        return True

    with mock.patch.object(analysis.chromaprint, 'is_available', lambda: True), \
            mock.patch.object(analysis, 'CHROMAPRINT_COLLECTION_ENABLED', True), \
            mock.patch.object(analysis, '_bind_server_context', lambda s: s), \
            mock.patch.object(
                server_context, 'use_server',
                lambda *a, **k: contextlib.nullcontext()), \
            mock.patch.object(analysis, '_backfill_one_track', _fake_download):
        analysis._run_chromaprint_backfill(server_ids)
    return downloaded


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

    def test_a_covered_track_is_measured_when_inherit_falls_back(
        self, db, use_test_db
    ):
        from database import persist_chromaprint
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_fb', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_fb', 'new', 'A-album', '/two/new.flac', server_id='srv2')
        db.commit()
        persist_chromaprint('srv', 'old', _blob(1))

        assert _chromaprint_backfill_targets('srv2', 5) == []
        targets = _chromaprint_backfill_targets('srv2', 5, exclude_covered=False)
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

    def test_adding_a_server_never_starts_a_backfill_even_with_no_fingerprints(
        self, db, use_test_db
    ):
        from tasks.analysis.main import _chromaprint_backfill_targets
        from tasks.provider_migration_matcher import SWEEP_MATCH_TIERS

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            for index, tier in enumerate(SWEEP_MATCH_TIERS):
                _seed(cur, 'fp_w%d' % index, 'old%d' % index,
                      'album-%d' % index, '/one/%d.flac' % index)
                _seed(cur, 'fp_w%d' % index, 'new%d' % index,
                      'album-%d' % index, '/two/%d.flac' % index,
                      server_id='srv2', match_tier=tier)
        db.commit()

        assert _chromaprint_backfill_targets('srv2', 100) == []

    def test_a_swept_mapping_is_never_downloaded_even_when_it_is_a_local_duplicate(
        self, db, use_test_db
    ):
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_wd', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_wd', 'newA', 'A-album', '/two/a.flac',
                  server_id='srv2', match_tier='path')
            _seed(cur, 'fp_wd', 'newB', 'A-album', '/two/b.flac',
                  server_id='srv2', match_tier='norm_meta')
        db.commit()

        assert _chromaprint_backfill_targets('srv2', 100) == []

    def test_what_this_server_analysed_itself_is_still_backfilled(
        self, db, use_test_db
    ):
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_own', 'mine', 'A-album', '/two/mine.flac',
                  server_id='srv2', match_tier='fingerprint')
            _seed(cur, 'fp_leg', 'legacy', 'B-album', '/two/legacy.flac',
                  server_id='srv2', match_tier=None)
        db.commit()

        targets = _chromaprint_backfill_targets('srv2', 100)
        assert {provider_id for provider_id, _path in targets} == {'mine', 'legacy'}

    def test_a_swept_track_ends_up_with_a_chromaprint_once_any_server_measures_it(
        self, db, use_test_db
    ):
        from database import persist_chromaprint, get_chromaprint
        from tasks.analysis.main import _chromaprint_backfill_targets

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_c', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_c', 'new', 'A-album', '/two/new.flac',
                  server_id='srv2', match_tier='path')
        db.commit()

        assert get_chromaprint('srv2', 'new') is None
        assert _chromaprint_backfill_targets('srv2', 100) == []

        targets = _chromaprint_backfill_targets('srv', 100)
        assert {provider_id for provider_id, _path in targets} == {'old'}
        persist_chromaprint('srv', 'old', _blob(4))

        assert _run_backfill_over(['srv', 'srv2']) == []
        assert get_chromaprint('srv2', 'new') == _blob(4)

    def test_the_swept_track_is_covered_whatever_order_the_servers_run_in(
        self, db, use_test_db
    ):
        from database import persist_chromaprint, get_chromaprint

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_o', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_o', 'new', 'A-album', '/two/new.flac',
                  server_id='srv2', match_tier='path')
        db.commit()
        persist_chromaprint('srv', 'old', _blob(5))

        assert _run_backfill_over(['srv2', 'srv']) == []
        assert get_chromaprint('srv2', 'new') == _blob(5)

    def test_a_measurement_made_during_the_run_reaches_the_swept_track(
        self, db, use_test_db
    ):
        from database import get_chromaprint

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            _seed(cur, 'fp_run', 'old', 'A-album', '/one/old.flac')
            _seed(cur, 'fp_run', 'new', 'A-album', '/two/new.flac',
                  server_id='srv2', match_tier='path')
        db.commit()

        downloaded = _run_backfill_over(['srv2', 'srv'], measures=_blob(7))

        assert downloaded == [('srv', 'old')]
        assert get_chromaprint('srv2', 'new') == _blob(7)

    def test_the_hand_over_completes_past_a_batch_boundary(self, db, use_test_db):
        import config
        from database import (
            persist_chromaprint,
            get_chromaprint,
            inherit_chromaprints_for_mapped_tracks,
        )

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            for index in range(7):
                item = 'fp_b%d' % index
                _seed(cur, item, 'old%d' % index, 'A-album', '/one/%d.flac' % index)
                _seed(cur, item, 'new%d' % index, 'A-album', '/two/%d.flac' % index,
                      server_id='srv2', match_tier='path')
        db.commit()
        for index in range(7):
            persist_chromaprint('srv', 'old%d' % index, _blob(index + 1))

        original = config.CHROMAPRINT_INHERIT_BATCH_SIZE
        config.CHROMAPRINT_INHERIT_BATCH_SIZE = 2
        try:
            assert inherit_chromaprints_for_mapped_tracks(conn=db) == 7
        finally:
            config.CHROMAPRINT_INHERIT_BATCH_SIZE = original

        for index in range(7):
            assert get_chromaprint('srv2', 'new%d' % index) == _blob(index + 1)

    def test_a_failure_part_way_through_keeps_what_was_already_committed(
        self, db, use_test_db
    ):
        import config
        import database

        with db.cursor() as cur:
            _add_server(cur, 'srv2')
            for index in range(6):
                item = 'fp_p%d' % index
                _seed(cur, item, 'old%d' % index, 'A-album', '/one/%d.flac' % index)
                _seed(cur, item, 'new%d' % index, 'A-album', '/two/%d.flac' % index,
                      server_id='srv2', match_tier='path')
        db.commit()
        for index in range(6):
            database.persist_chromaprint('srv', 'old%d' % index, _blob(index + 1))

        original = config.CHROMAPRINT_INHERIT_BATCH_SIZE
        config.CHROMAPRINT_INHERIT_BATCH_SIZE = 2
        try:
            reported = database.inherit_chromaprints_for_mapped_tracks(
                conn=_DropsOnExecute(db, 3)
            )
        finally:
            config.CHROMAPRINT_INHERIT_BATCH_SIZE = original

        stored = sum(
            1 for index in range(6)
            if database.get_chromaprint('srv2', 'new%d' % index) is not None
        )
        assert stored == 4
        assert reported == stored

        assert database.inherit_chromaprints_for_mapped_tracks(conn=db) == 2
        assert all(
            database.get_chromaprint('srv2', 'new%d' % index) is not None
            for index in range(6)
        )

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
