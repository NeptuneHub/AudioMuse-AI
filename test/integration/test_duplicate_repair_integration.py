# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Drive the real startup duplicate check against a real PostgreSQL.

The check runs its own SQL at Flask boot - a session advisory lock, the
NULL-duration group query, an execute_values duration stamp and a scoped
unmap - none of which the unit tests exercise (they mock the cursor). This
proves the real statements against a real server, and proves the property that
matters most: it is a table-derived one-time step, so a second run is an instant
no-op with no server contact.

Main Features:
* A real false duplicate loses only its track_server_map rows; its score and
  embedding rows are never touched.
* A real duplicate is kept and its length is stamped onto score.duration, then its
  id is bumped to the current scheme, so the second run's version gate is an
  instant no-op that never calls the music server again.
* A survivor that already carries a duration (a legacy-migrated row) is never
  re-fetched - it is only relabelled to the current scheme.
"""

import os

import pytest

try:
    import psycopg2
except Exception:  # pragma: no cover
    psycopg2 = None

pytestmark = pytest.mark.integration


# The repair now closes with the fp_2 -> fp_3 scheme relabel, which drops/re-adds
# the embedding-table FKs, rewrites item_id across score/playlist/embedding tables
# and repoints the similarity index maps - so the schema has to carry those tables
# too, exactly as production does, or the relabel cannot run.
_SCHEMA = [
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, "
    "duration DOUBLE PRECISION)",
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA)",
    "CREATE TABLE clap_embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA)",
    "CREATE TABLE lyrics_embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA, axis_vector BYTEA)",
    "CREATE TABLE playlist (id SERIAL PRIMARY KEY, playlist_name TEXT, item_id TEXT, "
    "title TEXT, author TEXT, server_id TEXT)",
    "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, "
    "server_type TEXT, creds JSONB DEFAULT '{}', is_default BOOLEAN DEFAULT TRUE, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE track_server_map ("
    "item_id TEXT NOT NULL REFERENCES score (item_id) ON UPDATE CASCADE ON DELETE CASCADE, "
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_track_id TEXT NOT NULL, match_tier TEXT, file_path TEXT, "
    "PRIMARY KEY (server_id, provider_track_id))",
    "CREATE TABLE map_projection_data (index_name VARCHAR(255) PRIMARY KEY, "
    "projection_data BYTEA NOT NULL, id_map_json TEXT NOT NULL, "
    "embedding_dimension INTEGER NOT NULL)",
    "CREATE TABLE ivf_dir (name TEXT PRIMARY KEY, blob_data BYTEA NOT NULL)",
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


def _fp_id(suffix):
    from tasks.simhash import CANONICAL_ID_LEN

    body = (suffix.encode('utf-8').hex() or '0') * CANONICAL_ID_LEN
    return ('fp_2' + body)[:CANONICAL_ID_LEN]


def _current(item_id):
    # The id a seeded fp_2 row carries AFTER the repair's scheme relabel bumps it.
    from tasks import simhash

    return simhash.to_current_scheme_id(item_id)


@pytest.fixture
def db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    with conn.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS track_server_map, music_servers, embedding, "
            "clap_embedding, lyrics_embedding, playlist, map_projection_data, "
            "ivf_dir, score CASCADE"
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


def _seed_group(cur, item_id, provider_ids, duration=None):
    cur.execute(
        "INSERT INTO score (item_id, title, duration) VALUES (%s, %s, %s)",
        (item_id, item_id, duration),
    )
    cur.execute(
        "INSERT INTO embedding (item_id, embedding) VALUES (%s, %s)",
        (item_id, b'\x00\x00'),
    )
    for provider_id in provider_ids:
        cur.execute(
            "INSERT INTO track_server_map (item_id, server_id, provider_track_id, "
            "match_tier, file_path) VALUES (%s, 'srv', %s, 'default', %s)",
            (item_id, provider_id, "/music/%s/song.flac" % provider_id),
        )


def _maps(conn, item_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT provider_track_id FROM track_server_map WHERE item_id = %s "
            "ORDER BY provider_track_id",
            (item_id,),
        )
        return [row[0] for row in cur.fetchall()]


def _duration(conn, item_id):
    with conn.cursor() as cur:
        cur.execute("SELECT duration FROM score WHERE item_id = %s", (item_id,))
        row = cur.fetchone()
        return row[0] if row else 'gone'


def _run(db, monkeypatch, durations):
    from tasks import duplicate_repair as dr

    monkeypatch.setattr(dr, '_server_durations', lambda server: durations)
    monkeypatch.setattr(
        dr.registry, 'get_server',
        lambda server_id, conn=None: {
            'server_id': server_id, 'name': server_id,
            'server_type': 'navidrome', 'creds': {},
        },
    )
    return dr.repair_duplicate_track_maps(conn=db)


class TestRealDuplicateRepair:
    def test_real_kept_and_stamped_false_unmapped_then_idempotent(self, db, monkeypatch):
        real = _fp_id('a')
        false = _fp_id('b')
        with db.cursor() as cur:
            _seed_group(cur, real, ['pr1', 'pr2'])
            _seed_group(cur, false, ['pf1', 'pf2'])
        db.commit()

        durations = {'pr1': 200.0, 'pr2': 201.0, 'pf1': 120.0, 'pf2': 240.0}
        result = _run(db, monkeypatch, durations)
        db.commit()

        assert (result['real'], result['false'], result['removed']) == (1, 1, 2)
        # Real duplicate: length stamped, so its id is bumped to the current scheme;
        # both files still map to it under that new id.
        assert _maps(db, _current(real)) == ['pr1', 'pr2']
        assert _duration(db, _current(real)) == pytest.approx(200.0)
        # False duplicate: map rows gone and no length -> it's an ORPHAN. It is now
        # bumped to the current scheme (kept, never deleted) so the version gate can
        # go cold; a future server that has the track can re-map it under the new id.
        assert _maps(db, _current(false)) == []
        assert _duration(db, false) == 'gone'  # old-scheme id no longer exists
        assert _duration(db, _current(false)) is None  # bumped, still carries no length
        with db.cursor() as cur:
            cur.execute("SELECT count(*) FROM score")
            assert cur.fetchone()[0] == 2, "a false merge never deletes a score row"
            cur.execute("SELECT count(*) FROM embedding")
            assert cur.fetchone()[0] == 2

        # Second run short-circuits on the version gate and NEVER calls the server:
        # the real survivor carries a length and the false orphan is now current-scheme,
        # so no older-scheme row is left to keep the migration alive.
        def explode(server):
            raise AssertionError("the second run must not contact the music server")

        monkeypatch.setattr(
            __import__('tasks.duplicate_repair', fromlist=['x']),
            '_server_durations', explode,
        )
        second = _run(db, monkeypatch, durations)
        assert second == {'skipped': 'up_to_date'}

    def test_single_file_rows_are_backfilled_and_then_idempotent(self, db, monkeypatch):
        # The 88% case: rows mapping ONE file get their length stamped (never
        # unmapped), and the whole check no-ops on the next run.
        a = _fp_id('d')
        b = _fp_id('e')
        with db.cursor() as cur:
            _seed_group(cur, a, ['pa'])
            _seed_group(cur, b, ['pb'])
        db.commit()

        result = _run(db, monkeypatch, {'pa': 200.0, 'pb': 314.0})
        db.commit()

        assert result['backfilled'] == 2
        assert result['removed'] == 0
        # Both got a length, so both were bumped to the current scheme.
        assert _duration(db, _current(a)) == pytest.approx(200.0)
        assert _duration(db, _current(b)) == pytest.approx(314.0)
        assert _maps(db, _current(a)) == ['pa'] and _maps(db, _current(b)) == ['pb']

        def explode(server):
            raise AssertionError("backfilled single-file rows must not be re-listed")

        from tasks import duplicate_repair as dr
        monkeypatch.setattr(dr, '_server_durations', explode)
        # No older-scheme row is left, so the version gate short-circuits instantly.
        second = _run(db, monkeypatch, {})
        assert second == {'skipped': 'up_to_date'}

    def test_single_file_without_server_length_gets_sentinel_and_is_one_time(
        self, db, monkeypatch
    ):
        from tasks import duplicate_repair as dr

        # A RELIABLE listing (most lengths present, so the server is not skipped)
        # that just misses one file: that file gets the 0 sentinel so it is
        # one-time; the others get their real length.
        k1, k2, missing = _fp_id('f'), _fp_id('g'), _fp_id('h')
        with db.cursor() as cur:
            _seed_group(cur, k1, ['pk1'])
            _seed_group(cur, k2, ['pk2'])
            _seed_group(cur, missing, ['pmiss'])
        db.commit()

        result = _run(db, monkeypatch, {'pk1': 200.0, 'pk2': 300.0})  # 2/3 known
        db.commit()

        assert result['backfilled'] == 2 and result['no_length'] == 1
        # The sentinel (0.0) is a non-NULL length, so the row is also relabelled.
        assert _duration(db, _current(missing)) == pytest.approx(dr._NO_SERVER_DURATION)
        assert _maps(db, _current(missing)) == ['pmiss'], "a single file is never unmapped"

        def explode(server):
            raise AssertionError("a sentinel row must not be re-listed")

        monkeypatch.setattr(dr, '_server_durations', explode)
        # Every row now carries a length and is on the current scheme: instant skip.
        assert _run(db, monkeypatch, {}) == {'skipped': 'up_to_date'}

    def test_single_file_survivor_with_duration_is_relabelled_not_re_fetched(self, db, monkeypatch):
        # A single-file row that already has a duration cannot be a wrong merge, so
        # the check must not look at it (no second duration fetch after a legacy
        # upgrade), but the scheme relabel still bumps it up to the current id.
        already = _fp_id('c')
        with db.cursor() as cur:
            _seed_group(cur, already, ['p1'], duration=200.0)
        db.commit()

        def explode(server):
            raise AssertionError("a single-file duration-bearing row must not be re-fetched")

        from tasks import duplicate_repair as dr
        monkeypatch.setattr(dr, '_server_durations', explode)
        result = dr.repair_duplicate_track_maps(conn=db)

        assert result['checked'] == 0
        assert result['relabelled'] == 1
        assert _maps(db, _current(already)) == ['p1']
        assert _duration(db, _current(already)) == pytest.approx(200.0)

    def test_existing_stamped_merge_is_re_split_when_lengths_now_disagree(self, db, monkeypatch):
        # A scheme bump (e.g. fp_3 -> fp_4 tightening 7s to 1s) re-verifies EXISTING
        # merges: a stamped group whose files actually differ in length by more than
        # the current tolerance is unmapped so each re-analyzes under its own id.
        merged = _fp_id('r')
        with db.cursor() as cur:
            _seed_group(cur, merged, ['p1', 'p2'], duration=200.0)
        db.commit()

        result = _run(db, monkeypatch, {'p1': 200.0, 'p2': 260.0})
        db.commit()

        assert result['false'] == 1
        assert result['removed'] == 2
        assert _maps(db, _current(merged)) == []

    def test_existing_stamped_merge_within_tolerance_survives_re_verify(self, db, monkeypatch):
        # The same re-verify keeps a genuine merge whose files agree within tolerance,
        # without dropping its stored length.
        merged = _fp_id('s')
        with db.cursor() as cur:
            _seed_group(cur, merged, ['p1', 'p2'], duration=200.0)
        db.commit()

        result = _run(db, monkeypatch, {'p1': 200.0, 'p2': 200.5})
        db.commit()

        assert result['false'] == 0
        assert _maps(db, _current(merged)) == ['p1', 'p2']
        assert _duration(db, _current(merged)) == pytest.approx(200.0)

    def test_prefetched_durations_avoid_a_second_server_listing(self, db, monkeypatch):
        # The legacy migration already listed this server earlier in the same boot
        # and handed its durations to the repair; the repair must reuse them and
        # never list the server a second time (the whole point of the fix).
        a = _fp_id('m')
        with db.cursor() as cur:
            _seed_group(cur, a, ['pa'])
        db.commit()

        from tasks import duplicate_repair as dr

        def explode(server):
            raise AssertionError("a prefetched server must not be listed again")

        monkeypatch.setattr(dr, '_server_durations', explode)
        result = dr.repair_duplicate_track_maps(
            conn=db, prefetched_durations={'srv': {'pa': 200.0}},
        )
        db.commit()

        assert result['backfilled'] == 1
        assert _duration(db, _current(a)) == pytest.approx(200.0)


def _seed_paths(cur, item_id, mappings):
    cur.execute(
        "INSERT INTO score (item_id, title, duration) VALUES (%s, %s, 200.0)",
        (item_id, item_id),
    )
    cur.execute(
        "INSERT INTO embedding (item_id, embedding) VALUES (%s, %s)",
        (item_id, b'\x00\x00'),
    )
    for provider_id, file_path in mappings:
        cur.execute(
            "INSERT INTO track_server_map (item_id, server_id, provider_track_id, "
            "match_tier, file_path) VALUES (%s, 'srv', %s, 'default', %s)",
            (item_id, provider_id, file_path),
        )


class TestSameFolderCleanup:
    def test_splits_same_folder_keeps_cross_folder_and_same_file(self, db):
        from tasks import duplicate_repair as dr

        same_folder = _fp_id('sf')
        cross_folder = _fp_id('cf')
        same_file = _fp_id('sx')
        with db.cursor() as cur:
            _seed_paths(cur, same_folder, [
                ('p1', '/media/music/Artist/Album/01 - One.flac'),
                ('p2', '/media/music/Artist/Album/02 - Two.flac'),
            ])
            _seed_paths(cur, cross_folder, [
                ('p3', '/media/music/Artist/Album A/01 - Song.flac'),
                ('p4', '/media/music/Artist/Album B/05 - Song.flac'),
            ])
            _seed_paths(cur, same_file, [
                ('p5', '/media/music/Artist/Album/07 - Same.flac'),
                ('p6', '/media/music/Artist/Album/07 - Same.flac'),
            ])
        db.commit()

        result = dr.split_same_folder_merges(conn=db)
        db.commit()

        assert result == {'split': 1, 'removed': 2}
        assert _maps(db, same_folder) == [], "same-folder files are unmapped"
        assert _maps(db, cross_folder) == ['p3', 'p4'], "cross-folder dup survives"
        assert _maps(db, same_file) == ['p5', 'p6'], "same physical file survives"
        # A split never deletes the catalogue row itself.
        assert _duration(db, same_folder) == pytest.approx(200.0)

    def test_second_run_is_an_instant_noop(self, db):
        from tasks import duplicate_repair as dr

        same_folder = _fp_id('sf2')
        with db.cursor() as cur:
            _seed_paths(cur, same_folder, [
                ('q1', '/media/music/A/Alb/01 - One.flac'),
                ('q2', '/media/music/A/Alb/02 - Two.flac'),
            ])
        db.commit()

        assert dr.split_same_folder_merges(conn=db)['split'] == 1
        db.commit()
        assert dr.split_same_folder_merges(conn=db) == {'split': 0, 'removed': 0}
