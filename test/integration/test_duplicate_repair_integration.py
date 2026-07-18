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
* A real duplicate is kept and its length is stamped onto score.duration, so the
  second run selects nothing and never calls the music server again.
* A survivor that already carries a duration (a legacy-migrated row) is never
  examined - no double duration fetch on a legacy upgrade.
"""

import os

import pytest

try:
    import psycopg2
except Exception:  # pragma: no cover
    psycopg2 = None

pytestmark = pytest.mark.integration


_SCHEMA = [
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, "
    "duration DOUBLE PRECISION)",
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA)",
    "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, "
    "server_type TEXT, creds JSONB DEFAULT '{}', is_default BOOLEAN DEFAULT TRUE, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE track_server_map ("
    "item_id TEXT NOT NULL REFERENCES score (item_id) ON UPDATE CASCADE ON DELETE CASCADE, "
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_track_id TEXT NOT NULL, match_tier TEXT, "
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


def _fp_id(suffix):
    from tasks.simhash import CANONICAL_ID_LEN

    body = suffix * CANONICAL_ID_LEN
    return ('fp_2' + body)[:CANONICAL_ID_LEN]


@pytest.fixture
def db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    with conn.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS track_server_map, music_servers, embedding, "
            "score CASCADE"
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
            "match_tier) VALUES (%s, 'srv', %s, 'default')",
            (item_id, provider_id),
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

        assert result == {'checked': 2, 'real': 1, 'false': 1, 'removed': 2}
        # Real duplicate: both files still mapped, length stamped.
        assert _maps(db, real) == ['pr1', 'pr2']
        assert _duration(db, real) == pytest.approx(200.0)
        # False duplicate: map rows gone, but the score/embedding row survives.
        assert _maps(db, false) == []
        assert _duration(db, false) is None
        with db.cursor() as cur:
            cur.execute("SELECT count(*) FROM score")
            assert cur.fetchone()[0] == 2, "a false merge never deletes a score row"
            cur.execute("SELECT count(*) FROM embedding")
            assert cur.fetchone()[0] == 2

        # Second run is a no-op and NEVER calls the server: the real survivor now
        # carries a duration and the false group has dissolved.
        def explode(server):
            raise AssertionError("the second run must not contact the music server")

        monkeypatch.setattr(
            __import__('tasks.duplicate_repair', fromlist=['x']),
            '_server_durations', explode,
        )
        second = _run(db, monkeypatch, durations)
        assert second == {'checked': 0, 'real': 0, 'false': 0, 'removed': 0}

    def test_survivor_with_duration_is_never_examined(self, db, monkeypatch):
        # A legacy-migrated survivor already has a duration; the check must skip it
        # entirely (no second duration fetch after a legacy upgrade).
        already = _fp_id('c')
        with db.cursor() as cur:
            _seed_group(cur, already, ['p1', 'p2'], duration=200.0)
        db.commit()

        def explode(server):
            raise AssertionError("a duration-bearing survivor must not be checked")

        from tasks import duplicate_repair as dr
        monkeypatch.setattr(dr, '_server_durations', explode)
        result = dr.repair_duplicate_track_maps(conn=db)

        assert result == {'checked': 0, 'real': 0, 'false': 0, 'removed': 0}
        assert _maps(db, already) == ['p1', 'p2']
