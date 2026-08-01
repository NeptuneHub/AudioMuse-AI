# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Drive the real startup catalogue migration against a real PostgreSQL.

The whole-catalogue key rewrite is the single most dangerous operation in the
codebase: it rewrites score.item_id, cascades into every embedding table, is the
only code allowed to DELETE FROM score, and commits at Flask boot before anyone
can look at it. Until now nothing exercised it end to end - the only test stubbed
_build_mapping to return nothing, so it proved the no-op path and nothing else.

Main Features:
* A legacy catalogue is relabelled for real: provider ids become content ids,
  the provider ids survive in track_server_map, and the legacy score.file_path
  moves onto the server's own map row.
* Duplicate candidates come from the audio IVF cluster directory (seeded here):
  identical audio in one cluster with matching duration merges into ONE catalogue
  row and keeps its path; a different or unknown duration splits; and a track the
  index does not cover - or a missing index entirely - merges nothing (fail-safe).
* The verification gate ROLLS BACK a rewrite that violates its invariants, leaving
  the catalogue byte-for-byte as it was.
* The embedding cascade is re-added even when no constraint existed to find.
"""

import os

import numpy as np
import pytest

try:
    import psycopg2
except Exception:  # pragma: no cover
    psycopg2 = None

from tasks import simhash

pytestmark = pytest.mark.integration


_SCHEMA = [
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, "
    "album TEXT, album_artist TEXT, tempo REAL, key TEXT, scale TEXT, "
    "mood_vector TEXT, energy REAL, other_features TEXT, year INTEGER, "
    "rating INTEGER, file_path TEXT, duration DOUBLE PRECISION)",
    "CREATE TABLE playlist (id SERIAL PRIMARY KEY, playlist_name TEXT, "
    "item_id TEXT, title TEXT, author TEXT, server_id TEXT)",
    "CREATE UNIQUE INDEX idx_playlist_name_item_server "
    "ON playlist (playlist_name, item_id, server_id)",
    "CREATE TABLE clap_embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA)",
    "CREATE TABLE lyrics_embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA, axis_vector BYTEA)",
    "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, "
    "server_type TEXT, creds JSONB DEFAULT '{}', music_libraries TEXT DEFAULT '', "
    "is_default BOOLEAN NOT NULL DEFAULT FALSE, track_count INTEGER, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE track_server_map ("
    "item_id TEXT NOT NULL REFERENCES score (item_id) ON UPDATE CASCADE ON DELETE CASCADE, "
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_track_id TEXT NOT NULL, match_tier TEXT, file_path TEXT, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "PRIMARY KEY (server_id, provider_track_id))",
    "CREATE TABLE chromaprint ("
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_track_id TEXT NOT NULL, fingerprint BYTEA, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "PRIMARY KEY (server_id, provider_track_id))",
    "CREATE TABLE map_projection_data (index_name VARCHAR(255) PRIMARY KEY, "
    "projection_data BYTEA NOT NULL, id_map_json TEXT NOT NULL, "
    "embedding_dimension INTEGER NOT NULL)",
    "CREATE TABLE ivf_dir (name TEXT PRIMARY KEY, blob_data BYTEA NOT NULL, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
]

_EMBEDDING_FK = (
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA)"
)
_EMBEDDING_NO_FK = "CREATE TABLE embedding (item_id TEXT PRIMARY KEY, embedding BYTEA)"


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


def _distinct_embedding(seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(simhash.SIGNATURE_BITS).astype(np.float32)


def _build(conn, tracks, embedding_ddl=_EMBEDDING_FK):
    """A legacy catalogue: provider-keyed item_ids, paths on the SHARED score row,
    no track_server_map rows at all. This is what an install looks like before the
    startup migration has ever run."""
    with conn.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS chromaprint, track_server_map, music_servers, embedding, "
            "clap_embedding, lyrics_embedding, playlist, map_projection_data, "
            "ivf_dir, score CASCADE"
        )
        for ddl in _SCHEMA:
            cur.execute(ddl)
        cur.execute(embedding_ddl)
        cur.execute(
            "INSERT INTO music_servers (server_id, name, server_type, is_default) "
            "VALUES ('srv', 'Jellyfin', 'jellyfin', TRUE)"
        )
        for item_id, path, vector in tracks:
            cur.execute(
                "INSERT INTO score (item_id, title, file_path) VALUES (%s, %s, %s)",
                (item_id, item_id, path),
            )
            cur.execute(
                "INSERT INTO embedding (item_id, embedding) VALUES (%s, %s)",
                (item_id, psycopg2.Binary(vector.astype(np.float32).tobytes())),
            )
    conn.commit()


def _seed_ivf(conn, entries):
    from tasks.paged_ivf import pack_directory
    import config as cfg

    item_ids = [item_id for item_id, _cell in entries]
    id2cell = np.array([cell for _item_id, cell in entries], dtype=np.uint32)
    n_cells = int(id2cell.max()) + 1 if id2cell.size else 1
    centroids = np.zeros((n_cells, simhash.SIGNATURE_BITS), dtype=np.float32)
    blob = pack_directory(centroids, id2cell, item_ids, simhash.SIGNATURE_BITS, 'angular')
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ivf_dir (name, blob_data) VALUES (%s, %s) "
            "ON CONFLICT (name) DO UPDATE SET blob_data = EXCLUDED.blob_data",
            ('%s__ivf_dir' % cfg.INDEX_NAME, psycopg2.Binary(blob)),
        )
    conn.commit()


def _seed_ivf_all(conn, tracks):
    _seed_ivf(conn, [(item_id, 0) for item_id, _path, _vec in tracks])


def _score(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT item_id, file_path FROM score ORDER BY item_id")
        return cur.fetchall()


def _maps(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT provider_track_id, item_id, file_path FROM track_server_map "
            "ORDER BY provider_track_id"
        )
        return cur.fetchall()


def _embedding_cascade_exists(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_constraint "
            "WHERE conrelid = 'embedding'::regclass AND contype = 'f'"
        )
        return cur.fetchone()[0] > 0


@pytest.fixture
def db(pg_dsn):
    conn = psycopg2.connect(pg_dsn)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def no_provider_listing(monkeypatch):
    from tasks import fingerprint_canonicalize as fc

    monkeypatch.setattr(fc, '_fetch_provider_durations', lambda source_id, conn: {})


class TestRealCanonicalization:
    def test_legacy_catalogue_is_relabelled_and_the_path_moves_to_the_map_row(self, db):
        from tasks import fingerprint_canonicalize as fc

        tracks = [
            ('jf-1', '/music/A/01.flac', _distinct_embedding(1)),
            ('jf-2', '/music/B/02.flac', _distinct_embedding(2)),
            ('jf-3', '/music/C/03.flac', _distinct_embedding(3)),
        ]
        _build(db, tracks)

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['relabelled'] == 3
        assert result['duplicates'] == 0

        rows = _score(db)
        assert len(rows) == 3
        assert all(item_id.startswith(simhash.CURRENT_ID_HEAD) for item_id, _ in rows), (
            "every legacy provider id must become a content id"
        )

        # The provider's real ids survive on the map row, and the legacy path has
        # moved off the shared score row onto the server that actually holds the file.
        assert [(p, path) for p, _item, path in _maps(db)] == [
            ('jf-1', '/music/A/01.flac'),
            ('jf-2', '/music/B/02.flac'),
            ('jf-3', '/music/C/03.flac'),
        ]

        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM embedding e WHERE NOT EXISTS "
                "(SELECT 1 FROM score s WHERE s.item_id = e.item_id)"
            )
            assert cur.fetchone()[0] == 0, "no embedding may lose its score parent"

    def test_backslash_in_id_and_path_survives_the_copy_streams(self, db):
        from tasks import fingerprint_canonicalize as fc

        # backslash is COPY's escape char, so a Windows path (or a provider id with
        # a backslash) must be escaped or the relabel-map and track_server_map COPY
        # streams corrupt the row. Exercises both _copy_pairs and
        # _copy_track_server_map.
        tracks = [
            ('jf\\odd', 'C:\\Music\\Album\\01 Song.flac', _distinct_embedding(1)),
            ('jf-2', '/music/B/02.flac', _distinct_embedding(2)),
        ]
        _build(db, tracks)

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['relabelled'] == 2
        rows = _score(db)
        assert all(item_id.startswith(simhash.CURRENT_ID_HEAD) for item_id, _ in rows), (
            "the backslash id must still relabel, not corrupt the relabel map"
        )
        by_provider = {p: path for p, _item, path in _maps(db)}
        assert 'jf\\odd' in by_provider, "the backslash provider id survives the COPY"
        assert by_provider['jf\\odd'] == 'C:\\Music\\Album\\01 Song.flac', (
            "the Windows path round-trips intact"
        )

    def test_carriage_return_tab_and_newline_round_trip_the_copy_streams(self, db):
        from tasks import fingerprint_canonicalize as fc

        tracks = [
            ('jf\rodd', '/music/Weird\rAlbum/01\tSong\nTake.flac', _distinct_embedding(1)),
            ('jf-2', '/music/B/02.flac', _distinct_embedding(2)),
        ]
        _build(db, tracks)

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['relabelled'] == 2
        rows = _score(db)
        assert all(item_id.startswith(simhash.CURRENT_ID_HEAD) for item_id, _ in rows), (
            "the carriage-return id must still relabel, not crash the COPY stream"
        )
        by_provider = {p: path for p, _item, path in _maps(db)}
        assert 'jf\rodd' in by_provider, (
            "a provider id holding a literal CR survives the COPY byte-for-byte"
        )
        assert by_provider['jf\rodd'] == '/music/Weird\rAlbum/01\tSong\nTake.flac', (
            "CR, tab and newline in the path round-trip intact, not mangled to spaces"
        )

    def test_identical_audio_with_matching_duration_merges_to_one_row(
        self, db, monkeypatch
    ):
        from tasks import fingerprint_canonicalize as fc

        # jf-2 sits within the length tolerance of jf-1, whatever the tolerance is.
        tol = fc.config.DURATION_TOLERANCE_SECONDS
        monkeypatch.setattr(
            fc,
            '_fetch_provider_durations',
            lambda source_id, conn: {'jf-1': 200.0, 'jf-2': 200.0 + tol, 'jf-3': 300.0},
        )
        same = _distinct_embedding(7)
        tracks = [
            ('jf-1', '/music/Album/01 Rio.flac', same),
            ('jf-2', '/music/Best Of/07 Rio.flac', same.copy()),
            ('jf-3', '/music/Other/03.flac', _distinct_embedding(9)),
        ]
        _build(db, tracks)
        _seed_ivf_all(db, tracks)

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['duplicates'] == 1
        rows = _score(db)
        assert len(rows) == 2, "the same audio is ONE song in the catalogue"

        # Both provider files still map, both to the SAME song, each keeping its own
        # path. The merge re-inserts the loser's map rows under the winner's id, and
        # it must carry file_path across: the map row is the only copy of the path.
        maps = _maps(db)
        assert [p for p, _i, _f in maps] == ['jf-1', 'jf-2', 'jf-3']
        by_provider = {p: (item, path) for p, item, path in maps}
        assert by_provider['jf-1'][0] == by_provider['jf-2'][0], (
            "two files of the same audio map to one AudioMuse id"
        )
        assert by_provider['jf-1'][1] == '/music/Album/01 Rio.flac'
        assert by_provider['jf-2'][1] == '/music/Best Of/07 Rio.flac'

        with db.cursor() as cur:
            cur.execute(
                "SELECT duration FROM score WHERE item_id = %s",
                (by_provider['jf-3'][0],),
            )
            assert cur.fetchone()[0] == pytest.approx(300.0), (
                "the migration must backfill score.duration from the server metadata"
            )

    def test_same_folder_files_never_share_an_id_during_canonicalize(self, db, monkeypatch):
        from tasks import fingerprint_canonicalize as fc

        # jf-1 and jf-3 are DIFFERENT songs that happen to sit in the SAME folder;
        # both are near-identical to jf-2 in another folder. Folding the folder rule
        # into the id calculation must keep jf-1 and jf-3 on separate ids in this
        # one pass - never form the merge and then unmap it (which would orphan a row).
        monkeypatch.setattr(
            fc, '_fetch_provider_durations',
            lambda source_id, conn: {'jf-1': 200.0, 'jf-2': 200.0, 'jf-3': 200.0},
        )
        same = _distinct_embedding(11)
        tracks = [
            ('jf-1', '/music/Album/01.flac', same),
            ('jf-2', '/music/Other/02.flac', same.copy()),
            ('jf-3', '/music/Album/03.flac', same.copy()),
        ]
        _build(db, tracks)
        _seed_ivf_all(db, tracks)

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['duplicates'] == 1, "only the cross-folder file may merge"
        assert len(_score(db)) == 2
        maps = {p: item for p, item, _path in _maps(db)}
        assert set(maps) == {'jf-1', 'jf-2', 'jf-3'}, "every file stays mapped (no orphan)"
        assert maps['jf-1'] != maps['jf-3'], "same-folder files must not share an id"
        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM score s WHERE NOT EXISTS "
                "(SELECT 1 FROM track_server_map t WHERE t.item_id = s.item_id)"
            )
            assert cur.fetchone()[0] == 0, "no catalogue row is left orphaned"

    def test_same_sounding_audio_with_different_length_stays_two_songs(
        self, db, monkeypatch
    ):
        from tasks import fingerprint_canonicalize as fc

        monkeypatch.setattr(
            fc,
            '_fetch_provider_durations',
            lambda source_id, conn: {'jf-1': 200.0, 'jf-2': 210.0},
        )
        same = _distinct_embedding(7)
        tracks = [
            ('jf-1', '/music/Brendel/nocturne.flac', same),
            ('jf-2', '/music/Arrau/nocturne.flac', same.copy()),
        ]
        _build(db, tracks)
        _seed_ivf_all(db, tracks)

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['duplicates'] == 0, (
            "same embedding but different length is a DIFFERENT recording"
        )
        rows = _score(db)
        assert len(rows) == 2
        maps = _maps(db)
        assert len({item for _p, item, _f in maps}) == 2, (
            "each file keeps its own catalogue id"
        )

    def test_unknown_durations_merge_nothing(self, db):
        from tasks import fingerprint_canonicalize as fc

        same = _distinct_embedding(7)
        tracks = [
            ('jf-1', '/music/Album/01 Rio.flac', same),
            ('jf-2', '/music/Best Of/07 Rio.flac', same.copy()),
        ]
        _build(db, tracks)
        _seed_ivf_all(db, tracks)

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['duplicates'] == 0, (
            "without durations nothing can be proven identical, so nothing merges"
        )
        assert len(_score(db)) == 2

    def test_without_an_ivf_index_every_track_stays_unique(self, db, monkeypatch):
        from tasks import fingerprint_canonicalize as fc

        monkeypatch.setattr(
            fc, '_fetch_provider_durations',
            lambda source_id, conn: {'jf-1': 200.0, 'jf-2': 200.0},
        )
        same = _distinct_embedding(7)
        tracks = [
            ('jf-1', '/music/Album/01 Rio.flac', same),
            ('jf-2', '/music/Best Of/07 Rio.flac', same.copy()),
        ]
        _build(db, tracks)

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['duplicates'] == 0, (
            "with no IVF index there are no clusters to compare, so nothing merges"
        )
        assert len(_score(db)) == 2
        assert {p for p, _i, _f in _maps(db)} == {'jf-1', 'jf-2'}, "both files stay mapped"
        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM score s WHERE NOT EXISTS "
                "(SELECT 1 FROM track_server_map t WHERE t.item_id = s.item_id)"
            )
            assert cur.fetchone()[0] == 0, "the fail-safe leaves no orphan"

    def test_a_track_missing_from_the_ivf_index_stays_unique_while_others_merge(
        self, db, monkeypatch
    ):
        from tasks import fingerprint_canonicalize as fc

        monkeypatch.setattr(
            fc, '_fetch_provider_durations',
            lambda source_id, conn: {'jf-1': 200.0, 'jf-2': 200.0, 'jf-3': 200.0},
        )
        same = _distinct_embedding(7)
        tracks = [
            ('jf-1', '/music/Album/01.flac', same),
            ('jf-2', '/music/Best Of/07.flac', same.copy()),
            ('jf-3', '/music/Live/03.flac', same.copy()),
        ]
        _build(db, tracks)
        _seed_ivf(db, [('jf-1', 0), ('jf-2', 0)])

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['duplicates'] == 1, "the two indexed copies merge"
        assert len(_score(db)) == 2, "jf-3 is not in the index, so it keeps its own id"
        by_provider = {p: item for p, item, _f in _maps(db)}
        assert by_provider['jf-1'] == by_provider['jf-2']
        assert by_provider['jf-3'] != by_provider['jf-1'], (
            "a track missing from the index is never merged"
        )

    def test_unsignable_embeddings_never_propose_or_receive_merges(self, db):
        from tasks import fingerprint_canonicalize as fc

        flat = np.full(simhash.SIGNATURE_BITS, 0.5, dtype=np.float32)
        near_flat = flat.copy()
        near_flat[0] += 1e-3
        same = _distinct_embedding(7)
        tracks = [
            ('jf-flat', '/music/A/tone.flac', flat),
            ('jf-nearflat', '/music/B/tone2.flac', near_flat),
            ('jf-r1', '/music/C/song.flac', same),
            ('jf-r2', '/music/D/song.flac', same.copy()),
        ]
        _build(db, tracks)
        _seed_ivf_all(db, tracks)
        with db.cursor() as cur:
            cur.execute("UPDATE score SET duration = 200.0")
        db.commit()

        ids = [item_id for item_id, _path, _vec in tracks]
        valid = np.array(
            [simhash.embedding_signature(vec) is not None for _i, _p, vec in tracks]
        )
        assert not valid[0], "precondition: a constant embedding has no signature"
        assert valid[1:].all()

        with db.cursor() as cur:
            left, right = fc._ivf_candidate_pairs(cur, ids, valid, len(ids), {}, 'srv')

        pairs = set(zip(left.tolist(), right.tolist()))
        assert pairs == {(2, 3)}, (
            "only the two real copies pair; the signatureless constant embedding "
            "proposes nothing and receives nothing, even though its cosine to the "
            "near-constant track is ~1.0"
        )

    def test_a_constant_embedding_row_migrates_to_fp0_and_cannot_brick_the_boot(
        self, db, monkeypatch
    ):
        from tasks import fingerprint_canonicalize as fc

        monkeypatch.setattr(
            fc, '_fetch_provider_durations',
            lambda source_id, conn: {'jf-flat': 90.0, 'jf-1': 200.0, 'jf-2': 200.0},
        )
        flat = np.full(simhash.SIGNATURE_BITS, 0.5, dtype=np.float32)
        same = _distinct_embedding(7)
        tracks = [
            ('jf-flat', '/music/T/tone.flac', flat),
            ('jf-1', '/music/A/song.flac', same),
            ('jf-2', '/music/B/song.flac', same.copy()),
        ]
        _build(db, tracks)
        _seed_ivf_all(db, tracks)

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['relabelled'] == 3
        assert result['duplicates'] == 1
        expected = simhash.unsignable_canonical_id('srv', 'jf-flat')
        assert expected.startswith('fp_0')
        maps = {p: item for p, item, _f in _maps(db)}
        assert maps['jf-flat'] == expected, (
            "the signatureless track gets the SAME server-scoped fp_0 id "
            "analysis would mint for it"
        )
        assert maps['jf-1'] == maps['jf-2'], "the real duplicate still merges"
        with db.cursor() as cur:
            cur.execute(
                "SELECT match_tier FROM track_server_map "
                "WHERE provider_track_id = 'jf-flat'"
            )
            assert cur.fetchone()[0] == 'analysis', (
                "the analysis tier is what exempts fp_0 rows from the legacy count"
            )
            cur.execute(
                "SELECT count(*) FROM embedding WHERE item_id = %s", (expected,)
            )
            assert cur.fetchone()[0] == 1, "its analysis rows follow the relabel"

        second = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')
        assert second == {'relabelled': 0, 'duplicates': 0}, (
            "the fp_0 row never re-triggers the migration"
        )

    def test_no_corruption_shape_of_the_embedding_can_brick_the_migration(self, db):
        from tasks import fingerprint_canonicalize as fc

        tracks = [
            ('jf-missing', '/music/A/01.flac', _distinct_embedding(1)),
            ('jf-null', '/music/B/02.flac', _distinct_embedding(2)),
            ('jf-truncated', '/music/C/03.flac', _distinct_embedding(3)),
            ('jf-wrong-size', '/music/D/04.flac', _distinct_embedding(4)),
            ('jf-ok', '/music/E/05.flac', _distinct_embedding(5)),
        ]
        _build(db, tracks)
        with db.cursor() as cur:
            cur.execute("DELETE FROM embedding WHERE item_id = 'jf-missing'")
            cur.execute("UPDATE embedding SET embedding = NULL WHERE item_id = 'jf-null'")
            cur.execute(
                "UPDATE embedding SET embedding = %s WHERE item_id = 'jf-truncated'",
                (psycopg2.Binary(b'\x00\x01\x02'),),
            )
            cur.execute(
                "UPDATE embedding SET embedding = %s WHERE item_id = 'jf-wrong-size'",
                (psycopg2.Binary(np.zeros(10, dtype=np.float32).tobytes()),),
            )
        db.commit()

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['relabelled'] == 5, "every corruption shape still relabels"
        maps = {p: item for p, item, _f in _maps(db)}
        for provider in ('jf-missing', 'jf-null', 'jf-truncated', 'jf-wrong-size'):
            assert maps[provider] == simhash.unsignable_canonical_id('srv', provider)
        assert maps['jf-ok'].startswith(simhash.CURRENT_ID_HEAD)
        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM track_server_map WHERE match_tier = 'analysis'"
            )
            assert cur.fetchone()[0] == 4
        second = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')
        assert second == {'relabelled': 0, 'duplicates': 0}, (
            "no corruption shape leaves a legacy id behind to re-run the migration"
        )

    def test_unsignable_fp0_uses_the_source_servers_provider_id_when_mapped(self, db):
        from tasks import fingerprint_canonicalize as fc

        flat = np.full(simhash.SIGNATURE_BITS, 0.5, dtype=np.float32)
        tracks = [
            ('old-key', '/music/T/tone.flac', flat),
            ('jf-1', '/music/A/song.flac', _distinct_embedding(1)),
        ]
        _build(db, tracks)
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO track_server_map "
                "(item_id, server_id, provider_track_id, match_tier) "
                "VALUES ('old-key', 'srv', 'nav-123', 'default')"
            )
        db.commit()

        fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        expected = simhash.unsignable_canonical_id('srv', 'nav-123')
        maps = {p: item for p, item, _f in _maps(db)}
        assert maps['nav-123'] == expected, (
            "the fp_0 id is minted from the server's REAL provider id, not the "
            "legacy score key, so a later re-analysis resolves to the same row"
        )
        with db.cursor() as cur:
            cur.execute(
                "SELECT match_tier FROM track_server_map "
                "WHERE provider_track_id = 'nav-123'"
            )
            assert cur.fetchone()[0] == 'analysis'
        second = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')
        assert second == {'relabelled': 0, 'duplicates': 0}

    def test_an_unbound_fp0_row_is_not_handed_back_as_legacy_work(self, db):
        """A provider migration unbinds every song the target does not have, which
        deletes the 'analysis' map row that used to be the ONLY evidence a row was
        minted unsignable. The row then looked like legacy work to this migration -
        which can never relabel it - so every boot re-hashed the whole catalogue and
        re-minted a mapping to a provider id the server no longer has. The fp_0 head
        is evidence nothing can unbind."""
        from tasks import fingerprint_canonicalize as fc

        flat = np.full(simhash.SIGNATURE_BITS, 0.5, dtype=np.float32)
        unsignable_id = simhash.unsignable_canonical_id('srv', 'nav-gone')
        _build(db, [(unsignable_id, '/music/T/tone.flac', flat)])
        db.commit()

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result == {'relabelled': 0, 'duplicates': 0}, (
            "an unbound fp_0 row is already canonical; it is not legacy work"
        )
        with db.cursor() as cur:
            cur.execute("SELECT item_id FROM score")
            assert [r[0] for r in cur.fetchall()] == [unsignable_id]
            cur.execute("SELECT count(*) FROM track_server_map")
            assert cur.fetchone()[0] == 0, (
                "no phantom mapping to a provider id the server no longer has"
            )

    def test_an_id_that_only_looks_like_fp0_is_still_legacy_work(self, db):
        from tasks import fingerprint_canonicalize as fc

        _build(db, [('fp_0short', '/music/A/song.flac', _distinct_embedding(1))])
        db.commit()

        result = fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert result['relabelled'] == 1, (
            "the exemption is the exact minted fp_0 shape, not any id starting fp_0"
        )

    def test_relabel_preserves_an_existing_provider_migration_tier(self, db):
        from tasks import fingerprint_canonicalize as fc

        tracks = [('jf-1', '/music/A/song.flac', _distinct_embedding(1))]
        _build(db, tracks)
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO track_server_map "
                "(item_id, server_id, provider_track_id, match_tier) "
                "VALUES ('jf-1', 'srv', 'jf-1', 'exact')"
            )
        db.commit()

        fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        with db.cursor() as cur:
            cur.execute(
                "SELECT match_tier FROM track_server_map "
                "WHERE provider_track_id = 'jf-1'"
            )
            assert cur.fetchone()[0] == 'exact', (
                "a signable row's relabel must not clobber the stored match tier"
            )

    def test_a_rewrite_that_fails_its_own_checks_is_rolled_back(self, db, monkeypatch):
        """The point of no return. A rewrite that would commit a corrupt catalogue
        must instead leave it byte-for-byte unchanged and fail the boot loudly."""
        from tasks import fingerprint_canonicalize as fc

        tracks = [
            ('jf-1', '/music/A/01.flac', _distinct_embedding(1)),
            ('jf-2', '/music/B/02.flac', _distinct_embedding(2)),
        ]
        _build(db, tracks)
        before = _score(db)

        # A song quietly vanishes during the rewrite. This is the shape of the damage
        # that used to commit and be trusted forever after: the rewrite "succeeds",
        # the catalogue is short a row, and its analysis is gone.
        original = fc._repoint_indexes

        def losing_a_song(cur, renames):
            original(cur, renames)
            cur.execute("DELETE FROM score WHERE item_id = (SELECT min(item_id) FROM score)")

        monkeypatch.setattr(fc, '_repoint_indexes', losing_a_song)

        with pytest.raises(fc.CanonicalizationVerificationError, match="expected"):
            fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        db.rollback()
        assert _score(db) == before, "the catalogue must be EXACTLY as it was"
        assert _maps(db) == [], "and no half-written mappings may survive"

    def test_the_embedding_cascade_is_recreated_even_when_none_existed(self, db):
        """find_fk finding nothing used to mean 'skip the re-add', leaving the table
        with no cascade at all and saying nothing about it."""
        from tasks import fingerprint_canonicalize as fc

        tracks = [('jf-1', '/music/A/01.flac', _distinct_embedding(1))]
        _build(db, tracks, embedding_ddl=_EMBEDDING_NO_FK)
        assert not _embedding_cascade_exists(db), "precondition: no cascade to find"

        fc.canonicalize_fingerprinted_ids(conn=db, source_server_id='srv')

        assert _embedding_cascade_exists(db), (
            "the rewrite must leave the cascade in place, not silently drop it"
        )
