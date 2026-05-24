"""Unit tests for the /api/sync endpoint (app_sync blueprint).

Mocks `get_db` and `load_map_projection` at the blueprint-module level so we
never touch a real database. Mirrors the loader pattern in
``test_provider_migration_blueprint.py``.
"""
import base64
import importlib.util
import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _load_bp_module():
    mod_name = 'app_sync'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    mod_path = os.path.join(repo_root, 'app_sync.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bp_mod():
    return _load_bp_module()


@pytest.fixture
def app(bp_mod):
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(bp_mod.sync_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def mediaserver_type_jellyfin():
    """Default MEDIASERVER_TYPE to 'jellyfin' for every test (and restore after).
    Tests that need a different value override it inline."""
    import config
    saved = getattr(config, 'MEDIASERVER_TYPE', 'jellyfin')
    config.MEDIASERVER_TYPE = 'jellyfin'
    yield
    config.MEDIASERVER_TYPE = saved


# ---------------------------------------------------------------------------
# Fake DB plumbing
# ---------------------------------------------------------------------------

class FakeCursor:
    """Mock psycopg2 DictCursor that yields queued results in execute order."""

    def __init__(self):
        self.queries = []           # all SQL strings executed, in order
        self.query_params = []      # corresponding param tuples
        self._fetchone_queue = []   # fetchone() pops from here
        self._fetchall_queue = []   # fetchall() pops from here
        self._raise_on_execute = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def execute(self, sql, params=None):
        self.queries.append(sql)
        self.query_params.append(params)
        if self._raise_on_execute:
            raise self._raise_on_execute

    def fetchone(self):
        if not self._fetchone_queue:
            return None
        return self._fetchone_queue.pop(0)

    def fetchall(self):
        if not self._fetchall_queue:
            return []
        return self._fetchall_queue.pop(0)


@pytest.fixture
def fake_db(bp_mod):
    """Patch app_sync.get_db to return a mock connection with a FakeCursor."""
    cur = FakeCursor()
    conn = MagicMock()
    conn.cursor.return_value = cur

    bp_mod.get_db = MagicMock(return_value=conn)
    # By default no UMAP cache loaded
    bp_mod.load_map_projection = MagicMock(return_value=(None, None))
    return conn, cur


def make_dict_row(mapping: dict):
    """psycopg2-DictRow-like: supports both dict-key and attribute access."""
    class FakeRow(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
    return FakeRow(mapping)


def _minimal_track_row(**overrides):
    """A row with all the columns the endpoint SELECTs from `score`."""
    base = {
        'item_id': 'track-1',
        'title': 'Echoes',
        'author': 'Pink Floyd',
        'album': 'Meddle',
        'album_artist': 'Pink Floyd',
        'year': 1971,
        'tempo': 117.5,
        'key': 'C# Minor',
        'scale': 'minor',
        'mood_vector': 'progressive rock:0.92,psychedelic rock:0.85',
        'other_features': 'relaxed:0.8',
        'energy': 0.07,
        'rating': 5,
        'updated_at': datetime(2026, 3, 15, 14, 30, 0),
        # Embedding columns mirror the SELECT when include_embeddings=true; LEFT
        # JOIN returns NULL when the row is missing from embedding/clap_embedding.
        # Tests that want non-empty blobs override these.
        'musicnn_blob': None,
        'clap_blob': None,
    }
    base.update(overrides)
    return make_dict_row(base)


def _setup_happy_path(cur, tracks=None, total=None, deleted=None):
    """Queue cursor fetches in the order the endpoint calls them:
       1. SELECT COUNT(*) → fetchone -> total
       2. SELECT ... FROM score ... → fetchall -> tracks
       3. (optional, only if since given) SELECT FROM deleted_tracks → fetchall
    """
    tracks = tracks if tracks is not None else []
    total = total if total is not None else len(tracks)
    cur._fetchone_queue.append(make_dict_row({'n': total}))
    cur._fetchall_queue.append(tracks)
    if deleted is not None:
        cur._fetchall_queue.append([make_dict_row({'item_id': d}) for d in deleted])


# ---------------------------------------------------------------------------
# Entry-gate tests (no DB needed)
# ---------------------------------------------------------------------------


class TestMpdGate:
    def test_mpd_returns_501(self, bp_mod, client):
        import config
        config.MEDIASERVER_TYPE = 'mpd'

        resp = client.get('/api/sync?limit=1')

        assert resp.status_code == 501
        body = resp.get_json()
        assert 'mpd' in body['error'].lower()


class TestSinceParsing:
    def test_bad_since_returns_400(self, bp_mod, client):
        resp = client.get('/api/sync?since=garbage')

        assert resp.status_code == 400
        body = resp.get_json()
        assert 'since' in body['error'].lower()

    def test_since_with_z_suffix(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, tracks=[], total=0, deleted=[])

        resp = client.get('/api/sync?since=2026-03-15T14:30:00Z')

        assert resp.status_code == 200
        # Naive UTC normalized (no tzinfo) was passed to SQL
        since_param = cur.query_params[0][0]
        assert isinstance(since_param, datetime)
        assert since_param.tzinfo is None
        assert since_param == datetime(2026, 3, 15, 14, 30, 0)

    def test_since_with_offset_normalized_to_utc(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, tracks=[], total=0, deleted=[])

        # 14:30 +05:30 == 09:00 UTC
        resp = client.get('/api/sync?since=2026-03-15T14:30:00%2B05:30')

        assert resp.status_code == 200
        since_param = cur.query_params[0][0]
        assert since_param.tzinfo is None
        assert since_param == datetime(2026, 3, 15, 9, 0, 0)


# ---------------------------------------------------------------------------
# Envelope shape (happy path)
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_envelope_keys_present(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, tracks=[], total=0)

        resp = client.get('/api/sync?limit=1')

        assert resp.status_code == 200
        body = resp.get_json()
        for key in ('tracks', 'deleted_ids', 'total_tracks',
                    'provider_type', 'has_more', 'next_page'):
            assert key in body, f"missing {key}"

    def test_no_model_version_key(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, tracks=[], total=0)

        resp = client.get('/api/sync?limit=1')

        body = resp.get_json()
        assert 'model_version' not in body

    def test_provider_type_from_config(self, bp_mod, client, fake_db):
        import config
        config.MEDIASERVER_TYPE = 'navidrome'
        _, cur = fake_db
        _setup_happy_path(cur, tracks=[], total=0)

        resp = client.get('/api/sync?limit=1')

        assert resp.get_json()['provider_type'] == 'navidrome'

    def test_total_tracks_from_count_query(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, tracks=[], total=15000)

        resp = client.get('/api/sync?limit=1')

        assert resp.get_json()['total_tracks'] == 15000


# ---------------------------------------------------------------------------
# Pagination math
# ---------------------------------------------------------------------------


class TestPaginationMath:
    def test_has_more_when_more_pages_exist(self, bp_mod, client, fake_db):
        _, cur = fake_db
        # 750 total, page=1 limit=500 → 500 rows on this page, has_more=true
        _setup_happy_path(
            cur, total=750,
            tracks=[_minimal_track_row(item_id=f't{i}') for i in range(500)],
        )

        body = client.get('/api/sync?page=1&limit=500').get_json()

        assert body['has_more'] is True
        assert body['next_page'] == 2

    def test_no_has_more_on_last_page(self, bp_mod, client, fake_db):
        _, cur = fake_db
        # 750 total, page=2 limit=500 → 250 rows on this page, has_more=false
        _setup_happy_path(
            cur, total=750,
            tracks=[_minimal_track_row(item_id=f't{i}') for i in range(250)],
        )

        body = client.get('/api/sync?page=2&limit=500').get_json()

        assert body['has_more'] is False
        assert body['next_page'] is None

    def test_offset_passed_to_sql(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=0, tracks=[])

        client.get('/api/sync?page=3&limit=100')

        # 2nd query is the page query; its tail params are (limit, offset)
        page_params = cur.query_params[1]
        assert page_params[-2] == 100  # limit
        assert page_params[-1] == 200  # offset = (3-1)*100


# ---------------------------------------------------------------------------
# Limit / page clamping
# ---------------------------------------------------------------------------


class TestClamping:
    def test_limit_cap_at_1000(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=0, tracks=[])

        client.get('/api/sync?limit=99999')

        assert cur.query_params[1][-2] == 1000  # limit clamped

    def test_limit_floor_at_1(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=0, tracks=[])

        client.get('/api/sync?limit=0')

        assert cur.query_params[1][-2] == 1

    def test_page_floor_at_1(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=0, tracks=[])

        client.get('/api/sync?page=0&limit=1')

        assert cur.query_params[1][-1] == 0  # offset = (1-1)*1 = 0


# ---------------------------------------------------------------------------
# Per-track field shape
# ---------------------------------------------------------------------------


class TestTrackShape:
    def test_artist_renamed_from_author(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(author='Pink Floyd'),
        ])

        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]

        assert track['artist'] == 'Pink Floyd'
        assert 'author' not in track

    def test_id_field_from_item_id(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(item_id='abc-123'),
        ])

        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]

        assert track['id'] == 'abc-123'

    def test_energy_is_raw_value(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(energy=0.07),
        ])

        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]

        assert track['energy'] == 0.07  # not normalized

    def test_updated_at_isoformat(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(updated_at=datetime(2026, 3, 15, 14, 30, 0)),
        ])

        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]

        assert track['updated_at'] == '2026-03-15T14:30:00'

    def test_updated_at_null_passes_through(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(updated_at=None),
        ])

        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]

        assert track['updated_at'] is None


# ---------------------------------------------------------------------------
# include_embeddings + CLAP toggle
# ---------------------------------------------------------------------------


class TestIncludeEmbeddings:
    def test_default_includes_embedding(self, bp_mod, client, fake_db):
        import config
        config.CLAP_ENABLED = True
        blob = np.arange(200, dtype=np.float32).tobytes()
        clap_blob = np.arange(512, dtype=np.float32).tobytes()
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(musicnn_blob=blob, clap_blob=clap_blob),
        ])

        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]

        assert 'embedding' in track
        assert 'clap_embedding' in track

    def test_explicit_false_omits_keys(self, bp_mod, client, fake_db):
        import config
        config.CLAP_ENABLED = True
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(),
        ])

        track = client.get('/api/sync?limit=1&include_embeddings=false').get_json()['tracks'][0]

        assert 'embedding' not in track
        assert 'clap_embedding' not in track

    def test_case_insensitive_false(self, bp_mod, client, fake_db):
        import config
        config.CLAP_ENABLED = True
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(),
        ])

        track = client.get('/api/sync?limit=1&include_embeddings=FALSE').get_json()['tracks'][0]

        assert 'embedding' not in track

    def test_non_false_value_includes_embeddings(self, bp_mod, client, fake_db):
        """Per spec: ONLY literal 'false' (case-insensitive) disables. '0' does not."""
        import config
        config.CLAP_ENABLED = True
        blob = np.arange(200, dtype=np.float32).tobytes()
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(musicnn_blob=blob, clap_blob=None),
        ])

        track = client.get('/api/sync?limit=1&include_embeddings=0').get_json()['tracks'][0]

        assert 'embedding' in track

    def test_clap_disabled_omits_clap_key(self, bp_mod, client, fake_db):
        import config
        config.CLAP_ENABLED = False
        blob = np.arange(200, dtype=np.float32).tobytes()
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(musicnn_blob=blob),
        ])

        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]

        assert 'embedding' in track
        assert 'clap_embedding' not in track

    def test_empty_blob_yields_null(self, bp_mod, client, fake_db):
        import config
        config.CLAP_ENABLED = True
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(musicnn_blob=None, clap_blob=None),
        ])

        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]

        # Keys present (because include_embeddings defaulted to true) but null
        assert track['embedding'] is None
        assert track['clap_embedding'] is None

    def test_base64_roundtrip(self, bp_mod, client, fake_db):
        import config
        config.CLAP_ENABLED = True
        original = np.arange(200, dtype=np.float32)
        clap_orig = np.linspace(-1, 1, 512, dtype=np.float32)
        _, cur = fake_db
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(
                musicnn_blob=original.tobytes(),
                clap_blob=clap_orig.tobytes(),
            ),
        ])

        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]

        decoded = np.frombuffer(base64.b64decode(track['embedding']), dtype=np.float32)
        assert decoded.shape == (200,)
        np.testing.assert_array_equal(decoded, original)

        decoded_clap = np.frombuffer(base64.b64decode(track['clap_embedding']), dtype=np.float32)
        assert decoded_clap.shape == (512,)
        np.testing.assert_allclose(decoded_clap, clap_orig)


# ---------------------------------------------------------------------------
# deleted_ids gating
# ---------------------------------------------------------------------------


class TestDeletedIds:
    def test_no_since_returns_empty(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=0, tracks=[])  # no deleted queue

        body = client.get('/api/sync').get_json()

        assert body['deleted_ids'] == []
        # And no DELETED_TRACKS query was issued
        assert not any('deleted_tracks' in sql.lower() for sql in cur.queries)

    def test_with_since_populates_list(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_happy_path(cur, total=0, tracks=[], deleted=['gone-1', 'gone-2'])

        body = client.get('/api/sync?since=2026-01-01T00:00:00').get_json()

        assert body['deleted_ids'] == ['gone-1', 'gone-2']


# ---------------------------------------------------------------------------
# UMAP behavior
# ---------------------------------------------------------------------------


class TestUmap:
    def test_lookup_populates_coords(self, bp_mod, client, fake_db):
        _, cur = fake_db
        id_map = ['track-A', 'track-B']
        proj = np.array([[1.5, -2.5], [3.0, 4.0]], dtype=np.float32)
        bp_mod.load_map_projection = MagicMock(return_value=(id_map, proj))
        _setup_happy_path(cur, total=2, tracks=[
            _minimal_track_row(item_id='track-A'),
            _minimal_track_row(item_id='track-B'),
        ])

        tracks = client.get('/api/sync?limit=10').get_json()['tracks']

        assert tracks[0]['umap_x'] == pytest.approx(1.5)
        assert tracks[0]['umap_y'] == pytest.approx(-2.5)
        assert tracks[1]['umap_x'] == pytest.approx(3.0)

    def test_missing_track_yields_null(self, bp_mod, client, fake_db):
        _, cur = fake_db
        id_map = ['track-A']
        proj = np.array([[1.5, -2.5]], dtype=np.float32)
        bp_mod.load_map_projection = MagicMock(return_value=(id_map, proj))
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(item_id='track-NOT-IN-MAP'),
        ])

        track = client.get('/api/sync?limit=10').get_json()['tracks'][0]

        assert track['umap_x'] is None
        assert track['umap_y'] is None

    def test_projection_unavailable(self, bp_mod, client, fake_db):
        _, cur = fake_db
        bp_mod.load_map_projection = MagicMock(return_value=(None, None))
        _setup_happy_path(cur, total=1, tracks=[
            _minimal_track_row(),
        ])

        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]

        assert track['umap_x'] is None
        assert track['umap_y'] is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_db_error_returns_500(self, bp_mod, client, fake_db):
        _, cur = fake_db
        cur._raise_on_execute = RuntimeError("simulated DB failure")

        resp = client.get('/api/sync?limit=1')

        assert resp.status_code == 500
        body = resp.get_json()
        assert 'error' in body
