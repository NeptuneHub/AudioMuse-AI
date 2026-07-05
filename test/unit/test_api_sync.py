# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the app_sync client-sync blueprint endpoints.

Drives the sync payload, manifest, and UMAP endpoints against a fake DB to
check the response envelope, pagination, track shape, and embedding output.

Main Features:
* Envelope keys, pagination math, and limit/page clamping.
* Track field renaming, fingerprint SQL, and ids-filter behavior.
* Include-embeddings toggle, base64 roundtrip, UMAP coords, and DB-error 500.
"""

import base64
import importlib.util
import os
import sys
from unittest.mock import MagicMock

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
    import config

    saved = getattr(config, 'MEDIASERVER_TYPE', 'jellyfin')
    config.MEDIASERVER_TYPE = 'jellyfin'
    yield
    config.MEDIASERVER_TYPE = saved


class FakeCursor:
    def __init__(self):
        self.queries = []
        self.query_params = []
        self._fetchone_queue = []
        self._fetchall_queue = []
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
    cur = FakeCursor()
    conn = MagicMock()
    conn.cursor.return_value = cur
    bp_mod.get_db = MagicMock(return_value=conn)
    bp_mod.load_map_projection = MagicMock(return_value=(None, None))
    return conn, cur


def make_dict_row(mapping):
    class FakeRow(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name) from None

    return FakeRow(mapping)


def _minimal_track_row(**overrides):
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
        'fp': 'deadbeefcafe0001',
        'musicnn_blob': None,
        'clap_blob': None,
    }
    base.update(overrides)
    return make_dict_row(base)


def _manifest_row(item_id='track-1', fp='deadbeefcafe0001'):
    return make_dict_row({'item_id': item_id, 'fp': fp})


def _setup_payload(cur, tracks=None, total=None):
    tracks = tracks if tracks is not None else []
    total = total if total is not None else len(tracks)
    cur._fetchone_queue.append(make_dict_row({'n': total}))
    cur._fetchall_queue.append(tracks)


def _setup_manifest(cur, rows=None, total=None):
    rows = rows if rows is not None else []
    total = total if total is not None else len(rows)
    cur._fetchone_queue.append(make_dict_row({'n': total}))
    cur._fetchall_queue.append(rows)


def _setup_ids(cur, tracks=None):
    cur._fetchall_queue.append(tracks if tracks is not None else [])


class TestEnvelope:
    def test_envelope_keys_present(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, tracks=[], total=0)
        body = client.get('/api/sync?limit=1').get_json()
        for key in ('tracks', 'total_tracks', 'provider_type', 'has_more', 'next_page'):
            assert key in body, f"missing {key}"

    def test_no_deleted_ids_key(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, tracks=[], total=0)
        assert 'deleted_ids' not in client.get('/api/sync?limit=1').get_json()

    def test_provider_type_from_config(self, bp_mod, client, fake_db):
        import config

        config.MEDIASERVER_TYPE = 'navidrome'
        _, cur = fake_db
        _setup_payload(cur, tracks=[], total=0)
        assert client.get('/api/sync?limit=1').get_json()['provider_type'] == 'navidrome'

    def test_total_tracks_from_count_query(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, tracks=[], total=15000)
        assert client.get('/api/sync?limit=1').get_json()['total_tracks'] == 15000


class TestPaginationMath:
    def test_has_more_when_more_pages_exist(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(
            cur, total=750, tracks=[_minimal_track_row(item_id=f't{i}') for i in range(500)]
        )
        body = client.get('/api/sync?page=1&limit=500').get_json()
        assert body['has_more'] is True
        assert body['next_page'] == 2

    def test_no_has_more_on_last_page(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(
            cur, total=750, tracks=[_minimal_track_row(item_id=f't{i}') for i in range(250)]
        )
        body = client.get('/api/sync?page=2&limit=500').get_json()
        assert body['has_more'] is False
        assert body['next_page'] is None

    def test_offset_passed_to_sql(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=0, tracks=[])
        client.get('/api/sync?page=3&limit=100')
        page_params = cur.query_params[1]
        assert page_params[-2] == 100
        assert page_params[-1] == 200


class TestClamping:
    def test_payload_limit_cap_at_500(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=0, tracks=[])
        client.get('/api/sync?limit=99999')
        assert cur.query_params[1][-2] == 500

    def test_manifest_limit_cap_at_1000(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_manifest(cur, rows=[], total=0)
        client.get('/api/sync?fields=index&limit=99999')
        assert cur.query_params[1][-2] == 1000

    def test_limit_floor_at_1(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=0, tracks=[])
        client.get('/api/sync?limit=0')
        assert cur.query_params[1][-2] == 1

    def test_page_floor_at_1(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=0, tracks=[])
        client.get('/api/sync?page=0&limit=1')
        assert cur.query_params[1][-1] == 0


class TestTrackShape:
    def test_artist_renamed_from_author(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=1, tracks=[_minimal_track_row(author='Pink Floyd')])
        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]
        assert track['artist'] == 'Pink Floyd'
        assert 'author' not in track

    def test_id_field_from_item_id(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=1, tracks=[_minimal_track_row(item_id='abc-123')])
        assert client.get('/api/sync?limit=1').get_json()['tracks'][0]['id'] == 'abc-123'

    def test_energy_is_raw_value(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=1, tracks=[_minimal_track_row(energy=0.07)])
        assert client.get('/api/sync?limit=1').get_json()['tracks'][0]['energy'] == 0.07

    def test_no_updated_at_key(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=1, tracks=[_minimal_track_row()])
        assert 'updated_at' not in client.get('/api/sync?limit=1').get_json()['tracks'][0]


class TestFingerprint:
    def test_fp_present_on_payload_rows(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=1, tracks=[_minimal_track_row(fp='abc123def4560000')])
        assert client.get('/api/sync?limit=1').get_json()['tracks'][0]['fp'] == 'abc123def4560000'

    def test_fp_computed_in_sql(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=0, tracks=[])
        client.get('/api/sync?limit=1')
        select_sql = cur.queries[1]
        assert 'md5(' in select_sql
        assert 'AS fp' in select_sql

    def test_distinct_fp_passes_through(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(
            cur,
            total=2,
            tracks=[
                _minimal_track_row(item_id='a', fp='1111111111111111'),
                _minimal_track_row(item_id='b', fp='2222222222222222'),
            ],
        )
        tracks = client.get('/api/sync?limit=10').get_json()['tracks']
        assert tracks[0]['fp'] != tracks[1]['fp']


class TestManifest:
    def test_rows_are_id_fp_only(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_manifest(cur, total=1, rows=[_manifest_row('track-X', 'ffffffff00000000')])
        track = client.get('/api/sync?fields=index&limit=1').get_json()['tracks'][0]
        assert track == {'id': 'track-X', 'fp': 'ffffffff00000000'}

    def test_envelope_no_deleted_ids(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_manifest(cur, total=0, rows=[])
        body = client.get('/api/sync?fields=index').get_json()
        assert 'deleted_ids' not in body
        for key in ('tracks', 'total_tracks', 'provider_type', 'has_more', 'next_page'):
            assert key in body

    def test_paginates_at_1000(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_manifest(cur, total=2000, rows=[_manifest_row(f't{i}') for i in range(1000)])
        body = client.get('/api/sync?fields=index&page=1&limit=1000').get_json()
        assert body['has_more'] is True
        assert body['next_page'] == 2


class TestIdsFilter:
    def test_filters_to_set(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_ids(cur, tracks=[_minimal_track_row(item_id='a'), _minimal_track_row(item_id='b')])
        body = client.get('/api/sync?ids=a,b').get_json()
        assert [t['id'] for t in body['tracks']] == ['a', 'b']
        assert body['has_more'] is False
        assert body['next_page'] is None
        assert 'item_id IN (' in cur.queries[0]
        assert cur.query_params[0] == ('a', 'b')

    def test_empty_ids_runs_no_query(self, bp_mod, client, fake_db):
        _, cur = fake_db
        body = client.get('/api/sync?ids=').get_json()
        assert body['tracks'] == []
        assert body['total_tracks'] == 0
        assert cur.queries == []


class TestIncludeEmbeddings:
    def test_default_includes_both(self, bp_mod, client, fake_db):
        import config

        config.CLAP_ENABLED = True
        _, cur = fake_db
        _setup_payload(
            cur,
            total=1,
            tracks=[
                _minimal_track_row(
                    musicnn_blob=np.arange(200, dtype=np.float32).tobytes(),
                    clap_blob=np.arange(512, dtype=np.float32).tobytes(),
                )
            ],
        )
        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]
        assert 'embedding' in track and 'clap_embedding' in track

    def test_explicit_false_omits_keys(self, bp_mod, client, fake_db):
        import config

        config.CLAP_ENABLED = True
        _, cur = fake_db
        _setup_payload(cur, total=1, tracks=[_minimal_track_row()])
        track = client.get('/api/sync?limit=1&include_embeddings=false').get_json()['tracks'][0]
        assert 'embedding' not in track and 'clap_embedding' not in track

    def test_case_insensitive_false(self, bp_mod, client, fake_db):
        import config

        config.CLAP_ENABLED = True
        _, cur = fake_db
        _setup_payload(cur, total=1, tracks=[_minimal_track_row()])
        track = client.get('/api/sync?limit=1&include_embeddings=FALSE').get_json()['tracks'][0]
        assert 'embedding' not in track

    def test_non_false_value_includes_embeddings(self, bp_mod, client, fake_db):
        import config

        config.CLAP_ENABLED = True
        _, cur = fake_db
        _setup_payload(
            cur,
            total=1,
            tracks=[
                _minimal_track_row(
                    musicnn_blob=np.arange(200, dtype=np.float32).tobytes(), clap_blob=None
                )
            ],
        )
        track = client.get('/api/sync?limit=1&include_embeddings=0').get_json()['tracks'][0]
        assert 'embedding' in track

    def test_clap_disabled_omits_clap_key(self, bp_mod, client, fake_db):
        import config

        config.CLAP_ENABLED = False
        _, cur = fake_db
        _setup_payload(
            cur,
            total=1,
            tracks=[_minimal_track_row(musicnn_blob=np.arange(200, dtype=np.float32).tobytes())],
        )
        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]
        assert 'embedding' in track and 'clap_embedding' not in track

    def test_empty_blob_yields_null(self, bp_mod, client, fake_db):
        import config

        config.CLAP_ENABLED = True
        _, cur = fake_db
        _setup_payload(cur, total=1, tracks=[_minimal_track_row(musicnn_blob=None, clap_blob=None)])
        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]
        assert track['embedding'] is None and track['clap_embedding'] is None

    def test_base64_roundtrip(self, bp_mod, client, fake_db):
        import config

        config.CLAP_ENABLED = True
        original = np.arange(200, dtype=np.float32)
        clap_orig = np.linspace(-1, 1, 512, dtype=np.float32)
        _, cur = fake_db
        _setup_payload(
            cur,
            total=1,
            tracks=[
                _minimal_track_row(musicnn_blob=original.tobytes(), clap_blob=clap_orig.tobytes())
            ],
        )
        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]
        np.testing.assert_array_equal(
            np.frombuffer(base64.b64decode(track['embedding']), dtype=np.float32), original
        )
        np.testing.assert_allclose(
            np.frombuffer(base64.b64decode(track['clap_embedding']), dtype=np.float32), clap_orig
        )


class TestUmap:
    def test_lookup_populates_coords(self, bp_mod, client, fake_db):
        _, cur = fake_db
        bp_mod.load_map_projection = MagicMock(
            return_value=(
                ['track-A', 'track-B'],
                np.array([[1.5, -2.5], [3.0, 4.0]], dtype=np.float32),
            )
        )
        _setup_payload(
            cur,
            total=2,
            tracks=[_minimal_track_row(item_id='track-A'), _minimal_track_row(item_id='track-B')],
        )
        tracks = client.get('/api/sync?limit=10').get_json()['tracks']
        assert tracks[0]['umap_x'] == pytest.approx(1.5)
        assert tracks[0]['umap_y'] == pytest.approx(-2.5)
        assert tracks[1]['umap_x'] == pytest.approx(3.0)

    def test_missing_track_yields_null(self, bp_mod, client, fake_db):
        _, cur = fake_db
        bp_mod.load_map_projection = MagicMock(
            return_value=(['track-A'], np.array([[1.5, -2.5]], dtype=np.float32))
        )
        _setup_payload(cur, total=1, tracks=[_minimal_track_row(item_id='track-NOT-IN-MAP')])
        track = client.get('/api/sync?limit=10').get_json()['tracks'][0]
        assert track['umap_x'] is None and track['umap_y'] is None

    def test_projection_unavailable(self, bp_mod, client, fake_db):
        _, cur = fake_db
        bp_mod.load_map_projection = MagicMock(return_value=(None, None))
        _setup_payload(cur, total=1, tracks=[_minimal_track_row()])
        track = client.get('/api/sync?limit=1').get_json()['tracks'][0]
        assert track['umap_x'] is None and track['umap_y'] is None


class TestErrorHandling:
    def test_db_error_returns_structured_503(self, bp_mod, client, fake_db):
        _, cur = fake_db
        _setup_payload(cur, total=0, tracks=[])
        cur._raise_on_execute = RuntimeError("simulated DB failure")
        resp = client.get('/api/sync?limit=1')
        # A DB failure on this data endpoint now surfaces the coded database error
        # (503 Service Unavailable) instead of a generic, uncoded 500.
        assert resp.status_code == 503
        body = resp.get_json()
        assert body['error_code'] == 4002
        assert 'error' in body
        assert 'simulated DB failure' not in body['error']
