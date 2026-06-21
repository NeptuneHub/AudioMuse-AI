"""Input-validation / bound-cap tests for PR.md theme #5.

Covers the search 'limit' caps (500 tracks in app_ivf, 100 artists in
app_artist_similarity), 'n' coercion+clamping (app_alchemy /api/alchemy,
app_ivf /api/similar_tracks), and playlist/anchor name type+empty checks
(app_ivf /api/create_playlist, app_alchemy /api/anchors).

The backend index/db calls are patched per test so no real backend runs;
every assertion checks the value actually handed to the (mocked) backend,
or the HTTP status, so none of them are tautologies.
"""
from unittest.mock import patch
import pytest
from flask import Flask

import config
import app_ivf
import app_artist_similarity
import app_alchemy


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(app_ivf.ivf_bp)
    app.register_blueprint(app_artist_similarity.artist_similarity_bp)
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture
def alchemy_client():
    app = Flask(__name__)
    app.register_blueprint(app_alchemy.alchemy_bp)
    app.config['TESTING'] = True
    return app.test_client()


class TestIvfSearchLimitCap:
    def test_huge_end_is_capped_to_500(self, client):
        with patch.object(app_ivf, 'search_tracks_unified', return_value=[]) as backend:
            resp = client.get('/api/search_tracks',
                              query_string={'search_query': 'a', 'start': 0, 'end': 999999})
        assert resp.status_code == 200
        used = backend.call_args.kwargs['limit']
        assert used == 500
        assert used < 999999

    def test_small_range_is_not_inflated(self, client):
        with patch.object(app_ivf, 'search_tracks_unified', return_value=[]) as backend:
            resp = client.get('/api/search_tracks',
                              query_string={'search_query': 'a', 'start': 0, 'end': 30})
        assert resp.status_code == 200
        assert backend.call_args.kwargs['limit'] == 30


class TestArtistSearchLimitCap:
    def test_huge_end_is_capped_to_100(self, client):
        with patch.object(app_artist_similarity, 'search_artists_by_name', return_value=[]) as backend:
            resp = client.get('/api/search_artists',
                              query_string={'query': 'ab', 'start': 0, 'end': 999999})
        assert resp.status_code == 200
        used = backend.call_args.kwargs['limit']
        assert used == 100
        assert used < 999999

    def test_small_range_is_not_inflated(self, client):
        with patch.object(app_artist_similarity, 'search_artists_by_name', return_value=[]) as backend:
            resp = client.get('/api/search_artists',
                              query_string={'query': 'ab', 'start': 0, 'end': 25})
        assert resp.status_code == 200
        assert backend.call_args.kwargs['limit'] == 25


class TestIvfSimilarTracksNCoercion:
    def test_non_numeric_n_falls_back_to_default_not_500(self, client):
        with patch.object(app_ivf, 'find_nearest_neighbors_by_id', return_value=[]) as backend:
            resp = client.get('/api/similar_tracks',
                              query_string={'item_id': 'x', 'n': 'notanint'})
        # 404 because the (mocked) search found nothing -- never a 500/crash.
        assert resp.status_code != 500
        assert backend.call_args.kwargs['n'] == 10

    def test_valid_n_is_passed_through(self, client):
        with patch.object(app_ivf, 'find_nearest_neighbors_by_id', return_value=[]) as backend:
            resp = client.get('/api/similar_tracks',
                              query_string={'item_id': 'x', 'n': 7})
        assert resp.status_code != 500
        assert backend.call_args.kwargs['n'] == 7

    def test_negative_n_is_clamped_to_a_valid_bound(self, client):
        with patch.object(app_ivf, 'find_nearest_neighbors_by_id', return_value=[]) as backend:
            client.get('/api/similar_tracks', query_string={'item_id': 'x', 'n': -5})
        assert backend.call_args.kwargs['n'] >= 1


class TestAlchemyNClamp:
    def test_huge_n_is_clamped_to_max(self, alchemy_client):
        with patch.object(app_alchemy, 'song_alchemy', return_value={'results': []}) as backend, \
                patch.object(app_alchemy, 'attach_song_features'):
            resp = alchemy_client.post('/api/alchemy',
                                       json={'items': [{'op': 'ADD', 'id': 's1'}], 'n': 999999})
        assert resp.status_code == 200
        used = backend.call_args.kwargs['n_results']
        assert used == config.ALCHEMY_MAX_N_RESULTS
        assert used < 999999

    def test_non_numeric_n_falls_back_to_default(self, alchemy_client):
        with patch.object(app_alchemy, 'song_alchemy', return_value={'results': []}) as backend, \
                patch.object(app_alchemy, 'attach_song_features'):
            resp = alchemy_client.post('/api/alchemy',
                                       json={'items': [{'op': 'ADD', 'id': 's1'}], 'n': 'notanint'})
        assert resp.status_code == 200
        assert backend.call_args.kwargs['n_results'] == config.ALCHEMY_DEFAULT_N_RESULTS

    def test_negative_n_is_clamped_to_min_one(self, alchemy_client):
        with patch.object(app_alchemy, 'song_alchemy', return_value={'results': []}) as backend, \
                patch.object(app_alchemy, 'attach_song_features'):
            resp = alchemy_client.post('/api/alchemy',
                                       json={'items': [{'op': 'ADD', 'id': 's1'}], 'n': -7})
        assert resp.status_code == 200
        used = backend.call_args.kwargs['n_results']
        assert used == 1
        assert used >= 1


class TestIvfCreatePlaylistName:
    def test_empty_name_rejected_400(self, client):
        with patch.object(app_ivf, 'create_playlist_from_ids') as backend:
            resp = client.post('/api/create_playlist',
                               json={'playlist_name': '', 'track_ids': ['a']})
        assert resp.status_code == 400
        backend.assert_not_called()

    def test_missing_name_rejected_400(self, client):
        with patch.object(app_ivf, 'create_playlist_from_ids') as backend:
            resp = client.post('/api/create_playlist', json={'track_ids': ['a']})
        assert resp.status_code == 400
        backend.assert_not_called()

    def test_non_string_name_rejected_4xx(self, client):
        with patch.object(app_ivf, 'create_playlist_from_ids', return_value='pid') as backend:
            resp = client.post('/api/create_playlist',
                               json={'playlist_name': 123, 'track_ids': ['a']})
        assert 400 <= resp.status_code < 500
        backend.assert_not_called()


class TestAlchemyAnchorName:
    def test_empty_name_rejected_400(self, alchemy_client):
        with patch('database.save_alchemy_anchor') as backend:
            resp = alchemy_client.post('/api/anchors',
                                       json={'name': '', 'centroid': [1.0, 2.0]})
        assert resp.status_code == 400
        backend.assert_not_called()

    def test_whitespace_name_rejected_400(self, alchemy_client):
        with patch('database.save_alchemy_anchor') as backend:
            resp = alchemy_client.post('/api/anchors',
                                       json={'name': '   ', 'centroid': [1.0, 2.0]})
        assert resp.status_code == 400
        backend.assert_not_called()

    def test_non_string_name_rejected_4xx(self, alchemy_client):
        with patch('database.save_alchemy_anchor', return_value={'id': 1, 'name': 'n'}) as backend:
            resp = alchemy_client.post('/api/anchors',
                                       json={'name': 123, 'centroid': [1.0, 2.0]})
        assert 400 <= resp.status_code < 500
        backend.assert_not_called()
