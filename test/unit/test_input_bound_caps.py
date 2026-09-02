# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Input bounds and name validation on the search and alchemy API routes.

Drives the IVF, artist-similarity, and alchemy blueprints to confirm untrusted
range and count params are bounded and playlist/anchor names are validated.

Two different knobs, two different rules. A PAGINATION page size (search
start/end) is capped, because it bounds one response body. A requested RESULT
COUNT - how many neighbours/songs to return - is NEVER capped in the backend:
its configured maximum bounds only the page's own input box, so a direct API
caller may ask for any number. Both get a floor and a default.

Main Features:
* Search pagination limits cap at 500 (tracks) / 100 (artists)
* Similar-tracks and alchemy n coerce non-numeric to the default and floor
  negatives at 1, while a huge count reaches the backend unchanged
* Empty, whitespace, and non-string playlist/anchor names are rejected 4xx
  before the backend is ever called
"""

import io
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
            resp = client.get(
                '/api/search_tracks', query_string={'search_query': 'a', 'start': 0, 'end': 999999}
            )
        assert resp.status_code == 200
        used = backend.call_args.kwargs['limit']
        assert used == 500
        assert used < 999999

    def test_small_range_is_not_inflated(self, client):
        with patch.object(app_ivf, 'search_tracks_unified', return_value=[]) as backend:
            resp = client.get(
                '/api/search_tracks', query_string={'search_query': 'a', 'start': 0, 'end': 30}
            )
        assert resp.status_code == 200
        assert backend.call_args.kwargs['limit'] == 30


class TestArtistSearchLimitCap:
    def test_huge_end_is_capped_to_100(self, client):
        with patch.object(
            app_artist_similarity, 'search_artists_by_name', return_value=[]
        ) as backend:
            resp = client.get(
                '/api/search_artists', query_string={'query': 'ab', 'start': 0, 'end': 999999}
            )
        assert resp.status_code == 200
        used = backend.call_args.kwargs['limit']
        assert used == 100
        assert used < 999999

    def test_small_range_is_not_inflated(self, client):
        with patch.object(
            app_artist_similarity, 'search_artists_by_name', return_value=[]
        ) as backend:
            resp = client.get(
                '/api/search_artists', query_string={'query': 'ab', 'start': 0, 'end': 25}
            )
        assert resp.status_code == 200
        assert backend.call_args.kwargs['limit'] == 25


class TestIvfSimilarTracksNCoercion:
    def test_non_numeric_n_falls_back_to_default_not_500(self, client):
        with patch.object(app_ivf, 'find_nearest_neighbors_by_id', return_value=[]) as backend:
            resp = client.get('/api/similar_tracks', query_string={'item_id': 'x', 'n': 'notanint'})
        assert resp.status_code != 500
        assert backend.call_args.kwargs['n'] == config.SIMILARITY_DEFAULT_N_RESULTS

    def test_valid_n_is_passed_through(self, client):
        with patch.object(app_ivf, 'find_nearest_neighbors_by_id', return_value=[]) as backend:
            resp = client.get('/api/similar_tracks', query_string={'item_id': 'x', 'n': 7})
        assert resp.status_code != 500
        assert backend.call_args.kwargs['n'] == 7

    def test_negative_n_is_clamped_to_a_valid_bound(self, client):
        with patch.object(app_ivf, 'find_nearest_neighbors_by_id', return_value=[]) as backend:
            client.get('/api/similar_tracks', query_string={'item_id': 'x', 'n': -5})
        assert backend.call_args.kwargs['n'] >= 1


class TestAlchemyNClamp:
    def test_n_above_the_frontend_maximum_reaches_the_backend_uncapped(self, alchemy_client):
        with (
            patch.object(app_alchemy, 'song_alchemy', return_value={'results': []}) as backend,
            patch.object(app_alchemy, 'attach_song_features'),
        ):
            resp = alchemy_client.post(
                '/api/alchemy', json={'items': [{'op': 'ADD', 'id': 's1'}], 'n': 999999}
            )
        assert resp.status_code == 200
        used = backend.call_args.kwargs['n_results']
        assert used == 999999
        assert used > config.ALCHEMY_MAX_N_RESULTS

    def test_non_numeric_n_falls_back_to_default(self, alchemy_client):
        with (
            patch.object(app_alchemy, 'song_alchemy', return_value={'results': []}) as backend,
            patch.object(app_alchemy, 'attach_song_features'),
        ):
            resp = alchemy_client.post(
                '/api/alchemy', json={'items': [{'op': 'ADD', 'id': 's1'}], 'n': 'notanint'}
            )
        assert resp.status_code == 200
        assert backend.call_args.kwargs['n_results'] == config.ALCHEMY_DEFAULT_N_RESULTS

    def test_negative_n_is_clamped_to_min_one(self, alchemy_client):
        with (
            patch.object(app_alchemy, 'song_alchemy', return_value={'results': []}) as backend,
            patch.object(app_alchemy, 'attach_song_features'),
        ):
            resp = alchemy_client.post(
                '/api/alchemy', json={'items': [{'op': 'ADD', 'id': 's1'}], 'n': -7}
            )
        assert resp.status_code == 200
        used = backend.call_args.kwargs['n_results']
        assert used == 1
        assert used >= 1


class TestIvfCreatePlaylistName:
    def test_empty_name_rejected_400(self, client):
        with patch('app_server_context.create_instant_playlist_for_server') as backend:
            resp = client.post(
                '/api/create_playlist', json={'playlist_name': '', 'track_ids': ['a']}
            )
        assert resp.status_code == 400
        backend.assert_not_called()

    def test_missing_name_rejected_400(self, client):
        with patch('app_server_context.create_instant_playlist_for_server') as backend:
            resp = client.post('/api/create_playlist', json={'track_ids': ['a']})
        assert resp.status_code == 400
        backend.assert_not_called()

    def test_non_string_name_rejected_4xx(self, client):
        with patch(
            'app_server_context.create_instant_playlist_for_server',
            return_value={'result': 'pid', 'requested': 1, 'mapped': 1, 'skipped': 0},
        ) as backend:
            resp = client.post(
                '/api/create_playlist', json={'playlist_name': 123, 'track_ids': ['a']}
            )
        assert 400 <= resp.status_code < 500
        backend.assert_not_called()


class TestAlchemyAnchorName:
    def test_empty_name_rejected_400(self, alchemy_client):
        with patch('database.save_alchemy_anchor') as backend:
            resp = alchemy_client.post('/api/anchors', json={'name': '', 'centroid': [1.0, 2.0]})
        assert resp.status_code == 400
        backend.assert_not_called()

    def test_whitespace_name_rejected_400(self, alchemy_client):
        with patch('database.save_alchemy_anchor') as backend:
            resp = alchemy_client.post('/api/anchors', json={'name': '   ', 'centroid': [1.0, 2.0]})
        assert resp.status_code == 400
        backend.assert_not_called()

    def test_non_string_name_rejected_4xx(self, alchemy_client):
        with patch('database.save_alchemy_anchor', return_value={'id': 1, 'name': 'n'}) as backend:
            resp = alchemy_client.post('/api/anchors', json={'name': 123, 'centroid': [1.0, 2.0]})
        assert 400 <= resp.status_code < 500
        backend.assert_not_called()


class TestNoBackendCeilingOnRequestedCounts:
    def test_alchemy_engine_does_not_reclamp_the_count_it_is_handed(self):
        import tasks.song_alchemy as sa

        src = io.open(sa.__file__, encoding='utf-8').read()
        assert 'min(n_results, config.ALCHEMY_MAX_N_RESULTS)' not in src

    def test_no_route_clamps_a_requested_count_to_a_configured_maximum(self):
        import glob
        import re

        offenders = []
        for path in sorted(glob.glob('app_*.py')) + ['tasks/song_alchemy.py']:
            for num, line in enumerate(io.open(path, encoding='utf-8'), start=1):
                m = re.search(r'min\(\s*(n_results|num_neighbors|get_songs|n)\s*,\s*([^),]+)', line)
                if m and not m.group(2).strip().startswith('len('):
                    offenders.append(path + ':' + str(num) + ': ' + line.strip())
        assert offenders == [], (
            'A requested result count (neighbours/songs) is clamped in the '
            'backend. Its only ceiling belongs in the page input; the API stays '
            'uncapped. Pagination page size is a separate knob and may cap: '
            + '\n'.join(offenders)
        )
