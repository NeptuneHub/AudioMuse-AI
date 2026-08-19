# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Song Alchemy centroid math and playlist-component generation.

Covers the alchemy engine that blends song/artist centroids into a new playlist,
including vector arithmetic, temperature sampling and cluster component selection.

Main Features:
* Temperature sampling and euclidean/angular metric distance behavior
* get_playlist_components uses cell groups, caps clusters and samples large playlists
* Artist anchors expand their GMM components into weighted points that blend
  into the ADD centroid
* Full alchemy flow dedups songs and applies the distance filter
* ADD-ed anchors re-apply their stored exclusions at the saved per-point radius
  and the run exports subtract regions as `exclusions` for anchor saving
"""

import pytest
from unittest.mock import patch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tasks import song_alchemy


def _score_side_effect(details_by_id):
    def _fn(ids):
        return [details_by_id[i] for i in ids if i in details_by_id]

    return _fn


class TestSongAlchemy:
    @pytest.fixture
    def mock_dependencies(self):
        with (
            patch('tasks.song_alchemy.get_vector_by_id') as mock_get_vec,
            patch('tasks.song_alchemy.multi_query_ids') as mock_multi_query,
            patch('tasks.song_alchemy.find_nearest_neighbors_by_id') as mock_find_nn_id,
            patch('tasks.song_alchemy.get_score_data_by_ids') as mock_get_score,
            patch('tasks.song_alchemy._filter_by_distance') as mock_filter_dist,
            patch('database.get_db') as mock_get_db,
            patch('tasks.song_alchemy.load_map_projection') as mock_load_proj,
            patch('tasks.song_alchemy.config') as mock_config,
        ):
            mock_filter_dist.side_effect = lambda song_results, db_conn: song_results

            mock_config.ALCHEMY_DEFAULT_N_RESULTS = 10
            mock_config.ALCHEMY_MAX_N_RESULTS = 50
            mock_config.ALCHEMY_TEMPERATURE = 1.0
            mock_config.PATH_DISTANCE_METRIC = 'euclidean'
            mock_config.ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN = 0.5
            mock_config.ALCHEMY_SUBTRACT_DISTANCE_ANGULAR = 0.2
            mock_config.ALCHEMY_MAX_ANCHOR_POINTS = 16
            mock_config.ALCHEMY_PLAYLIST_MAX_SONGS = 500
            mock_config.ALCHEMY_PLAYLIST_MAX_CENTROIDS = 10

            yield {
                'get_vector_by_id': mock_get_vec,
                'multi_query_ids': mock_multi_query,
                'find_nearest_neighbors_by_id': mock_find_nn_id,
                'get_score_data_by_ids': mock_get_score,
                'filter_by_distance': mock_filter_dist,
                'get_db': mock_get_db,
                'load_map_projection': mock_load_proj,
                'config': mock_config,
            }

    def test_song_alchemy_basic_flow(self, mock_dependencies):
        mock_dependencies['get_vector_by_id'].return_value = [1.0, 0.0]
        mock_dependencies['multi_query_ids'].return_value = ['r1', 'r2']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                's1': {'item_id': 's1', 'title': 'Seed', 'author': 'Seed Author'},
                'r1': {'item_id': 'r1', 'title': 'Result 1', 'author': 'Author 1'},
                'r2': {'item_id': 'r2', 'title': 'Result 2', 'author': 'Author 2'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        result = song_alchemy.song_alchemy(add_items=[{'type': 'song', 'id': 's1'}], n_results=5)

        assert len(result['results']) == 2
        assert result['results'][0]['item_id'] in ['r1', 'r2']
        assert 'projection' in result

    def test_song_alchemy_subtraction(self, mock_dependencies):
        def get_vec(id):
            vectors = {'s1': [1.0, 0.0], 'sub1': [0.0, 1.0], 'r1': [0.9, 0.1], 'r2': [0.1, 0.9]}
            return vectors.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec

        mock_dependencies['multi_query_ids'].return_value = ['r1', 'r2']

        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                's1': {'item_id': 's1', 'title': 'Seed', 'author': 'A0'},
                'sub1': {'item_id': 'sub1', 'title': 'Sub', 'author': 'A0'},
                'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'},
                'r2': {'item_id': 'r2', 'title': 'R2', 'author': 'A2'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        result = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}],
            subtract_items=[{'type': 'song', 'id': 'sub1'}],
            subtract_distance=0.5,
        )

        result_ids = [r['item_id'] for r in result['results']]
        filtered_ids = [r['item_id'] for r in result['filtered_out']]

        assert 'r1' in result_ids
        assert 'r2' in filtered_ids

    def test_project_to_2d(self):
        vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        proj = song_alchemy._project_to_2d(vectors)
        assert len(proj) == 3
        assert len(proj[0]) == 2
        for p in proj:
            assert -1.0 <= p[0] <= 1.0
            assert -1.0 <= p[1] <= 1.0

    def test_temperature_sampling(self, mock_dependencies):
        mock_dependencies['get_vector_by_id'].return_value = [1.0, 0.0]

        mock_dependencies['multi_query_ids'].return_value = ['r1', 'r2']
        mock_dependencies['find_nearest_neighbors_by_id'].return_value = [
            {'item_id': 'r1', 'score': 0.1},
            {'item_id': 'r2', 'score': 0.2},
        ]
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                's1': {'item_id': 's1', 'title': 'Seed', 'author': 'A0'},
                'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'},
                'r2': {'item_id': 'r2', 'title': 'R2', 'author': 'A2'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        result_zero = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}], temperature=0.0
        )
        assert len(result_zero['results']) > 0

        result_high = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}], temperature=10.0
        )
        assert len(result_high['results']) > 0

    def test_metric_distance_euclidean_and_angular(self, mock_dependencies):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        mock_dependencies['config'].PATH_DISTANCE_METRIC = 'euclidean'
        assert np.isclose(song_alchemy._metric_distance(a, b), np.sqrt(2.0))
        mock_dependencies['config'].PATH_DISTANCE_METRIC = 'angular'
        assert np.isclose(song_alchemy._metric_distance(a, b), 0.5)

    def test_get_playlist_components_uses_cell_groups(self, mock_dependencies):
        groups = [(np.array([10.0, 0.0]), 60), (np.array([0.0, 10.0]), 40)]
        with (
            patch('tasks.mediaserver.get_playlist_track_ids', return_value=['t0', 't1']),
            patch('tasks.ivf_manager.get_cell_groups_for_items', return_value=groups),
        ):
            vecs, weights = song_alchemy._get_playlist_components('pl1')

        assert len(vecs) == 2
        assert np.allclose(weights, [0.6, 0.4])

    def test_get_playlist_components_caps_clusters_at_max(self, mock_dependencies):
        groups = [(np.array([float(i), 0.0]), 1) for i in range(40)]
        with (
            patch(
                'tasks.mediaserver.get_playlist_track_ids',
                return_value=[f't{i}' for i in range(40)],
            ),
            patch('tasks.ivf_manager.get_cell_groups_for_items', return_value=groups),
        ):
            vecs, weights = song_alchemy._get_playlist_components('pl1')

        assert len(vecs) == 10
        assert np.isclose(sum(weights), 1.0)

    def test_get_playlist_components_coherent_returns_single(self, mock_dependencies):
        groups = [(np.array([1.0, 0.0]), 50)]
        with (
            patch('tasks.mediaserver.get_playlist_track_ids', return_value=['t0']),
            patch('tasks.ivf_manager.get_cell_groups_for_items', return_value=groups),
        ):
            vecs, weights = song_alchemy._get_playlist_components('pl1')

        assert len(vecs) == 1
        assert weights == [1.0]

    def test_get_playlist_components_samples_large_playlist(self, mock_dependencies):
        mock_dependencies['config'].ALCHEMY_PLAYLIST_MAX_SONGS = 50
        track_ids = [f't{i}' for i in range(200)]
        captured = {}

        def fake_groups(ids):
            captured['n'] = len(list(ids))
            return [(np.array([1.0, 0.0]), captured['n'])]

        with (
            patch('tasks.mediaserver.get_playlist_track_ids', return_value=track_ids),
            patch('tasks.ivf_manager.get_cell_groups_for_items', side_effect=fake_groups),
        ):
            _, weights = song_alchemy._get_playlist_components('pl1')

        assert captured['n'] == 50
        assert np.isclose(sum(weights), 1.0)

    def test_get_playlist_components_no_index_match(self, mock_dependencies):
        with (
            patch('tasks.mediaserver.get_playlist_track_ids', return_value=['t0', 't1']),
            patch('tasks.ivf_manager.get_cell_groups_for_items', return_value=[]),
        ):
            vecs, weights = song_alchemy._get_playlist_components('pl1')

        assert vecs == []
        assert weights == []

    def test_select_spread_centroids_picks_far_apart(self, mock_dependencies):
        groups = [
            (np.array([0.0, 0.0]), 100),
            (np.array([0.2, 0.0]), 5),
            (np.array([10.0, 0.0]), 30),
        ]
        kept = song_alchemy._select_spread_centroids(groups, 2)
        kept_x = sorted(float(v[0]) for v, _ in kept)
        assert kept_x == [0.0, 10.0]

    def test_gather_anchor_points_playlist_expands(self, mock_dependencies):
        with patch(
            'tasks.song_alchemy._get_playlist_components',
            return_value=([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [0.6, 0.4]),
        ):
            points = song_alchemy._gather_anchor_points([{'type': 'playlist', 'id': 'pl1'}])

        assert len(points) == 2
        assert all(p['source_type'] == 'playlist' for p in points)
        assert [p['comp_idx'] for p in points] == [0, 1]

    def test_gather_anchor_points_artist_expands_gmm_components(self):
        with patch(
            'tasks.song_alchemy._get_artist_gmm_vectors_and_weights',
            return_value=([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [0.75, 0.25]),
        ):
            points = song_alchemy._gather_anchor_points([{'type': 'artist', 'id': 'art1'}])

        assert [p['source_type'] for p in points] == ['artist', 'artist']
        assert [p['source_id'] for p in points] == ['art1', 'art1']
        assert [p['comp_idx'] for p in points] == [0, 1]
        assert [p['weight'] for p in points] == [0.75, 0.25]
        assert np.allclose(points[0]['vector'], [1.0, 0.0])
        assert np.allclose(points[1]['vector'], [0.0, 1.0])

    def test_artist_gmm_components_blend_into_weighted_add_centroid(self):
        with patch(
            'tasks.song_alchemy._get_artist_gmm_vectors_and_weights',
            return_value=([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [0.75, 0.25]),
        ):
            points = song_alchemy._gather_anchor_points([{'type': 'artist', 'id': 'art1'}])
        centroid = song_alchemy._compute_centroid_from_points(points)

        assert np.allclose(centroid, [0.75, 0.25])

    def test_song_alchemy_playlist_matches_any_cluster(self, mock_dependencies):
        def get_vec(id):
            return {
                'cand_a': [1.0, 0.0],
                'cand_b': [0.0, 1.0],
                'mid': [0.5, 0.5],
            }.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec
        mock_dependencies['multi_query_ids'].return_value = ['cand_a', 'cand_b', 'mid']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                'cand_a': {'item_id': 'cand_a', 'title': 'A', 'author': 'AA'},
                'cand_b': {'item_id': 'cand_b', 'title': 'B', 'author': 'BB'},
                'mid': {'item_id': 'mid', 'title': 'M', 'author': 'MM'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch(
            'tasks.song_alchemy._get_playlist_components',
            return_value=([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [0.5, 0.5]),
        ):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'playlist', 'id': 'pl1'}], temperature=0.0
            )

        ids = [r['item_id'] for r in result['results']]
        assert 'cand_a' in ids and 'cand_b' in ids
        assert ids.index('mid') == len(ids) - 1
        assert np.isclose(result['results'][0]['distance'], 0.0)

    def test_song_alchemy_dedups_duplicate_songs(self, mock_dependencies):
        mock_dependencies['get_vector_by_id'].return_value = [1.0, 0.0]
        mock_dependencies['multi_query_ids'].return_value = [
            'dupe_a',
            'dupe_b',
            'unique',
            'seed_clone',
        ]
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                's1': {'item_id': 's1', 'title': 'Seed Song', 'author': 'Seed Artist'},
                'dupe_a': {'item_id': 'dupe_a', 'title': 'Same Song', 'author': 'Same Artist'},
                'dupe_b': {'item_id': 'dupe_b', 'title': 'same song', 'author': 'SAME ARTIST'},
                'unique': {'item_id': 'unique', 'title': 'Other', 'author': 'Other Artist'},
                'seed_clone': {
                    'item_id': 'seed_clone',
                    'title': 'Seed Song',
                    'author': 'Seed Artist',
                },
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        result = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}],
            temperature=1.0,
        )

        ids = [r['item_id'] for r in result['results']]
        assert ids.count('dupe_a') + ids.count('dupe_b') == 1
        assert 'unique' in ids
        assert 'seed_clone' not in ids

    def test_song_alchemy_exports_subtract_run_as_exclusions(self, mock_dependencies):
        def get_vec(id):
            vectors = {'s1': [1.0, 0.0], 'sub1': [0.0, 1.0], 'r1': [0.9, 0.1]}
            return vectors.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec
        mock_dependencies['multi_query_ids'].return_value = ['r1']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                's1': {'item_id': 's1', 'title': 'Seed', 'author': 'A0'},
                'sub1': {'item_id': 'sub1', 'title': 'Sub', 'author': 'A0'},
                'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        result = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}],
            subtract_items=[{'type': 'song', 'id': 'sub1'}],
            subtract_distance=0.5,
        )

        assert result['exclusions'] == [{'vector': [0.0, 1.0], 'distance': 0.5}]

    def test_song_alchemy_no_subtract_returns_empty_exclusions(self, mock_dependencies):
        mock_dependencies['get_vector_by_id'].return_value = [1.0, 0.0]
        mock_dependencies['multi_query_ids'].return_value = ['r1']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                's1': {'item_id': 's1', 'title': 'Seed', 'author': 'A0'},
                'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        result = song_alchemy.song_alchemy(add_items=[{'type': 'song', 'id': 's1'}])

        assert result['exclusions'] == []

    def test_song_alchemy_added_anchor_reapplies_stored_exclusions(self, mock_dependencies):
        anchor = {
            'id': 7,
            'name': 'Anchor',
            'centroid': [1.0, 0.0],
            'exclusions': [{'vector': [0.0, 1.0], 'distance': 0.5}],
        }

        def get_vec(id):
            vectors = {'near_ex': [0.1, 0.9], 'far': [0.9, 0.1]}
            return vectors.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec
        mock_dependencies['multi_query_ids'].return_value = ['near_ex', 'far']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                'near_ex': {'item_id': 'near_ex', 'title': 'Near', 'author': 'A1'},
                'far': {'item_id': 'far', 'title': 'Far', 'author': 'A2'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}], temperature=0.0
            )

        result_ids = [r['item_id'] for r in result['results']]
        filtered_ids = [r['item_id'] for r in result['filtered_out']]
        assert 'far' in result_ids
        assert 'near_ex' in filtered_ids

    def test_song_alchemy_stored_exclusion_radius_beats_request_distance(
        self, mock_dependencies
    ):
        anchor = {
            'id': 7,
            'name': 'Anchor',
            'centroid': [1.0, 0.0],
            'exclusions': [{'vector': [0.0, 1.0], 'distance': 0.05}],
        }

        def get_vec(id):
            vectors = {'edge': [0.1, 0.9], 'inside': [0.0, 0.99]}
            return vectors.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec
        mock_dependencies['multi_query_ids'].return_value = ['edge', 'inside']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                'edge': {'item_id': 'edge', 'title': 'Edge', 'author': 'A1'},
                'inside': {'item_id': 'inside', 'title': 'Inside', 'author': 'A2'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}],
                subtract_distance=1.0,
                temperature=0.0,
            )

        result_ids = [r['item_id'] for r in result['results']]
        filtered_ids = [r['item_id'] for r in result['filtered_out']]
        assert 'edge' in result_ids
        assert 'inside' in filtered_ids

    def test_song_alchemy_exclusions_merge_anchor_and_explicit_subtract(
        self, mock_dependencies
    ):
        anchor = {
            'id': 7,
            'name': 'Anchor',
            'centroid': [1.0, 0.0],
            'exclusions': [{'vector': [0.0, 1.0], 'distance': 0.25}],
        }

        def get_vec(id):
            vectors = {'sub1': [1.0, 1.0], 'r1': [0.9, 0.1]}
            return vectors.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec
        mock_dependencies['multi_query_ids'].return_value = ['r1']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                'sub1': {'item_id': 'sub1', 'title': 'Sub', 'author': 'A0'},
                'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}],
                subtract_items=[{'type': 'song', 'id': 'sub1'}],
                subtract_distance=0.4,
                temperature=0.0,
            )

        assert result['exclusions'] == [
            {'vector': [1.0, 1.0], 'distance': 0.4},
            {'vector': [0.0, 1.0], 'distance': 0.25},
        ]

    def test_song_alchemy_skips_malformed_stored_exclusions(self, mock_dependencies):
        anchor = {
            'id': 7,
            'name': 'Anchor',
            'centroid': [1.0, 0.0],
            'exclusions': [
                {'vector': 'bad'},
                {'vector': []},
                'not-a-dict',
                {'vector': [0.0, 1.0], 'distance': 'bad'},
            ],
        }

        mock_dependencies['get_vector_by_id'].side_effect = lambda x: {'r1': [0.1, 0.9]}.get(x)
        mock_dependencies['multi_query_ids'].return_value = ['r1']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'}}
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}], temperature=0.0
            )

        assert [r['item_id'] for r in result['results']] == ['r1']
        assert result['exclusions'] == []

    def test_song_alchemy_applies_distance_filter(self, mock_dependencies):
        mock_dependencies['get_vector_by_id'].return_value = [1.0, 0.0]
        mock_dependencies['multi_query_ids'].return_value = ['near_dup', 'keep']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                's1': {'item_id': 's1', 'title': 'Seed', 'author': 'A0'},
                'near_dup': {'item_id': 'near_dup', 'title': 'Near Dup', 'author': 'A1'},
                'keep': {'item_id': 'keep', 'title': 'Keep', 'author': 'A2'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)
        mock_dependencies['filter_by_distance'].side_effect = lambda song_results, db_conn: [
            s for s in song_results if s['item_id'] != 'near_dup'
        ]

        result = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}],
            temperature=1.0,
        )

        ids = [r['item_id'] for r in result['results']]
        assert 'keep' in ids
        assert 'near_dup' not in ids
        assert mock_dependencies['filter_by_distance'].called
