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
* The run exports every ADD point, not averaged, as `inclusions`; an ADD-ed
  anchor with stored inclusions searches each point, ranks by the nearest one,
  keeps stored song seeds out of the results, falls back to its centroid when
  none are stored, and is ignored when its embedding stamp or centroid size no
  longer matches (search, exclusions and projection alike, loaded once per run)
* A subtracted anchor excludes around each stored point and drops its seed
  songs; capped query points keep every input's heaviest point, and a saved
  anchor re-run picks the same points as its run through the stored groups;
  stored seed signatures drop other copies of a seed; the seed test ignores
  metric settings and int8 rounding; the embedding stamp fingerprints the model
  file content and tolerates an unreadable model on the dimension alone
"""

import hashlib
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
            patch(
                'tasks.song_alchemy.anchor_embedding_tag',
                return_value={'embedding_model_sha256': 'musicnn_embedding.onnx', 'dimension': 2},
            ),
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
            mock_config.EMBEDDING_DIMENSION = 2
            mock_config.EMBEDDING_MODEL_PATH = '/nonexistent/model/musicnn_embedding.onnx'

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

    @staticmethod
    def _tagged_anchor(points, model='musicnn_embedding.onnx', dimension=2, anchor_id=7):
        return {
            'id': anchor_id,
            'name': f'Anchor {anchor_id}',
            'centroid': [0.5, 0.5],
            'exclusions': [{'vector': [0.0, -1.0], 'distance': 0.1}],
            'inclusions': {'embedding_model_sha256': model, 'dimension': dimension, 'points': points},
        }

    def test_song_alchemy_exports_add_points_as_inclusions(self, mock_dependencies):
        def get_vec(id):
            vectors = {'s1': [1.0, 0.0], 's2': [0.0, 1.0], 'r1': [0.9, 0.1]}
            return vectors.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec
        mock_dependencies['multi_query_ids'].return_value = ['r1']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'}}
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        result = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}, {'type': 'song', 'id': 's2'}]
        )

        assert result['inclusions'] == [
            {'vector': [1.0, 0.0], 'weight': 1.0, 'seed': True, 'group': 0, 'signature': None},
            {'vector': [0.0, 1.0], 'weight': 1.0, 'seed': True, 'group': 1, 'signature': None},
        ]
        assert result['add_centroid_vector'] == [0.5, 0.5]
        assert result['inclusions_embedding'] == {'embedding_model_sha256': 'musicnn_embedding.onnx', 'dimension': 2}

    def test_song_alchemy_anchor_with_inclusions_searches_every_point(self, mock_dependencies):
        anchor = self._tagged_anchor(
            [
                {'vector': [1.0, 0.0], 'weight': 1.0, 'seed': True},
                {'vector': [0.0, 1.0], 'weight': 3.0, 'seed': False},
            ]
        )

        def get_vec(id):
            vectors = {
                'seed_copy': [1.0, 0.0],
                'near_a': [0.8, 0.2],
                'near_b': [0.2, 0.8],
                'middle': [0.6, 0.8],
            }
            return vectors.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec
        mock_dependencies['multi_query_ids'].return_value = ['seed_copy', 'middle', 'near_a', 'near_b']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                'seed_copy': {'item_id': 'seed_copy', 'title': 'Seed', 'author': 'A0'},
                'near_a': {'item_id': 'near_a', 'title': 'Near A', 'author': 'A1'},
                'near_b': {'item_id': 'near_b', 'title': 'Near B', 'author': 'A2'},
                'middle': {'item_id': 'middle', 'title': 'Middle', 'author': 'A3'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}], n_results=3, temperature=0.0
            )

        query_vectors = mock_dependencies['multi_query_ids'].call_args.args[0]
        assert [list(v) for v in query_vectors] == [[1.0, 0.0], [0.0, 1.0]]
        result_ids = [r['item_id'] for r in result['results']]
        assert 'seed_copy' not in result_ids
        assert result_ids[:2] in (['near_a', 'near_b'], ['near_b', 'near_a'])
        assert result['inclusions'] == [
            {'vector': [1.0, 0.0], 'weight': 0.25, 'seed': True, 'group': 0, 'signature': None},
            {'vector': [0.0, 1.0], 'weight': 0.75, 'seed': False, 'group': 0, 'signature': None},
        ]
        assert result['exclusions'] == [{'vector': [0.0, -1.0], 'distance': 0.1}]

    def test_song_alchemy_anchor_without_inclusions_uses_centroid(self, mock_dependencies):
        anchor = {'id': 7, 'name': 'Anchor', 'centroid': [0.5, 0.5], 'exclusions': None}

        mock_dependencies['get_vector_by_id'].side_effect = lambda x: {'r1': [0.4, 0.6]}.get(x)
        mock_dependencies['multi_query_ids'].return_value = ['r1']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'}}
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}], temperature=0.0
            )

        query_vectors = mock_dependencies['multi_query_ids'].call_args.args[0]
        assert [list(v) for v in query_vectors] == [[0.5, 0.5]]
        assert [r['item_id'] for r in result['results']] == ['r1']
        assert result['inclusions'] == [
            {'vector': [0.5, 0.5], 'weight': 1.0, 'seed': False, 'group': 0, 'signature': None}
        ]

    def test_song_alchemy_skips_malformed_stored_inclusions(self, mock_dependencies):
        anchor = self._tagged_anchor(
            [
                {'vector': 'bad'},
                {'vector': []},
                'not-a-dict',
                {'vector': [0.0, 1.0], 'weight': 'heavy'},
                {'vector': [0.0, 1.0], 'weight': -1.0},
            ]
        )

        mock_dependencies['get_vector_by_id'].side_effect = lambda x: {'r1': [0.4, 0.6]}.get(x)
        mock_dependencies['multi_query_ids'].return_value = ['r1']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'}}
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}], temperature=0.0
            )

        query_vectors = mock_dependencies['multi_query_ids'].call_args.args[0]
        assert [list(v) for v in query_vectors] == [[0.5, 0.5]]
        assert [r['item_id'] for r in result['results']] == ['r1']

    @pytest.mark.parametrize(
        'model, dimension',
        [('other_embedding.onnx', 2), ('musicnn_embedding.onnx', 3)],
    )
    def test_song_alchemy_ignores_anchor_saved_with_another_embedding(
        self, mock_dependencies, model, dimension
    ):
        anchor = self._tagged_anchor(
            [{'vector': [1.0, 0.0], 'weight': 1.0, 'seed': True}], model=model, dimension=dimension
        )
        mock_dependencies['multi_query_ids'].return_value = ['r1']

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}], temperature=0.0
            )

        assert result['results'] == []
        mock_dependencies['multi_query_ids'].assert_not_called()

    def test_song_alchemy_mismatched_anchor_drops_its_exclusions_too(self, mock_dependencies):
        anchor = self._tagged_anchor(
            [{'vector': [1.0, 0.0], 'weight': 1.0}], model='other_embedding.onnx'
        )
        mock_dependencies['get_vector_by_id'].side_effect = lambda x: {
            's1': [1.0, 0.0],
            'r1': [0.0, -1.0],
        }.get(x)
        mock_dependencies['multi_query_ids'].return_value = ['r1']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'}}
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor) as loader:
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'song', 'id': 's1'}, {'type': 'anchor', 'id': 7}],
                temperature=0.0,
            )

        assert [r['item_id'] for r in result['results']] == ['r1']
        assert result['exclusions'] == []
        assert [p for p in result['add_points'] if p.get('type') == 'anchor'] == []
        assert loader.call_count == 1

    def test_ignored_anchor_warning_cannot_forge_log_lines(self, mock_dependencies, caplog):
        anchor = {'id': 7, 'name': 'evil\nFAKE ERROR line', 'centroid': [0.5, 0.5, 0.5], 'exclusions': None}

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor), caplog.at_level('WARNING'):
            song_alchemy._load_usable_anchor('7\r\nforged', {})

        messages = [r.getMessage() for r in caplog.records if 'Ignoring anchor' in r.getMessage()]
        assert len(messages) == 1
        assert '\n' not in messages[0] and '\r' not in messages[0]
        assert "evil FAKE ERROR line" in messages[0]

    def test_song_alchemy_ignores_legacy_anchor_with_wrong_centroid_size(self, mock_dependencies):
        anchor = {'id': 7, 'name': 'Old', 'centroid': [0.5, 0.5, 0.5], 'exclusions': None}
        mock_dependencies['multi_query_ids'].return_value = ['r1']

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}], temperature=0.0
            )

        assert result['results'] == []
        mock_dependencies['multi_query_ids'].assert_not_called()

    def test_song_alchemy_loads_each_anchor_once_per_run(self, mock_dependencies):
        anchor = self._tagged_anchor([{'vector': [1.0, 0.0], 'weight': 1.0}])
        mock_dependencies['get_vector_by_id'].side_effect = lambda x: {'r1': [0.9, 0.1]}.get(x)
        mock_dependencies['multi_query_ids'].return_value = ['r1']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'}}
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor) as loader:
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}], temperature=0.0
            )

        assert [r['item_id'] for r in result['results']] == ['r1']
        assert [p['item_id'] for p in result['add_points'] if p.get('type') == 'anchor'] == [7]
        assert loader.call_count == 1

    def test_select_query_points_keeps_a_many_point_anchor_when_capped(self):
        songs = [
            {'vector': None, 'weight': 1.0, 'source_type': 'song', 'source_id': f's{i}'}
            for i in range(10)
        ]
        artists = [
            {'vector': None, 'weight': 1 / 9, 'source_type': 'artist', 'source_id': name}
            for name in ('a1', 'a2')
            for _ in range(9)
        ]
        anchor = [
            {'vector': None, 'weight': 0.05, 'source_type': 'anchor', 'source_id': 9}
            for _ in range(20)
        ]

        selected = song_alchemy._select_query_points(songs + artists + anchor, 16)

        assert len(selected) == 16
        assert sum(1 for p in selected if p['source_type'] == 'anchor') == 1
        assert sum(1 for p in selected if p['source_type'] == 'song') == 10
        assert {p['source_id'] for p in selected if p['source_type'] == 'artist'} == {'a1', 'a2'}
        assert [p['weight'] for p in selected] == sorted((p['weight'] for p in selected), reverse=True)

    def test_select_query_points_ranks_inputs_by_total_weight(self):
        songs = [
            {'vector': None, 'weight': 1.0, 'source_type': 'song', 'source_id': f's{i}'}
            for i in range(2)
        ]
        anchor = [
            {'vector': None, 'weight': 0.25, 'source_type': 'anchor', 'source_id': 9}
            for _ in range(4)
        ]

        selected = song_alchemy._select_query_points(anchor + songs, 2)

        assert [p['source_type'] for p in selected] == ['song', 'anchor']

    def test_select_query_points_fills_spare_slots_by_weight(self):
        points = [
            {'vector': None, 'weight': w, 'source_type': 'artist', 'source_id': 'a'}
            for w in (0.5, 0.3, 0.2)
        ] + [{'vector': None, 'weight': 1.0, 'source_type': 'song', 'source_id': 's'}]

        selected = song_alchemy._select_query_points(points, 3)

        assert [p['weight'] for p in selected] == [1.0, 0.5, 0.3]

    def test_song_alchemy_subtracted_anchor_excludes_around_each_point(self, mock_dependencies):
        anchor = self._tagged_anchor(
            [{'vector': [0.0, 1.0], 'weight': 1.0}, {'vector': [0.0, -1.0], 'weight': 1.0}],
            anchor_id=8,
        )

        def get_vec(id):
            vectors = {'s1': [1.0, 0.0], 'up': [0.1, 0.95], 'down': [0.1, -0.95], 'keep': [0.9, 0.1]}
            return vectors.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec
        mock_dependencies['multi_query_ids'].return_value = ['up', 'down', 'keep']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                'up': {'item_id': 'up', 'title': 'Up', 'author': 'A1'},
                'down': {'item_id': 'down', 'title': 'Down', 'author': 'A2'},
                'keep': {'item_id': 'keep', 'title': 'Keep', 'author': 'A3'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'song', 'id': 's1'}],
                subtract_items=[{'type': 'anchor', 'id': 8}],
                subtract_distance=0.2,
                temperature=0.5,
            )

        assert [r['item_id'] for r in result['results']] == ['keep']
        assert sorted(r['item_id'] for r in result['filtered_out']) == ['down', 'up']
        assert result['exclusions'] == [
            {'vector': [0.0, 1.0], 'distance': 0.2},
            {'vector': [0.0, -1.0], 'distance': 0.2},
        ]

    def test_stored_seeds_are_dropped_whatever_the_metric_settings(self, mock_dependencies):
        config = mock_dependencies['config']
        config.IVF_METRIC = 'dot'
        config.PATH_DISTANCE_METRIC = 'angular'
        config.DUPLICATE_DISTANCE_THRESHOLD_COSINE = 0.0
        config.DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN = 0.0
        anchor = self._tagged_anchor([{'vector': [0.6, 0.8], 'weight': 1.0, 'seed': True}])

        def get_vec(id):
            vectors = {
                'seed_quantized': [76 / 127, 102 / 127],
                'close': [0.8, 0.6],
                'far': [-0.6, 0.8],
            }
            return vectors.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec
        mock_dependencies['multi_query_ids'].return_value = ['seed_quantized', 'close', 'far']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {cid: {'item_id': cid, 'title': cid, 'author': cid} for cid in ('seed_quantized', 'close', 'far')}
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'anchor', 'id': 7}], n_results=4, temperature=0.0
            )

        assert sorted(r['item_id'] for r in result['results']) == ['close', 'far']

    def test_drop_stored_seeds_skips_candidates_without_vectors(self):
        vectors = {'a': [1.0, 0.0], 'b': None, 'c': [0.0, 1.0]}

        kept = song_alchemy._drop_stored_seeds(['a', 'b', 'c'], [np.array([1.0, 0.0])], vectors.get)

        assert kept == ['b', 'c']

    def test_string_seed_flag_is_not_a_seed(self, mock_dependencies):
        points = song_alchemy._stored_inclusion_points(
            {'points': [{'vector': [1.0, 0.0], 'seed': 'true'}, {'vector': [0.0, 1.0], 'seed': True}]}
        )

        assert [p['seed'] for p in points] == [False, True]

    def test_stored_points_with_wrong_size_or_non_finite_values_are_skipped(self, mock_dependencies):
        points = song_alchemy._stored_inclusion_points(
            {
                'points': [
                    {'vector': [1.0, 0.0, 0.0]},
                    {'vector': [float('nan'), 1.0]},
                    {'vector': [0.0, 1.0]},
                ]
            }
        )

        assert [list(p['vector']) for p in points] == [[0.0, 1.0]]

    def test_anchor_embedding_tag_fingerprints_the_model_file(self, tmp_path):
        model = tmp_path / 'renamed_model.onnx'
        model.write_bytes(b'model-bytes')
        expected = hashlib.sha256(b'model-bytes').hexdigest()[:16]

        with patch('tasks.song_alchemy.config') as config:
            config.EMBEDDING_MODEL_PATH = str(model)
            config.EMBEDDING_DIMENSION = 200
            tag = song_alchemy.anchor_embedding_tag()
            config.EMBEDDING_MODEL_PATH = str(tmp_path / 'missing.onnx')
            fallback = song_alchemy.anchor_embedding_tag()

        assert tag == {'embedding_model_sha256': expected, 'dimension': 200}
        assert fallback == {'embedding_model_sha256': None, 'dimension': 200}

    def test_embedding_tags_match_ignores_an_unknown_fingerprint_but_never_the_dimension(self):
        current = {'embedding_model_sha256': 'abc', 'dimension': 200}
        assert song_alchemy.embedding_tags_match({'embedding_model_sha256': 'abc', 'dimension': 200}, current)
        assert song_alchemy.embedding_tags_match({'embedding_model_sha256': None, 'dimension': 200}, current)
        assert song_alchemy.embedding_tags_match(
            {'embedding_model_sha256': 'abc', 'dimension': 200},
            {'embedding_model_sha256': None, 'dimension': 200},
        )
        assert not song_alchemy.embedding_tags_match({'embedding_model_sha256': 'xyz', 'dimension': 200}, current)
        assert not song_alchemy.embedding_tags_match({'embedding_model_sha256': None, 'dimension': 128}, current)
        assert not song_alchemy.embedding_tags_match('not-a-tag', current)

    def test_saved_anchor_rerun_queries_the_same_points_as_the_saved_run(self, mock_dependencies):
        live = [
            {'vector': np.array([1.0, float(i)]), 'weight': 1.0, 'source_type': 'song', 'source_id': f's{i}', 'seed': True}
            for i in range(10)
        ]
        component_weights = [0.2, 0.15, 0.12, 0.11, 0.1, 0.09, 0.08, 0.08, 0.07]
        for name, base in (('a1', 100.0), ('a2', 200.0)):
            live.extend(
                {'vector': np.array([base, float(j)]), 'weight': w, 'source_type': 'artist', 'source_id': name}
                for j, w in enumerate(component_weights)
            )
        live_selected = song_alchemy._select_query_points(live, 16)
        anchor = {
            'id': 7,
            'name': 'Saved',
            'centroid': [0.0, 0.0],
            'exclusions': None,
            'inclusions': {
                'embedding_model_sha256': 'musicnn_embedding.onnx',
                'dimension': 2,
                'points': song_alchemy._export_inclusions(live, {}),
            },
        }

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            loaded = song_alchemy._anchor_anchor_points(7)
        rerun_selected = song_alchemy._select_query_points(loaded, 16)

        assert {p['source_id'] for p in live_selected if p['source_type'] == 'artist'} == {'a1', 'a2'}
        assert [list(p['vector']) for p in rerun_selected] == [list(p['vector']) for p in live_selected]

    def test_stored_seed_signature_drops_other_copies_of_the_seed(self, mock_dependencies):
        anchor = self._tagged_anchor(
            [{'vector': [1.0, 0.0], 'weight': 1.0, 'seed': True, 'signature': ['seed title', 'seed artist']}]
        )
        mock_dependencies['get_vector_by_id'].side_effect = lambda x: {
            'remaster': [0.7, 0.7],
            'other': [0.9, 0.1],
        }.get(x)
        mock_dependencies['multi_query_ids'].return_value = ['remaster', 'other']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                'remaster': {'item_id': 'remaster', 'title': ' Seed Title ', 'author': 'SEED ARTIST'},
                'other': {'item_id': 'other', 'title': 'Other', 'author': 'Someone'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(add_items=[{'type': 'anchor', 'id': 7}], temperature=0.0)

        assert [r['item_id'] for r in result['results']] == ['other']

    def test_song_seed_signature_is_exported_for_a_saved_anchor(self, mock_dependencies):
        mock_dependencies['get_vector_by_id'].side_effect = lambda x: {'s1': [1.0, 0.0], 'r1': [0.9, 0.1]}.get(x)
        mock_dependencies['multi_query_ids'].return_value = ['r1']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                's1': {'item_id': 's1', 'title': 'My Song ', 'author': 'My Band'},
                'r1': {'item_id': 'r1', 'title': 'R1', 'author': 'A1'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        result = song_alchemy.song_alchemy(add_items=[{'type': 'song', 'id': 's1'}], temperature=0.5)

        assert result['inclusions'][0]['signature'] == ['my song', 'my band']

    def test_subtracted_anchor_seed_songs_are_dropped_even_with_a_zero_radius(self, mock_dependencies):
        anchor = self._tagged_anchor([{'vector': [0.6, 0.8], 'weight': 1.0, 'seed': True}], anchor_id=8)
        anchor['exclusions'] = None
        mock_dependencies['get_vector_by_id'].side_effect = lambda x: {
            's1': [1.0, 0.0],
            'seed_copy': [0.6, 0.8],
            'keep': [0.9, 0.1],
        }.get(x)
        mock_dependencies['multi_query_ids'].return_value = ['seed_copy', 'keep']
        mock_dependencies['get_score_data_by_ids'].side_effect = _score_side_effect(
            {
                'seed_copy': {'item_id': 'seed_copy', 'title': 'Copy', 'author': 'A1'},
                'keep': {'item_id': 'keep', 'title': 'Keep', 'author': 'A2'},
            }
        )
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('database.get_alchemy_anchor_by_id', return_value=anchor):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'song', 'id': 's1'}],
                subtract_items=[{'type': 'anchor', 'id': 8}],
                subtract_distance=0.0,
                temperature=0.5,
            )

        assert [r['item_id'] for r in result['results']] == ['keep']
        assert [r['item_id'] for r in result['filtered_out']] == []

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
