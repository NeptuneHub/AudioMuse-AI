import pytest
from unittest.mock import patch
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tasks import song_alchemy

class TestSongAlchemy:

    @pytest.fixture
    def mock_dependencies(self):
        with patch('tasks.song_alchemy.get_vector_by_id') as mock_get_vec, \
             patch('tasks.song_alchemy.multi_query_ids') as mock_multi_query, \
             patch('tasks.song_alchemy.find_nearest_neighbors_by_id') as mock_find_nn_id, \
             patch('tasks.song_alchemy.get_score_data_by_ids') as mock_get_score, \
             patch('tasks.song_alchemy.load_map_projection') as mock_load_proj, \
             patch('tasks.song_alchemy._get_artist_gmm_vectors_and_weights') as mock_get_gmm, \
             patch('tasks.song_alchemy.config') as mock_config:
            
            # Setup default config values
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
                'load_map_projection': mock_load_proj,
                'get_artist_gmm': mock_get_gmm,
                'config': mock_config
            }

    def test_compute_centroid_from_items_songs(self, mock_dependencies):
        mock_dependencies['get_vector_by_id'].side_effect = lambda x: [1.0, 0.0] if x == 's1' else [0.0, 1.0]
        
        items = [{'type': 'song', 'id': 's1'}, {'type': 'song', 'id': 's2'}]
        centroid = song_alchemy._compute_centroid_from_items(items)
        
        assert np.allclose(centroid, [0.5, 0.5])

    def test_compute_centroid_from_items_artist(self, mock_dependencies):
        # Artist with 2 components
        mock_dependencies['get_artist_gmm'].return_value = (
            [np.array([1.0, 0.0]), np.array([3.0, 0.0])], # vectors
            [0.5, 0.5] # weights
        )
        
        items = [{'type': 'artist', 'id': 'a1'}]
        centroid = song_alchemy._compute_centroid_from_items(items)
        
        # Weighted mean: (1.0*0.5 + 3.0*0.5) / (0.5+0.5) = 2.0
        assert np.allclose(centroid, [2.0, 0.0])

    def test_song_alchemy_basic_flow(self, mock_dependencies):
        # Setup mocks
        mock_dependencies['get_vector_by_id'].return_value = [1.0, 0.0]
        mock_dependencies['multi_query_ids'].return_value = ['r1', 'r2']
        mock_dependencies['get_score_data_by_ids'].return_value = [
            {'item_id': 'r1', 'title': 'Result 1', 'author': 'Author 1'},
            {'item_id': 'r2', 'title': 'Result 2', 'author': 'Author 2'}
        ]
        mock_dependencies['load_map_projection'].return_value = (None, None) # Force local projection
        
        result = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}],
            n_results=5
        )
        
        assert len(result['results']) == 2
        assert result['results'][0]['item_id'] in ['r1', 'r2']
        # Check that projection was attempted (defaults to pca or none if few points)
        assert 'projection' in result

    def test_song_alchemy_subtraction(self, mock_dependencies):
        # s1 is [1, 0], s2 (subtract) is [0, 1]
        # r1 is [0.9, 0.1] (close to s1), r2 is [0.1, 0.9] (close to s2)
        
        def get_vec(id):
            vectors = {
                's1': [1.0, 0.0],
                'sub1': [0.0, 1.0],
                'r1': [0.9, 0.1],
                'r2': [0.1, 0.9]
            }
            return vectors.get(id)
            
        mock_dependencies['get_vector_by_id'].side_effect = get_vec

        mock_dependencies['multi_query_ids'].return_value = ['r1', 'r2']

        mock_dependencies['get_score_data_by_ids'].return_value = [
            {'item_id': 'r1', 'title': 'R1'},
            {'item_id': 'r2', 'title': 'R2'}
        ]
        mock_dependencies['load_map_projection'].return_value = (None, None)

        # Set subtract distance threshold high enough to filter r2
        # Distance(sub1, r2) = sqrt(0.1^2 + 0.1^2) = sqrt(0.02) ~= 0.14
        # Distance(sub1, r1) = sqrt(0.9^2 + 0.9^2) = sqrt(1.62) ~= 1.27
        # If threshold is 0.5, r2 should be filtered out (dist < 0.5), r1 kept (dist > 0.5)
        
        result = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}],
            subtract_items=[{'type': 'song', 'id': 'sub1'}],
            subtract_distance=0.5
        )
        
        # r2 is close to subtract centroid, so it should be filtered out
        result_ids = [r['item_id'] for r in result['results']]
        filtered_ids = [r['item_id'] for r in result['filtered_out']]
        
        assert 'r1' in result_ids
        assert 'r2' in filtered_ids

    def test_project_to_2d(self):
        vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        proj = song_alchemy._project_to_2d(vectors)
        assert len(proj) == 3
        assert len(proj[0]) == 2
        # Check values are within [-1, 1]
        for p in proj:
            assert -1.0 <= p[0] <= 1.0
            assert -1.0 <= p[1] <= 1.0

    def test_temperature_sampling(self, mock_dependencies):
        # Mock neighbors
        mock_dependencies['get_vector_by_id'].return_value = [1.0, 0.0]
        
        mock_dependencies['multi_query_ids'].return_value = ['r1', 'r2']
        mock_dependencies['find_nearest_neighbors_by_id'].return_value = [
            {'item_id': 'r1', 'score': 0.1},
            {'item_id': 'r2', 'score': 0.2}
        ]
        mock_dependencies['get_score_data_by_ids'].return_value = [
            {'item_id': 'r1'}, {'item_id': 'r2'}
        ]
        mock_dependencies['load_map_projection'].return_value = (None, None)

        # Test deterministic (temp=0)
        result_zero = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}],
            temperature=0.0
        )
        assert len(result_zero['results']) > 0

        # Test high temperature
        result_high = song_alchemy.song_alchemy(
            add_items=[{'type': 'song', 'id': 's1'}],
            temperature=10.0
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
        with patch('tasks.mediaserver.get_playlist_track_ids', return_value=['t0', 't1']), \
             patch('tasks.ivf_manager.get_cell_groups_for_items', return_value=groups):
            vecs, weights = song_alchemy._get_playlist_components('pl1')

        assert len(vecs) == 2
        assert np.allclose(weights, [0.6, 0.4])

    def test_get_playlist_components_caps_clusters_at_max(self, mock_dependencies):
        groups = [(np.array([float(i), 0.0]), 1) for i in range(40)]
        with patch('tasks.mediaserver.get_playlist_track_ids', return_value=[f't{i}' for i in range(40)]), \
             patch('tasks.ivf_manager.get_cell_groups_for_items', return_value=groups):
            vecs, weights = song_alchemy._get_playlist_components('pl1')

        assert len(vecs) == 10
        assert np.isclose(sum(weights), 1.0)

    def test_get_playlist_components_coherent_returns_single(self, mock_dependencies):
        groups = [(np.array([1.0, 0.0]), 50)]
        with patch('tasks.mediaserver.get_playlist_track_ids', return_value=['t0']), \
             patch('tasks.ivf_manager.get_cell_groups_for_items', return_value=groups):
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

        with patch('tasks.mediaserver.get_playlist_track_ids', return_value=track_ids), \
             patch('tasks.ivf_manager.get_cell_groups_for_items', side_effect=fake_groups):
            vecs, weights = song_alchemy._get_playlist_components('pl1')

        assert captured['n'] == 50
        assert np.isclose(sum(weights), 1.0)

    def test_get_playlist_components_no_index_match(self, mock_dependencies):
        with patch('tasks.mediaserver.get_playlist_track_ids', return_value=['t0', 't1']), \
             patch('tasks.ivf_manager.get_cell_groups_for_items', return_value=[]):
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
        with patch('tasks.song_alchemy._get_playlist_components',
                   return_value=([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [0.6, 0.4])):
            points = song_alchemy._gather_anchor_points([{'type': 'playlist', 'id': 'pl1'}])

        assert len(points) == 2
        assert all(p['source_type'] == 'playlist' for p in points)
        assert [p['comp_idx'] for p in points] == [0, 1]

    def test_song_alchemy_playlist_matches_any_cluster(self, mock_dependencies):
        def get_vec(id):
            return {
                'cand_a': [1.0, 0.0],
                'cand_b': [0.0, 1.0],
                'mid': [0.5, 0.5],
            }.get(id)

        mock_dependencies['get_vector_by_id'].side_effect = get_vec
        mock_dependencies['multi_query_ids'].return_value = ['cand_a', 'cand_b', 'mid']
        mock_dependencies['get_score_data_by_ids'].return_value = [
            {'item_id': 'cand_a'}, {'item_id': 'cand_b'}, {'item_id': 'mid'}
        ]
        mock_dependencies['load_map_projection'].return_value = (None, None)

        with patch('tasks.song_alchemy._get_playlist_components',
                   return_value=([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [0.5, 0.5])):
            result = song_alchemy.song_alchemy(
                add_items=[{'type': 'playlist', 'id': 'pl1'}],
                temperature=0.0
            )

        ids = [r['item_id'] for r in result['results']]
        assert 'cand_a' in ids and 'cand_b' in ids
        assert ids.index('mid') == len(ids) - 1
        assert result['results'][0]['distance'] == 0.0
