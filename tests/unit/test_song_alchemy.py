import pytest
from unittest.mock import MagicMock, patch, ANY
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
             patch('tasks.song_alchemy.find_nearest_neighbors_by_vector') as mock_find_nn_vec, \
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
            
            yield {
                'get_vector_by_id': mock_get_vec,
                'find_nearest_neighbors_by_vector': mock_find_nn_vec,
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
        mock_dependencies['find_nearest_neighbors_by_vector'].return_value = [
            {'item_id': 'r1', 'score': 0.1},
            {'item_id': 'r2', 'score': 0.2}
        ]
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
        
        # Mock neighbors returning both
        mock_dependencies['find_nearest_neighbors_by_vector'].return_value = [
            {'item_id': 'r1', 'score': 0.1},
            {'item_id': 'r2', 'score': 0.2}
        ]
        
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
        
        neighbors_data = [
            {'item_id': 'r1', 'score': 0.1},
            {'item_id': 'r2', 'score': 0.2}
        ]
        mock_dependencies['find_nearest_neighbors_by_vector'].return_value = neighbors_data
        mock_dependencies['find_nearest_neighbors_by_id'].return_value = neighbors_data
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
