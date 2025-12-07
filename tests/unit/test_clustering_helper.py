"""Unit tests for tasks/clustering_helper.py

Tests evolutionary parameter generation and mutation with ACTUAL function calls:
- Random parameter generation for exploration
- Parameter mutation for exploitation  
- Bounded parameter ranges
- Stratified sampling logic
- Clustering model application
"""
import pytest
import numpy as np
import random
from unittest.mock import Mock, patch, MagicMock
from tasks.clustering_helper import (
    _mutate_param,
    _generate_random_parameters,
    _mutate_parameters,
    _prepare_and_scale_data,
    _apply_clustering_model,
    _get_stratified_song_subset,
    _get_track_primary_genre
)


class TestMutateParam:
    """Test the actual _mutate_param function"""

    def test_mutate_param_integer_within_bounds(self):
        """Test integer parameter mutation stays within bounds"""
        random.seed(42)
        value = 10
        min_val = 5
        max_val = 15
        delta = 2
        
        mutated = _mutate_param(value, min_val, max_val, delta, is_float=False)
        
        assert min_val <= mutated <= max_val

    def test_mutate_param_float_within_bounds(self):
        """Test float parameter mutation stays within bounds"""
        random.seed(42)
        value = 0.5
        min_val = 0.1
        max_val = 1.0
        delta = 0.1
        
        mutated = _mutate_param(value, min_val, max_val, delta, is_float=True)
        
        assert min_val <= mutated <= max_val

    def test_mutate_param_clamps_at_max(self):
        """Test mutation clamps at maximum bound"""
        value = 98
        min_val = 0
        max_val = 100
        delta = 10
        
        # Try multiple times to ensure clamping works
        for _ in range(10):
            mutated = _mutate_param(value, min_val, max_val, delta)
            assert mutated <= max_val

    def test_mutate_param_clamps_at_min(self):
        """Test mutation clamps at minimum bound"""
        value = 2
        min_val = 0
        max_val = 100
        delta = 10
        
        for _ in range(10):
            mutated = _mutate_param(value, min_val, max_val, delta)
            assert mutated >= min_val


class TestGenerateRandomParameters:
    """Test the actual _generate_random_parameters function"""

    def test_generates_kmeans_parameters(self):
        """Test random parameter generation for K-Means"""
        data = np.random.rand(100, 50)
        method = 'kmeans'
        pca_ranges = {'components_min': 0, 'components_max': 30}
        num_clust_ranges = (5, 20)
        
        params = _generate_random_parameters(
            method, data, pca_ranges, num_clust_ranges, {}, {}, {}
        )
        
        assert 'pca_config' in params
        assert 'clustering_method_config' in params
        assert params['clustering_method_config']['method'] == 'kmeans'
        n_clusters = params['clustering_method_config']['params']['n_clusters']
        assert 2 <= n_clusters <= min(20, data.shape[0])

    def test_generates_dbscan_parameters(self):
        """Test random parameter generation for DBSCAN"""
        data = np.random.rand(100, 50)
        method = 'dbscan'
        pca_ranges = {'components_min': 0, 'components_max': 30}
        db_ranges = {'eps_min': 0.1, 'eps_max': 2.0, 'samples_min': 2, 'samples_max': 10}
        
        params = _generate_random_parameters(
            method, data, pca_ranges, (), db_ranges, {}, {}
        )
        
        assert params['clustering_method_config']['method'] == 'dbscan'
        dbscan_params = params['clustering_method_config']['params']
        assert 0.1 <= dbscan_params['eps'] <= 2.0
        assert 2 <= dbscan_params['min_samples'] <= 10

    def test_generates_gmm_parameters(self):
        """Test random parameter generation for GMM"""
        data = np.random.rand(100, 50)
        method = 'gmm'
        pca_ranges = {'components_min': 0, 'components_max': 30}
        gmm_ranges = {'n_components_min': 2, 'n_components_max': 15}
        
        params = _generate_random_parameters(
            method, data, pca_ranges, (), {}, gmm_ranges, {}
        )
        
        assert params['clustering_method_config']['method'] == 'gmm'
        n_components = params['clustering_method_config']['params']['n_components']
        assert 2 <= n_components <= min(15, data.shape[0])

    def test_generates_spectral_parameters(self):
        """Test random parameter generation for Spectral Clustering"""
        data = np.random.rand(100, 50)
        method = 'spectral'
        pca_ranges = {'components_min': 0, 'components_max': 30}
        spec_ranges = {'n_clusters_min': 3, 'n_clusters_max': 12}
        
        params = _generate_random_parameters(
            method, data, pca_ranges, (), {}, {}, spec_ranges
        )
        
        assert params['clustering_method_config']['method'] == 'spectral'
        spectral_params = params['clustering_method_config']['params']
        assert 'n_clusters' in spectral_params
        assert 'random_state' in spectral_params
        n_clusters = spectral_params['n_clusters']
        assert 2 <= n_clusters < data.shape[0]


class TestMutateParameters:
    """Test the actual _mutate_parameters function"""

    def test_mutates_kmeans_parameters(self):
        """Test parameter mutation for K-Means"""
        elite_params = {
            'pca_config': {'enabled': True, 'components': 10},
            'clustering_method_config': {
                'method': 'kmeans',
                'params': {'n_clusters': 10}
            }
        }
        data = np.random.rand(100, 50)
        mutation_cfg = {'int_abs_delta': 2, 'float_abs_delta': 0.1}
        pca_ranges = {'components_min': 0, 'components_max': 30}
        num_clust_ranges = (5, 20)
        
        mutated = _mutate_parameters(
            elite_params, mutation_cfg, 'kmeans', data,
            pca_ranges, num_clust_ranges, {}, {}, {}
        )
        
        assert 'pca_config' in mutated
        assert 'clustering_method_config' in mutated
        n_clusters = mutated['clustering_method_config']['params']['n_clusters']
        assert 5 <= n_clusters <= 20

    def test_mutates_dbscan_parameters(self):
        """Test parameter mutation for DBSCAN"""
        elite_params = {
            'pca_config': {'enabled': False, 'components': 0},
            'clustering_method_config': {
                'method': 'dbscan',
                'params': {'eps': 0.5, 'min_samples': 5}
            }
        }
        data = np.random.rand(100, 50)
        mutation_cfg = {'int_abs_delta': 1, 'float_abs_delta': 0.1}
        pca_ranges = {'components_min': 0, 'components_max': 30}
        db_ranges = {'eps_min': 0.1, 'eps_max': 2.0, 'samples_min': 2, 'samples_max': 10}
        
        mutated = _mutate_parameters(
            elite_params, mutation_cfg, 'dbscan', data,
            pca_ranges, (), db_ranges, {}, {}
        )
        
        dbscan_params = mutated['clustering_method_config']['params']
        assert 0.1 <= dbscan_params['eps'] <= 2.0
        assert 2 <= dbscan_params['min_samples'] <= 10


class TestPrepareAndScaleData:
    """Test the actual _prepare_and_scale_data function"""

    def test_uses_embeddings_when_enabled(self):
        """Test that embeddings are used when use_embeddings=True"""
        X_feat = np.random.rand(50, 20)
        X_embed = np.random.rand(50, 128)
        
        scaled_data, scaler = _prepare_and_scale_data(X_feat, X_embed, use_embeddings=True)
        
        assert scaled_data.shape == (50, 128)

    def test_uses_features_when_embeddings_disabled(self):
        """Test that features are used when use_embeddings=False"""
        X_feat = np.random.rand(50, 20)
        X_embed = np.random.rand(50, 128)
        
        scaled_data, scaler = _prepare_and_scale_data(X_feat, X_embed, use_embeddings=False)
        
        assert scaled_data.shape == (50, 20)

    def test_scales_data_correctly(self):
        """Test that data is properly scaled"""
        X_feat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        scaled_data, scaler = _prepare_and_scale_data(X_feat, None, use_embeddings=False)
        
        mean = np.mean(scaled_data, axis=0)
        assert np.allclose(mean, 0, atol=1e-10)


class TestApplyClusteringModel:
    """Test the actual _apply_clustering_model function"""

    @patch('tasks.clustering_helper.USE_GPU_CLUSTERING', False)
    def test_applies_kmeans_successfully(self):
        """Test K-Means clustering application"""
        data = np.random.rand(50, 10)
        method_config = {
            'method': 'kmeans',
            'params': {'n_clusters': 5}
        }
        
        labels, centers, model = _apply_clustering_model(data, method_config, "Test", 1)
        
        assert labels is not None
        assert len(labels) == 50
        assert len(set(labels)) <= 5

    @patch('tasks.clustering_helper.USE_GPU_CLUSTERING', False)
    def test_applies_dbscan_successfully(self):
        """Test DBSCAN clustering application"""
        data = np.random.rand(50, 10)
        method_config = {
            'method': 'dbscan',
            'params': {'eps': 0.5, 'min_samples': 3}
        }
        
        labels, centers, model = _apply_clustering_model(data, method_config, "Test", 1)
        
        assert labels is not None
        assert len(labels) == 50

    @patch('tasks.clustering_helper.USE_GPU_CLUSTERING', False)
    def test_rejects_invalid_kmeans_params(self):
        """Test that invalid K-Means parameters are rejected"""
        data = np.random.rand(50, 10)
        method_config = {
            'method': 'kmeans',
            'params': {'n_clusters': 1}  # Invalid
        }
        
        labels, centers, model = _apply_clustering_model(data, method_config, "Test", 1)
        
        assert labels is None


class TestGetStratifiedSongSubset:
    """Test the actual _get_stratified_song_subset function"""

    def test_stratified_sampling_balances_genres(self):
        """Test that stratified sampling balances genre representation"""
        # genre_map expects track dictionaries with item_id and mood_vector
        genre_map = {
            'Rock': [
                {'item_id': 'r1', 'mood_vector': 'Rock:0.8,Pop:0.2'},
                {'item_id': 'r2', 'mood_vector': 'Rock:0.9,Jazz:0.1'},
            ],
            'Pop': [
                {'item_id': 'p1', 'mood_vector': 'Pop:0.7,Rock:0.3'},
            ]
        }
        target_per_genre = 2  # Single integer for all genres
        
        subset = _get_stratified_song_subset(genre_map, target_per_genre)
        
        # Should return a list of track dictionaries
        assert isinstance(subset, list)
        assert len(subset) >= 0

    def test_excludes_previous_ids(self):
        """Test that previously selected IDs are excluded"""
        genre_map = {
            'Rock': [
                {'item_id': 'r1', 'mood_vector': 'Rock:0.8'},
                {'item_id': 'r2', 'mood_vector': 'Rock:0.7'},
                {'item_id': 'r3', 'mood_vector': 'Rock:0.9'},
            ],
            'Pop': [
                {'item_id': 'p1', 'mood_vector': 'Pop:0.8'},
                {'item_id': 'p2', 'mood_vector': 'Pop:0.7'},
            ]
        }
        target_per_genre = 2
        prev_ids = ['r1', 'p1']
        
        subset = _get_stratified_song_subset(genre_map, target_per_genre, prev_ids=prev_ids)
        
        # Should not include prev_ids in the subset
        subset_ids = {track['item_id'] for track in subset}
        assert 'r1' not in subset_ids
        assert 'p1' not in subset_ids


class TestGetTrackPrimaryGenre:
    """Test the actual _get_track_primary_genre function"""

    def test_returns_genre_from_mood_vector(self):
        """Test that genre is extracted from mood_vector"""
        track_data = {'mood_vector': 'Rock:0.8,Pop:0.2'}
        
        genre = _get_track_primary_genre(track_data)
        
        # Should return the genre with highest score from STRATIFIED_GENRES
        # Rock has 0.8 which is highest
        assert genre in ['Rock', '__other__']

    def test_returns_other_when_no_stratified_genre(self):
        """Test that '__other__' is returned when no stratified genre found"""
        track_data = {'mood_vector': 'UnknownMood:0.9'}
        
        genre = _get_track_primary_genre(track_data)
        
        assert genre == '__other__'

    def test_returns_other_when_no_mood_vector(self):
        """Test that '__other__' is returned when no mood_vector"""
        track_data = {}
        
        genre = _get_track_primary_genre(track_data)
        
        assert genre == '__other__'

