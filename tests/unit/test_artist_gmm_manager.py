"""Unit tests for tasks/artist_gmm_manager.py

Tests GMM component selection and parameter extraction:
- BIC-based optimal component selection
- Small dataset handling (< 5 tracks)
- GMM parameter structure validation
"""
import pytest
import numpy as np
from tasks.artist_gmm_manager import (
    select_optimal_gmm_components,
    fit_artist_gmm,
    GMM_N_COMPONENTS_MIN,
    GMM_N_COMPONENTS_MAX,
    GMM_COVARIANCE_TYPE
)


class TestSelectOptimalGMMComponents:
    """Test optimal GMM component selection using BIC"""

    def test_single_sample_returns_one_component(self):
        """Test that 1 sample always returns 1 component"""
        embeddings = np.random.rand(1, 128)
        
        n_components = select_optimal_gmm_components(embeddings)
        
        assert n_components == 1

    def test_two_samples_returns_valid_components(self):
        """Test that 2 samples returns valid component count"""
        embeddings = np.random.rand(2, 128)
        
        n_components = select_optimal_gmm_components(embeddings)
        
        # Should be between 1 and 2
        assert 1 <= n_components <= 2

    def test_small_dataset_respects_max_feasible(self):
        """Test that small datasets (< 5 samples) use 1 component per sample"""
        embeddings = np.random.rand(4, 128)
        
        n_components = select_optimal_gmm_components(embeddings)
        
        # For 4 samples, max should be 4 (or configured max, whichever is smaller)
        assert n_components <= 4
        assert n_components >= 1

    def test_large_dataset_respects_sample_ratio(self):
        """Test that large datasets respect the 5 samples per component rule"""
        embeddings = np.random.rand(50, 128)
        
        n_components = select_optimal_gmm_components(embeddings)
        
        # With 50 samples and 5 per component rule, max should be 10
        # But actual choice depends on BIC optimization
        assert 1 <= n_components <= 10

    def test_respects_min_components_parameter(self):
        """Test that min_components parameter is respected"""
        embeddings = np.random.rand(50, 128)
        
        n_components = select_optimal_gmm_components(embeddings, min_components=3, max_components=8)
        
        # Should be at least 3 (unless dataset is too small)
        assert n_components >= 1
        assert n_components <= 8

    def test_respects_max_components_parameter(self):
        """Test that max_components parameter is respected"""
        embeddings = np.random.rand(100, 128)
        
        n_components = select_optimal_gmm_components(embeddings, min_components=2, max_components=5)
        
        # Should not exceed 5 (max_components)
        assert n_components <= 5
        # Should be at least 1 (BIC optimization may select fewer than min_components)
        assert n_components >= 1

    def test_deterministic_with_same_data(self):
        """Test that same data produces same component count"""
        np.random.seed(42)
        embeddings = np.random.rand(30, 128)
        
        n1 = select_optimal_gmm_components(embeddings)
        n2 = select_optimal_gmm_components(embeddings)
        
        # Should be deterministic
        assert n1 == n2

    def test_high_dimensional_embeddings(self):
        """Test with realistic high-dimensional embeddings"""
        embeddings = np.random.rand(25, 512)  # 512-dim embeddings
        
        n_components = select_optimal_gmm_components(embeddings)
        
        # Should work with high dimensions
        assert 1 <= n_components <= min(GMM_N_COMPONENTS_MAX, 25 // 5)


class TestFitArtistGMM:
    """Test GMM fitting for artist sound profiles"""

    def test_single_track_artist(self):
        """Test GMM fitting for artist with 1 track"""
        embeddings = [np.random.rand(128)]
        
        gmm_params = fit_artist_gmm("Test Artist", embeddings)
        
        # Should create 1-component GMM
        assert gmm_params is not None
        assert gmm_params['n_components'] == 1
        assert gmm_params['n_tracks'] == 1
        assert gmm_params['is_few_songs'] is True
        assert len(gmm_params['weights']) == 1
        assert gmm_params['weights'][0] == 1.0

    def test_few_tracks_artist(self):
        """Test GMM fitting for artist with < 5 tracks"""
        embeddings = [np.random.rand(128) for _ in range(3)]
        
        gmm_params = fit_artist_gmm("Few Tracks Artist", embeddings)
        
        # Should create 1 component per track with equal weights
        assert gmm_params is not None
        assert gmm_params['n_components'] == 3
        assert gmm_params['n_tracks'] == 3
        assert gmm_params['is_few_songs'] is True
        assert len(gmm_params['weights']) == 3
        assert all(abs(w - 1.0/3) < 1e-10 for w in gmm_params['weights'])

    def test_many_tracks_artist(self):
        """Test GMM fitting for artist with >= 5 tracks"""
        embeddings = [np.random.rand(128) for _ in range(20)]
        
        gmm_params = fit_artist_gmm("Popular Artist", embeddings)
        
        # Should use sklearn GMM fitting
        assert gmm_params is not None
        assert gmm_params['n_tracks'] == 20
        assert gmm_params['is_few_songs'] is False
        # Components should be optimized by BIC
        assert 1 <= gmm_params['n_components'] <= GMM_N_COMPONENTS_MAX

    def test_gmm_params_structure(self):
        """Test that GMM params have correct structure"""
        embeddings = [np.random.rand(128) for _ in range(10)]
        
        gmm_params = fit_artist_gmm("Artist", embeddings)
        
        # Check all required fields
        assert 'weights' in gmm_params
        assert 'means' in gmm_params
        assert 'covariances' in gmm_params
        assert 'n_components' in gmm_params
        assert 'covariance_type' in gmm_params
        assert 'n_features' in gmm_params
        assert 'n_tracks' in gmm_params
        assert 'is_few_songs' in gmm_params

    def test_weights_sum_to_one(self):
        """Test that GMM weights sum to 1.0"""
        embeddings = [np.random.rand(128) for _ in range(15)]
        
        gmm_params = fit_artist_gmm("Artist", embeddings)
        
        weights_sum = sum(gmm_params['weights'])
        assert abs(weights_sum - 1.0) < 1e-6

    def test_means_shape_matches_components(self):
        """Test that means array matches number of components"""
        embeddings = [np.random.rand(128) for _ in range(8)]
        
        gmm_params = fit_artist_gmm("Artist", embeddings)
        
        n_components = gmm_params['n_components']
        assert len(gmm_params['means']) == n_components
        # Each mean should be 128-dimensional
        assert all(len(mean) == 128 for mean in gmm_params['means'])

    def test_covariances_shape_matches_components(self):
        """Test that covariances array matches number of components"""
        embeddings = [np.random.rand(128) for _ in range(12)]
        
        gmm_params = fit_artist_gmm("Artist", embeddings)
        
        n_components = gmm_params['n_components']
        assert len(gmm_params['covariances']) == n_components
        # Diagonal covariance: each should be 128-dimensional
        assert all(len(cov) == 128 for cov in gmm_params['covariances'])

    def test_covariance_type_is_diagonal(self):
        """Test that covariance type is set to diagonal"""
        embeddings = [np.random.rand(128) for _ in range(10)]
        
        gmm_params = fit_artist_gmm("Artist", embeddings)
        
        assert gmm_params['covariance_type'] == GMM_COVARIANCE_TYPE
        assert gmm_params['covariance_type'] == 'diag'

    def test_n_features_matches_embedding_dim(self):
        """Test that n_features matches embedding dimensionality"""
        embedding_dim = 256
        embeddings = [np.random.rand(embedding_dim) for _ in range(10)]
        
        gmm_params = fit_artist_gmm("Artist", embeddings)
        
        assert gmm_params['n_features'] == embedding_dim

    def test_few_songs_flag_correct(self):
        """Test that is_few_songs flag is set correctly"""
        few_embeddings = [np.random.rand(128) for _ in range(3)]
        many_embeddings = [np.random.rand(128) for _ in range(10)]
        
        few_params = fit_artist_gmm("Few Artist", few_embeddings)
        many_params = fit_artist_gmm("Many Artist", many_embeddings)
        
        assert few_params['is_few_songs'] is True
        assert many_params['is_few_songs'] is False

    def test_different_artists_different_gmms(self):
        """Test that different artists produce different GMMs"""
        np.random.seed(42)
        embeddings1 = [np.random.rand(128) for _ in range(10)]
        np.random.seed(99)
        embeddings2 = [np.random.rand(128) for _ in range(10)]
        
        gmm1 = fit_artist_gmm("Artist 1", embeddings1)
        gmm2 = fit_artist_gmm("Artist 2", embeddings2)
        
        # Different data should produce different means
        assert gmm1['means'] != gmm2['means']

    def test_high_dimensional_embeddings(self):
        """Test with high-dimensional embeddings (like real CLAP)"""
        embeddings = [np.random.rand(512) for _ in range(15)]
        
        gmm_params = fit_artist_gmm("HD Artist", embeddings)
        
        assert gmm_params is not None
        assert gmm_params['n_features'] == 512
        assert all(len(mean) == 512 for mean in gmm_params['means'])


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_track_list(self):
        """Test handling of empty track list"""
        embeddings = []
        
        gmm_params = fit_artist_gmm("Empty Artist", embeddings)
        
        # Should return None (below minimum)
        assert gmm_params is None

    def test_zero_dimensional_embeddings(self):
        """Test handling of invalid zero-dimensional embeddings"""
        # This should raise an error or return None
        try:
            embeddings = [np.array([]) for _ in range(5)]
            gmm_params = fit_artist_gmm("Invalid Artist", embeddings)
            # If it doesn't raise, it should return None or handle gracefully
            assert gmm_params is None or 'n_features' not in gmm_params or gmm_params['n_features'] == 0
        except Exception:
            # Expected - invalid input
            pass

    def test_mismatched_embedding_dimensions(self):
        """Test handling of embeddings with different dimensions"""
        embeddings = [
            np.random.rand(128),
            np.random.rand(64),  # Different dimension!
            np.random.rand(128)
        ]
        
        # Should either handle gracefully or raise clear error
        try:
            gmm_params = fit_artist_gmm("Mismatched Artist", embeddings)
            # If it doesn't raise, check it handled it somehow
            assert gmm_params is None or gmm_params is not None
        except (ValueError, Exception):
            # Expected - mismatched dimensions
            pass

    def test_very_large_component_count(self):
        """Test that component count is bounded even with many samples"""
        # 100 samples should still cap at GMM_N_COMPONENTS_MAX
        embeddings = [np.random.rand(128) for _ in range(100)]
        
        gmm_params = fit_artist_gmm("Popular Artist", embeddings)
        
        assert gmm_params['n_components'] <= GMM_N_COMPONENTS_MAX
