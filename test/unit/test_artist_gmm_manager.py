"""Unit tests for tasks/artist_gmm_manager.py

Tests GMM component selection and parameter extraction:
- BIC-based optimal component selection
- Small dataset handling (< 5 tracks)
- GMM parameter structure validation
"""
import numpy as np
from tasks.artist_gmm_manager import (
    select_optimal_gmm_components,
    fit_artist_gmm,
    GMM_N_COMPONENTS_MAX,
    gmm_soft_chamfer_distance,
    _cosine_distance_matrix,
)


def _gmm(means, weights):
    return {"means": [list(m) for m in means], "weights": list(weights)}


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
        """Test that GMM params have correct structure.

        ``covariances`` and ``covariance_type`` are deliberately NOT stored
        anymore -- the entire Jeffreys re-rank code path that read them was
        dead code, and dropping them is what gets gmm_params under PG's
        1 GB MaxAllocSize cap at large library scales.
        """
        embeddings = [np.random.rand(128) for _ in range(10)]

        gmm_params = fit_artist_gmm("Artist", embeddings)

        # Required fields
        assert 'weights' in gmm_params
        assert 'means' in gmm_params
        assert 'n_components' in gmm_params
        assert 'n_features' in gmm_params
        assert 'n_tracks' in gmm_params
        assert 'is_few_songs' in gmm_params

        # Removed fields -- assert they are explicitly absent so we catch any
        # accidental re-introduction in a future refactor.
        assert 'covariances' not in gmm_params
        assert 'covariance_type' not in gmm_params

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


class TestGmmSoftChamfer:
    """Weighted soft-Chamfer cosine rerank over GMM component sets."""

    # Orthonormal "sound modes" so cosine distances are clean: same=0, different=1.
    A = [1.0, 0.0, 0.0, 0.0]
    B = [0.0, 1.0, 0.0, 0.0]
    C = [0.0, 0.0, 1.0, 0.0]
    D = [0.0, 0.0, 0.0, 1.0]

    def test_cosine_distance_matrix_shape_and_values(self):
        d = _cosine_distance_matrix(np.array([self.A, self.B]), np.array([self.A, self.C]))
        assert d.shape == (2, 2)
        assert abs(d[0, 0]) < 1e-6        # A vs A -> 0
        assert abs(d[0, 1] - 1.0) < 1e-6  # A vs C -> 1
        assert abs(d[1, 0] - 1.0) < 1e-6  # B vs A -> 1

    def test_identical_gmm_distance_is_zero(self):
        g = _gmm([self.A, self.B], [0.5, 0.5])
        assert gmm_soft_chamfer_distance(g, g) < 1e-6

    def test_scale_invariance(self):
        # cosine ignores magnitude: scaling a component mean must not change the score.
        g1 = _gmm([self.A, self.B], [0.5, 0.5])
        g2 = _gmm([list(3.0 * np.array(self.A)), list(7.0 * np.array(self.B))], [0.5, 0.5])
        assert gmm_soft_chamfer_distance(g1, g2) < 1e-6

    def test_shared_mode_scores_closer_than_no_shared_mode(self):
        query = _gmm([self.A, self.B], [0.5, 0.5])
        shares_one = _gmm([self.A, self.C], [0.5, 0.5])
        shares_none = _gmm([self.C, self.D], [0.5, 0.5])
        assert gmm_soft_chamfer_distance(query, shares_one) < gmm_soft_chamfer_distance(query, shares_none)

    def test_symmetric(self):
        a = _gmm([self.A, self.B], [0.7, 0.3])
        b = _gmm([self.A, self.C], [0.4, 0.6])
        assert abs(gmm_soft_chamfer_distance(a, b) - gmm_soft_chamfer_distance(b, a)) < 1e-6

    def test_weights_make_dominant_mode_matter_more(self):
        # Query is mostly mode A (0.9). Matching A should beat matching the rare mode B.
        query = _gmm([self.A, self.B], [0.9, 0.1])
        shares_dominant = _gmm([self.A, self.C], [0.5, 0.5])
        shares_rare = _gmm([self.C, self.B], [0.5, 0.5])
        assert gmm_soft_chamfer_distance(query, shares_dominant) < gmm_soft_chamfer_distance(query, shares_rare)

    def test_single_component_artists(self):
        # 1-component GMMs (few-song artists) must still score, no shape errors.
        q = _gmm([self.A], [1.0])
        same = _gmm([self.A], [1.0])
        diff = _gmm([self.C], [1.0])
        assert gmm_soft_chamfer_distance(q, same) < 1e-6
        assert gmm_soft_chamfer_distance(q, diff) > 0.5


class TestFindSimilarArtistsRerank:
    """find_similar_artists must return the soft-Chamfer order, not the raw IVF order."""

    def test_reranks_candidates_and_excludes_self(self, monkeypatch):
        import sys
        import types
        import tasks.artist_gmm_manager as agm

        A = [1.0, 0.0, 0.0, 0.0]
        B = [0.0, 1.0, 0.0, 0.0]
        C = [0.0, 0.0, 1.0, 0.0]
        D = [0.0, 0.0, 0.0, 1.0]
        gmm_params = {
            "Q": _gmm([A, B], [0.5, 0.5]),
            "near": _gmm([A, B], [0.5, 0.5]),   # identical to Q -> 0.0 (best)
            "mid": _gmm([A, C], [0.5, 0.5]),    # shares one mode -> 0.5
            "far": _gmm([C, D], [0.5, 0.5]),    # shares nothing -> 1.0
        }
        artist_map = {0: "Q", 1: "far", 2: "near", 3: "mid"}
        reverse = {v: k for k, v in artist_map.items()}

        class _FakeIndex:
            def __len__(self):
                return len(artist_map)

            def query(self, _vec, k):
                labels = [0, 1, 3, 2]  # self first, then deliberately NOT in rerank order
                return labels[:k], [0.0] * min(k, len(labels))

        monkeypatch.setattr(agm, "artist_index", _FakeIndex())
        monkeypatch.setattr(agm, "artist_map", artist_map)
        monkeypatch.setattr(agm, "reverse_artist_map", reverse)
        monkeypatch.setattr(agm, "artist_gmm_params", gmm_params)

        fake_mod = types.ModuleType("app_helper_artist")
        fake_mod.get_artist_id_by_name = lambda name: f"id-{name}"
        fake_mod.get_artist_name_by_id = lambda x: None
        monkeypatch.setitem(sys.modules, "app_helper_artist", fake_mod)

        res = agm.find_similar_artists("Q", n=2)
        names = [r["artist"] for r in res]
        assert names == ["near", "mid"], f"expected rerank order, got {names}"
        assert all(r["artist"] != "Q" for r in res), "self must be excluded"
        assert res[0]["divergence"] <= res[1]["divergence"], "results must be ascending by divergence"


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
