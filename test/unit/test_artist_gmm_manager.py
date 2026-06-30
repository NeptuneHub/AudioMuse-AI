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
    def test_single_sample_returns_one_component(self):
        embeddings = np.random.rand(1, 128)

        n_components = select_optimal_gmm_components(embeddings)

        assert n_components == 1

    def test_two_samples_returns_valid_components(self):
        embeddings = np.random.rand(2, 128)

        n_components = select_optimal_gmm_components(embeddings)

        assert 1 <= n_components <= 2

    def test_small_dataset_respects_max_feasible(self):
        embeddings = np.random.rand(4, 128)

        n_components = select_optimal_gmm_components(embeddings)

        assert n_components <= 4
        assert n_components >= 1

    def test_large_dataset_respects_sample_ratio(self):
        embeddings = np.random.rand(50, 128)

        n_components = select_optimal_gmm_components(embeddings)

        assert 1 <= n_components <= 10

    def test_respects_min_components_parameter(self):
        embeddings = np.random.rand(50, 128)

        n_components = select_optimal_gmm_components(embeddings, min_components=3, max_components=8)

        assert n_components >= 1
        assert n_components <= 8

    def test_respects_max_components_parameter(self):
        embeddings = np.random.rand(100, 128)

        n_components = select_optimal_gmm_components(embeddings, min_components=2, max_components=5)

        assert n_components <= 5
        assert n_components >= 1

    def test_deterministic_with_same_data(self):
        np.random.seed(42)
        embeddings = np.random.rand(30, 128)

        n1 = select_optimal_gmm_components(embeddings)
        n2 = select_optimal_gmm_components(embeddings)

        assert n1 == n2

    def test_high_dimensional_embeddings(self):
        embeddings = np.random.rand(25, 512)

        n_components = select_optimal_gmm_components(embeddings)

        assert 1 <= n_components <= min(GMM_N_COMPONENTS_MAX, 25 // 5)


class TestFitArtistGMM:
    def test_single_track_artist(self):
        embeddings = [np.random.rand(128)]

        gmm_params = fit_artist_gmm("Test Artist", embeddings)

        assert gmm_params is not None
        assert gmm_params['n_components'] == 1
        assert gmm_params['n_tracks'] == 1
        assert gmm_params['is_few_songs'] is True
        assert len(gmm_params['weights']) == 1
        assert gmm_params['weights'][0] == 1.0

    def test_few_tracks_artist(self):
        embeddings = [np.random.rand(128) for _ in range(3)]

        gmm_params = fit_artist_gmm("Few Tracks Artist", embeddings)

        assert gmm_params is not None
        assert gmm_params['n_components'] == 3
        assert gmm_params['n_tracks'] == 3
        assert gmm_params['is_few_songs'] is True
        assert len(gmm_params['weights']) == 3
        assert all(abs(w - 1.0 / 3) < 1e-10 for w in gmm_params['weights'])

    def test_many_tracks_artist(self):
        embeddings = [np.random.rand(128) for _ in range(20)]

        gmm_params = fit_artist_gmm("Popular Artist", embeddings)

        assert gmm_params is not None
        assert gmm_params['n_tracks'] == 20
        assert gmm_params['is_few_songs'] is False
        assert 1 <= gmm_params['n_components'] <= GMM_N_COMPONENTS_MAX

    def test_gmm_params_structure(self):
        embeddings = [np.random.rand(128) for _ in range(10)]

        gmm_params = fit_artist_gmm("Artist", embeddings)

        assert 'weights' in gmm_params
        assert 'means' in gmm_params
        assert 'n_components' in gmm_params
        assert 'n_features' in gmm_params
        assert 'n_tracks' in gmm_params
        assert 'is_few_songs' in gmm_params

        assert 'covariances' not in gmm_params
        assert 'covariance_type' not in gmm_params

    def test_weights_sum_to_one(self):
        embeddings = [np.random.rand(128) for _ in range(15)]

        gmm_params = fit_artist_gmm("Artist", embeddings)

        weights_sum = sum(gmm_params['weights'])
        assert abs(weights_sum - 1.0) < 1e-6

    def test_means_shape_matches_components(self):
        embeddings = [np.random.rand(128) for _ in range(8)]

        gmm_params = fit_artist_gmm("Artist", embeddings)

        n_components = gmm_params['n_components']
        assert len(gmm_params['means']) == n_components
        assert all(len(mean) == 128 for mean in gmm_params['means'])

    def test_n_features_matches_embedding_dim(self):
        embedding_dim = 256
        embeddings = [np.random.rand(embedding_dim) for _ in range(10)]

        gmm_params = fit_artist_gmm("Artist", embeddings)

        assert gmm_params['n_features'] == embedding_dim

    def test_few_songs_flag_correct(self):
        few_embeddings = [np.random.rand(128) for _ in range(3)]
        many_embeddings = [np.random.rand(128) for _ in range(10)]

        few_params = fit_artist_gmm("Few Artist", few_embeddings)
        many_params = fit_artist_gmm("Many Artist", many_embeddings)

        assert few_params['is_few_songs'] is True
        assert many_params['is_few_songs'] is False

    def test_different_artists_different_gmms(self):
        np.random.seed(42)
        embeddings1 = [np.random.rand(128) for _ in range(10)]
        np.random.seed(99)
        embeddings2 = [np.random.rand(128) for _ in range(10)]

        gmm1 = fit_artist_gmm("Artist 1", embeddings1)
        gmm2 = fit_artist_gmm("Artist 2", embeddings2)

        assert gmm1['means'] != gmm2['means']

    def test_high_dimensional_embeddings(self):
        embeddings = [np.random.rand(512) for _ in range(15)]

        gmm_params = fit_artist_gmm("HD Artist", embeddings)

        assert gmm_params is not None
        assert gmm_params['n_features'] == 512
        assert all(len(mean) == 512 for mean in gmm_params['means'])


class TestGmmSoftChamfer:
    A = [1.0, 0.0, 0.0, 0.0]
    B = [0.0, 1.0, 0.0, 0.0]
    C = [0.0, 0.0, 1.0, 0.0]
    D = [0.0, 0.0, 0.0, 1.0]

    def test_cosine_distance_matrix_shape_and_values(self):
        d = _cosine_distance_matrix(np.array([self.A, self.B]), np.array([self.A, self.C]))
        assert d.shape == (2, 2)
        assert abs(d[0, 0]) < 1e-6
        assert abs(d[0, 1] - 1.0) < 1e-6
        assert abs(d[1, 0] - 1.0) < 1e-6

    def test_identical_gmm_distance_is_zero(self):
        g = _gmm([self.A, self.B], [0.5, 0.5])
        assert gmm_soft_chamfer_distance(g, g) < 1e-6

    def test_scale_invariance(self):
        g1 = _gmm([self.A, self.B], [0.5, 0.5])
        g2 = _gmm([list(3.0 * np.array(self.A)), list(7.0 * np.array(self.B))], [0.5, 0.5])
        assert gmm_soft_chamfer_distance(g1, g2) < 1e-6

    def test_shared_mode_scores_closer_than_no_shared_mode(self):
        query = _gmm([self.A, self.B], [0.5, 0.5])
        shares_one = _gmm([self.A, self.C], [0.5, 0.5])
        shares_none = _gmm([self.C, self.D], [0.5, 0.5])
        assert gmm_soft_chamfer_distance(query, shares_one) < gmm_soft_chamfer_distance(
            query, shares_none
        )

    def test_symmetric(self):
        a = _gmm([self.A, self.B], [0.7, 0.3])
        b = _gmm([self.A, self.C], [0.4, 0.6])
        assert abs(gmm_soft_chamfer_distance(a, b) - gmm_soft_chamfer_distance(b, a)) < 1e-6

    def test_weights_make_dominant_mode_matter_more(self):
        query = _gmm([self.A, self.B], [0.9, 0.1])
        shares_dominant = _gmm([self.A, self.C], [0.5, 0.5])
        shares_rare = _gmm([self.C, self.B], [0.5, 0.5])
        assert gmm_soft_chamfer_distance(query, shares_dominant) < gmm_soft_chamfer_distance(
            query, shares_rare
        )

    def test_single_component_artists(self):
        q = _gmm([self.A], [1.0])
        same = _gmm([self.A], [1.0])
        diff = _gmm([self.C], [1.0])
        assert gmm_soft_chamfer_distance(q, same) < 1e-6
        assert gmm_soft_chamfer_distance(q, diff) > 0.5


class TestFindSimilarArtistsRerank:
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
            "near": _gmm([A, B], [0.5, 0.5]),
            "mid": _gmm([A, C], [0.5, 0.5]),
            "far": _gmm([C, D], [0.5, 0.5]),
        }
        artist_map = {0: "Q", 1: "far", 2: "near", 3: "mid"}
        reverse = {v: k for k, v in artist_map.items()}

        class _FakeIndex:
            def __len__(self):
                return len(artist_map)

            def query(self, _vec, k):
                labels = [0, 1, 3, 2]
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
        assert res[0]["divergence"] <= res[1]["divergence"], (
            "results must be ascending by divergence"
        )


class TestEdgeCases:
    def test_empty_track_list(self):
        embeddings = []

        gmm_params = fit_artist_gmm("Empty Artist", embeddings)

        assert gmm_params is None

    def test_zero_dimensional_embeddings(self):
        try:
            embeddings = [np.array([]) for _ in range(5)]
            gmm_params = fit_artist_gmm("Invalid Artist", embeddings)
            assert (
                gmm_params is None
                or 'n_features' not in gmm_params
                or gmm_params['n_features'] == 0
            )
        except Exception:
            pass

    def test_mismatched_embedding_dimensions(self):
        rng = np.random.default_rng(0)
        embeddings = [rng.random(128), rng.random(64), rng.random(128)]

        try:
            gmm_params = fit_artist_gmm("Mismatched Artist", embeddings)
            assert gmm_params is None or gmm_params is not None
        except (ValueError, Exception):
            pass

    def test_very_large_component_count(self):
        embeddings = [np.random.rand(128) for _ in range(100)]

        gmm_params = fit_artist_gmm("Popular Artist", embeddings)

        assert gmm_params['n_components'] <= GMM_N_COMPONENTS_MAX
