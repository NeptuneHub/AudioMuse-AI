# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Per-artist GMM fitting and soft-Chamfer similarity in artist_gmm_manager.

Covers GMM parameter construction from track embeddings, the mode-to-mode
divergence metric, and candidate reranking in find_similar_artists.

Main Features:
* fit_artist_gmm produces normalized weights, correct means shape, few-songs flag,
  and omits covariance fields
* gmm_soft_chamfer_distance is zero for identical, scale-invariant, symmetric,
  and weight-sensitive; find_similar_artists reranks and excludes self
* _fit_pending_artists refits in-process, loudly, when the worker pool cannot run
"""

import logging

import numpy as np
import pytest
from tasks.artist_gmm_manager import (
    fit_artist_gmm,
    get_representative_songs_for_component,
    GMM_N_COMPONENTS_MAX,
    gmm_soft_chamfer_distance,
    _cosine_distance_matrix,
)


def _gmm(means, weights):
    return {"means": [list(m) for m in means], "weights": list(weights)}


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

    def test_few_song_means_are_unit_normed_for_cosine(self):
        embeddings = [np.random.rand(128) for _ in range(3)]

        gmm_params = fit_artist_gmm("Cosine Artist", embeddings)

        means = np.array(gmm_params['means'])
        assert np.allclose(np.linalg.norm(means, axis=1), 1.0, atol=1e-5)

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

        from tasks.mediaserver import registry as _registry
        monkeypatch.setattr(_registry, "artist_names_for_ids", lambda ids, *a, **k: {})
        monkeypatch.setattr(
            _registry, "artist_ids_for_names",
            lambda names, *a, **k: {str(n): f"id-{n}" for n in names},
        )

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

    def test_zero_dimensional_embeddings_return_none_instead_of_raising(self, caplog):
        embeddings = [np.array([]) for _ in range(5)]

        with caplog.at_level(logging.ERROR, logger="tasks.artist_gmm_manager"):
            gmm_params = fit_artist_gmm("Invalid Artist", embeddings)

        assert gmm_params is None
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, "the fit failure must be logged at ERROR"
        assert "Failed to fit GMM for artist 'Invalid Artist'" in errors[0].message
        assert errors[0].exc_info is not None

    def test_mismatched_embedding_dimensions_return_none_instead_of_raising(self, caplog):
        rng = np.random.default_rng(0)
        embeddings = [rng.random(128), rng.random(64), rng.random(128)]

        with caplog.at_level(logging.ERROR, logger="tasks.artist_gmm_manager"):
            gmm_params = fit_artist_gmm("Mismatched Artist", embeddings)

        assert gmm_params is None
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, "the dimension mismatch must be logged at ERROR"
        assert "Failed to fit GMM for artist 'Mismatched Artist'" in errors[0].message
        assert errors[0].exc_info is not None

    def test_very_large_component_count(self):
        embeddings = [np.random.rand(128) for _ in range(100)]

        gmm_params = fit_artist_gmm("Popular Artist", embeddings)

        assert gmm_params['n_components'] <= GMM_N_COMPONENTS_MAX


class TestRepresentativeSongs:
    def test_ranks_by_direction_not_norm(self, monkeypatch):
        import tasks.artist_gmm_manager as agm

        monkeypatch.setattr(
            agm, "artist_gmm_params", {"Artist": {"means": [[1.0, 0.0]], "weights": [1.0]}}
        )
        rows = [
            ("same_small", "S small", np.array([0.01, 0.0], dtype=np.float32).tobytes()),
            ("wrong", "W", np.array([0.0, 5.0], dtype=np.float32).tobytes()),
            ("same_big", "S big", np.array([100.0, 0.0], dtype=np.float32).tobytes()),
        ]

        class _FakeCursor:
            def __init__(self, data):
                self._data = data

            def execute(self, *args, **kwargs):
                pass

            def fetchall(self):
                return self._data

            def close(self):
                pass

        class _FakeConn:
            def cursor(self):
                return _FakeCursor(rows)

        monkeypatch.setattr("database.get_db", lambda: _FakeConn())
        songs = get_representative_songs_for_component("Artist", 0, top_k=3)
        ids = [song["item_id"] for song in songs]
        assert set(ids[:2]) == {"same_small", "same_big"}
        assert ids[2] == "wrong"


class TestPoolFallback:
    ARTISTS = ["A", "B", "C"]

    def _harness(self, monkeypatch, workers, dispatches):
        import tasks.artist_gmm_manager as agm

        monkeypatch.setattr(agm, "_gmm_worker_count", lambda _pending: workers)
        monkeypatch.setattr(agm, "_shutdown_gmm_pool", lambda: None)
        monkeypatch.setattr(agm, "_release_gmm_pool_temp_folders", lambda: None)
        monkeypatch.setattr(
            agm, "_run_fit_batches",
            lambda cur, pending, tracks, hashes, dispatch: (
                dispatches.append(dispatch) or dict.fromkeys(pending, {"fitted": True})
            ),
        )
        return agm

    def test_a_dead_worker_pool_refits_every_artist_in_process(self, monkeypatch):
        dispatches = []
        agm = self._harness(monkeypatch, 4, dispatches)

        class _DeadPool:
            def __enter__(self):
                raise RuntimeError("A worker process managed by the executor died")

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(agm, "Parallel", lambda **kwargs: _DeadPool())

        fitted = agm._fit_pending_artists(None, self.ARTISTS, {}, {})

        assert sorted(fitted) == self.ARTISTS
        assert len(dispatches) == 1

    def test_a_dead_worker_pool_is_reported_with_a_traceback(self, monkeypatch, caplog):
        agm = self._harness(monkeypatch, 4, [])

        class _DeadPool:
            def __enter__(self):
                raise RuntimeError("A worker process managed by the executor died")

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(agm, "Parallel", lambda **kwargs: _DeadPool())

        with caplog.at_level(logging.ERROR, logger="tasks.artist_gmm_manager"):
            agm._fit_pending_artists(None, self.ARTISTS, {}, {})

        records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert records, "a dead worker pool must be logged at ERROR"
        assert "ARTIST GMM WORKER POOL FAILED" in records[0].message
        assert records[0].exc_info is not None

    def test_the_pool_is_shut_down_before_the_in_process_refit_starts(self, monkeypatch):
        dispatches = []
        agm = self._harness(monkeypatch, 4, dispatches)
        events = []

        class _DeadPool:
            def __enter__(self):
                raise RuntimeError("boom")

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(agm, "Parallel", lambda **kwargs: _DeadPool())
        monkeypatch.setattr(agm, "_shutdown_gmm_pool", lambda: events.append("shutdown"))
        monkeypatch.setattr(
            agm, "_fit_artists_in_process",
            lambda *args: events.append("refit") or {},
        )

        agm._fit_pending_artists(None, self.ARTISTS, {}, {})

        assert events == ["shutdown", "refit"]

    def test_a_healthy_worker_pool_is_used_and_never_refits_in_process(self, monkeypatch):
        dispatches = []
        agm = self._harness(monkeypatch, 4, dispatches)

        class _LivePool:
            def __enter__(self):
                return lambda jobs: jobs

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(agm, "Parallel", lambda **kwargs: _LivePool())
        monkeypatch.setattr(
            agm, "_fit_artists_in_process",
            lambda *args: pytest.fail("a healthy pool must not trigger the refit"),
        )

        fitted = agm._fit_pending_artists(None, self.ARTISTS, {}, {})

        assert sorted(fitted) == self.ARTISTS
        assert len(dispatches) == 1

    def test_a_single_worker_count_skips_the_pool_entirely(self, monkeypatch):
        dispatches = []
        agm = self._harness(monkeypatch, 1, dispatches)
        monkeypatch.setattr(
            agm, "Parallel",
            lambda **kwargs: pytest.fail("workers <= 1 must not build a pool"),
        )

        fitted = agm._fit_pending_artists(None, self.ARTISTS, {}, {})

        assert sorted(fitted) == self.ARTISTS
        assert len(dispatches) == 1
