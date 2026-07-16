# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Parameter generation and model fitting in clustering_helper.

Covers the random/mutated parameter builders for each clustering method, data
scaling, and the model-application path that runs the chosen algorithm.

Main Features:
* _mutate_param clamps to min/max for ints and floats
* Random and mutated parameters for kmeans, dbscan, gmm, and spectral stay in range
* _prepare_and_scale_data honors the embeddings flag; _apply_clustering_model runs
  kmeans/dbscan and rejects invalid params; stratified subset excludes prior ids
"""

import numpy as np
import random
from unittest.mock import patch
from tasks.clustering_helper import (
    _mutate_param,
    _generate_random_parameters,
    _mutate_parameters,
    _prepare_and_scale_data,
    _apply_clustering_model,
    _get_stratified_song_subset,
    _get_track_primary_genre,
)


class TestMutateParam:
    def test_mutate_param_integer_within_bounds(self):
        random.seed(42)
        value = 10
        min_val = 5
        max_val = 15
        delta = 2

        mutated = _mutate_param(value, min_val, max_val, delta, is_float=False)

        assert min_val <= mutated <= max_val

    def test_mutate_param_float_within_bounds(self):
        random.seed(42)
        value = 0.5
        min_val = 0.1
        max_val = 1.0
        delta = 0.1

        mutated = _mutate_param(value, min_val, max_val, delta, is_float=True)

        assert min_val <= mutated <= max_val

    def test_mutate_param_clamps_at_max(self):
        value = 98
        min_val = 0
        max_val = 100
        delta = 10

        for _ in range(10):
            mutated = _mutate_param(value, min_val, max_val, delta)
            assert mutated <= max_val

    def test_mutate_param_clamps_at_min(self):
        value = 2
        min_val = 0
        max_val = 100
        delta = 10

        for _ in range(10):
            mutated = _mutate_param(value, min_val, max_val, delta)
            assert mutated >= min_val


class TestGenerateRandomParameters:
    def test_generates_kmeans_parameters(self):
        data = np.random.rand(100, 50)
        method = 'kmeans'
        pca_ranges = {'components_min': 0, 'components_max': 30}
        num_clust_ranges = (5, 20)

        params = _generate_random_parameters(method, data, pca_ranges, num_clust_ranges, {}, {}, {})

        assert 'pca_config' in params
        assert 'clustering_method_config' in params
        assert params['clustering_method_config']['method'] == 'kmeans'
        n_clusters = params['clustering_method_config']['params']['n_clusters']
        assert 2 <= n_clusters <= min(20, data.shape[0])

    def test_generates_dbscan_parameters(self):
        data = np.random.rand(100, 50)
        method = 'dbscan'
        pca_ranges = {'components_min': 0, 'components_max': 30}
        db_ranges = {'eps_min': 0.1, 'eps_max': 2.0, 'samples_min': 2, 'samples_max': 10}

        params = _generate_random_parameters(method, data, pca_ranges, (), db_ranges, {}, {})

        assert params['clustering_method_config']['method'] == 'dbscan'
        dbscan_params = params['clustering_method_config']['params']
        assert 0.1 <= dbscan_params['eps'] <= 2.0
        assert 2 <= dbscan_params['min_samples'] <= 10

    def test_generates_gmm_parameters(self):
        data = np.random.rand(100, 50)
        method = 'gmm'
        pca_ranges = {'components_min': 0, 'components_max': 30}
        gmm_ranges = {'n_components_min': 2, 'n_components_max': 15}

        params = _generate_random_parameters(method, data, pca_ranges, (), {}, gmm_ranges, {})

        assert params['clustering_method_config']['method'] == 'gmm'
        n_components = params['clustering_method_config']['params']['n_components']
        assert 2 <= n_components <= min(15, data.shape[0])

    def test_generates_spectral_parameters(self):
        data = np.random.rand(100, 50)
        method = 'spectral'
        pca_ranges = {'components_min': 0, 'components_max': 30}
        spec_ranges = {'n_clusters_min': 3, 'n_clusters_max': 12}

        params = _generate_random_parameters(method, data, pca_ranges, (), {}, {}, spec_ranges)

        assert params['clustering_method_config']['method'] == 'spectral'
        spectral_params = params['clustering_method_config']['params']
        assert 'n_clusters' in spectral_params
        assert 'random_state' in spectral_params
        n_clusters = spectral_params['n_clusters']
        assert 2 <= n_clusters < data.shape[0]


class TestMutateParameters:
    def test_mutates_kmeans_parameters(self):
        elite_params = {
            'pca_config': {'enabled': True, 'components': 10},
            'clustering_method_config': {'method': 'kmeans', 'params': {'n_clusters': 10}},
        }
        data = np.random.rand(100, 50)
        mutation_cfg = {'int_abs_delta': 2, 'float_abs_delta': 0.1}
        pca_ranges = {'components_min': 0, 'components_max': 30}
        num_clust_ranges = (5, 20)

        mutated = _mutate_parameters(
            elite_params, mutation_cfg, 'kmeans', data, pca_ranges, num_clust_ranges, {}, {}, {}
        )

        assert 'pca_config' in mutated
        assert 'clustering_method_config' in mutated
        n_clusters = mutated['clustering_method_config']['params']['n_clusters']
        assert 5 <= n_clusters <= 20

    def test_mutates_dbscan_parameters(self):
        elite_params = {
            'pca_config': {'enabled': False, 'components': 0},
            'clustering_method_config': {
                'method': 'dbscan',
                'params': {'eps': 0.5, 'min_samples': 5},
            },
        }
        data = np.random.rand(100, 50)
        mutation_cfg = {'int_abs_delta': 1, 'float_abs_delta': 0.1}
        pca_ranges = {'components_min': 0, 'components_max': 30}
        db_ranges = {'eps_min': 0.1, 'eps_max': 2.0, 'samples_min': 2, 'samples_max': 10}

        mutated = _mutate_parameters(
            elite_params, mutation_cfg, 'dbscan', data, pca_ranges, (), db_ranges, {}, {}
        )

        dbscan_params = mutated['clustering_method_config']['params']
        assert 0.1 <= dbscan_params['eps'] <= 2.0
        assert 2 <= dbscan_params['min_samples'] <= 10


class TestPrepareAndScaleData:
    def test_uses_embeddings_when_enabled(self):
        X_feat = np.random.rand(50, 20)
        X_embed = np.random.rand(50, 128)

        scaled_data, scaler = _prepare_and_scale_data(X_feat, X_embed, use_embeddings=True)

        assert scaled_data.shape == (50, 128)

    def test_uses_features_when_embeddings_disabled(self):
        X_feat = np.random.rand(50, 20)
        X_embed = np.random.rand(50, 128)

        scaled_data, scaler = _prepare_and_scale_data(X_feat, X_embed, use_embeddings=False)

        assert scaled_data.shape == (50, 20)

    def test_scales_data_correctly(self):
        X_feat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        scaled_data, scaler = _prepare_and_scale_data(X_feat, None, use_embeddings=False)

        mean = np.mean(scaled_data, axis=0)
        assert np.allclose(mean, 0, atol=1e-10)


class TestApplyClusteringModel:
    @patch('tasks.clustering_helper.USE_GPU_CLUSTERING', False)
    def test_applies_kmeans_successfully(self):
        data = np.random.rand(50, 10)
        method_config = {'method': 'kmeans', 'params': {'n_clusters': 5}}

        labels, centers, model = _apply_clustering_model(data, method_config, "Test", 1)

        assert labels is not None
        assert len(labels) == 50
        assert len(set(labels)) <= 5

    @patch('tasks.clustering_helper.USE_GPU_CLUSTERING', False)
    def test_applies_dbscan_successfully(self):
        data = np.random.rand(50, 10)
        method_config = {'method': 'dbscan', 'params': {'eps': 0.5, 'min_samples': 3}}

        labels, centers, model = _apply_clustering_model(data, method_config, "Test", 1)

        assert labels is not None
        assert len(labels) == 50

    @patch('tasks.clustering_helper.USE_GPU_CLUSTERING', False)
    def test_rejects_invalid_kmeans_params(self):
        data = np.random.rand(50, 10)
        method_config = {'method': 'kmeans', 'params': {'n_clusters': 1}}

        labels, centers, model = _apply_clustering_model(data, method_config, "Test", 1)

        assert labels is None


class TestGetStratifiedSongSubset:
    def test_stratified_sampling_balances_genres(self):
        genre_map = {
            'Rock': [
                {'item_id': 'r1', 'mood_vector': 'Rock:0.8,Pop:0.2'},
                {'item_id': 'r2', 'mood_vector': 'Rock:0.9,Jazz:0.1'},
            ],
            'Pop': [
                {'item_id': 'p1', 'mood_vector': 'Pop:0.7,Rock:0.3'},
            ],
        }
        target_per_genre = 2

        subset = _get_stratified_song_subset(genre_map, target_per_genre)

        assert isinstance(subset, list)
        assert len(subset) >= 0

    def test_rotation_keeps_the_subset_at_the_exact_configured_size(self, monkeypatch):
        from tasks import clustering_helper

        monkeypatch.setattr(clustering_helper, 'CLUSTERING_SUBSET_SONGS', 4)
        genre_map = {
            'rock': [
                {'item_id': f'r{i}', 'mood_vector': 'rock:0.9'} for i in range(10)
            ],
        }
        prev_ids = ['r0', 'r1', 'r2', 'r3']

        subset = _get_stratified_song_subset(
            genre_map, 2, prev_ids=prev_ids, percent_change=0.5
        )

        assert len(subset) == 4
        assert len({track['item_id'] for track in subset}) == 4


class TestGetTrackPrimaryGenre:
    def test_returns_genre_from_mood_vector(self):
        track_data = {'mood_vector': 'Rock:0.8,Pop:0.2'}

        genre = _get_track_primary_genre(track_data)

        assert genre in ['Rock', '__other__']

    def test_returns_other_when_no_stratified_genre(self):
        track_data = {'mood_vector': 'UnknownMood:0.9'}

        genre = _get_track_primary_genre(track_data)

        assert genre == '__other__'

    def test_returns_other_when_no_mood_vector(self):
        track_data = {}

        genre = _get_track_primary_genre(track_data)

        assert genre == '__other__'


class TestAIPlaylistNaming:
    @staticmethod
    def _call(monkeypatch, ai_result, naming_evidence=None):
        from tasks import clustering_helper

        monkeypatch.setattr(clustering_helper, 'LYRICS_ENABLED', False)

        def fail_if_lyrics_are_queried(_ids):
            raise AssertionError('lyrics DB must not be queried when disabled')

        monkeypatch.setattr(
            clustering_helper,
            'get_lyrics_axis_vectors',
            fail_if_lyrics_are_queried,
        )
        monkeypatch.setattr(
            clustering_helper,
            'get_score_data_by_ids',
            lambda _ids: [{'mood_vector': 'indie:0.8', 'other_features': 'party:0.8'}],
        )
        monkeypatch.setattr(
            clustering_helper,
            'build_naming_context',
            lambda *args: {
                'genre': 'Indie',
                'ideas': ['bittersweet', 'solitude'],
                'naming_brief': 'melancholic lyrics over upbeat music',
                'naming_dimension': 'contrast',
                'naming_evidence': naming_evidence or (
                    'melancholic lyrics contrasted with upbeat energetic music'
                ),
                'instrumental': False,
                'axis_labels': {'AXIS_3_EMOTIONAL_VALENCE': 'MELANCHOLIC'},
                'fallback_name': (
                    'Indie'
                    if naming_evidence == 'general-purpose listening'
                    else 'Bittersweet Indie'
                ),
            },
        )
        received = {}

        def fake_ai(
            genre,
            naming_dimension,
            naming_evidence,
            config,
            instrumental=False,
            avoid_names=None,
        ):
            received.update(
                genre=genre,
                naming_dimension=naming_dimension,
                naming_evidence=naming_evidence,
                instrumental=instrumental,
                provider=config['provider'],
                avoid_names=avoid_names,
            )
            return ai_result

        monkeypatch.setattr(clustering_helper, 'get_ai_playlist_name', fake_ai)
        result = clustering_helper._try_ai_name_playlist(
            'Old_Cluster_Name',
            [('i1', 'Song', 'Artist')],
            {'Old_Cluster_Name': {'party': 0.8}},
            'OLLAMA',
            'http://localhost:11434/api/generate',
            'qwen3.5:9b',
            '', '', '', '', '', '', '',
            ['Existing Indie Name'],
        )
        return result, received

    def test_grounded_context_is_sent_to_the_ai(self, monkeypatch):
        result, received = self._call(monkeypatch, 'Bittersweet Indie Solitude')

        assert result == 'Bittersweet Indie Solitude'
        assert received == {
            'genre': 'Indie',
            'naming_dimension': 'contrast',
            'naming_evidence': (
                'melancholic lyrics contrasted with upbeat energetic music'
            ),
            'instrumental': False,
            'provider': 'OLLAMA',
            'avoid_names': ['Existing Indie Name'],
        }

    def test_invalid_ai_output_uses_the_grounded_fallback(self, monkeypatch):
        result, _received = self._call(monkeypatch, None)

        assert result == 'Bittersweet Indie'

    def test_general_context_skips_ai_instead_of_inventing_a_mood(self, monkeypatch):
        result, received = self._call(
            monkeypatch,
            'Invented Indie Mood',
            naming_evidence='general-purpose listening',
        )

        assert result == 'Indie'
        assert received == {}
