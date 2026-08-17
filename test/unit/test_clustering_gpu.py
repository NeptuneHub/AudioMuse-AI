# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""GPU/CPU routing logic for the clustering factory and helpers.

These tests run without a CUDA device, so every GPU path is exercised through
its CPU fallback or through a mocked factory.

Main Features:
* GPUSpectralClustering._gpu_supported accepts only the combinations cuML implements
* get_clustering_model threads n_init through both the CPU and the GPU branch
* _split_oversized_clusters goes through the factory on both the CPU and the GPU branch
* _apply_clustering_model derives use_gpu from the model's using_gpu flag, not from config
* GPUSpectralClustering falls back to scikit-learn when no GPU is present
* GPUPCA.inverse_transform reconstructs on CPU when the GPU call fails
"""

import numpy as np
import pytest

from tasks.clustering_gpu import (
    GPUSpectralClustering,
    GPUPCA,
    get_clustering_model,
)


class TestSpectralGpuSupported:
    def test_kmeans_assign_labels_with_nearest_neighbors(self):
        model = GPUSpectralClustering(
            n_clusters=3, assign_labels='kmeans', affinity='nearest_neighbors'
        )
        assert model._gpu_supported() is True

    def test_kmeans_assign_labels_with_precomputed(self):
        model = GPUSpectralClustering(
            n_clusters=3, assign_labels='kmeans', affinity='precomputed'
        )
        assert model._gpu_supported() is True

    @pytest.mark.parametrize(
        "assign_labels,affinity",
        [
            ('kmeans', 'rbf'),
            ('discretize', 'nearest_neighbors'),
            ('cluster_qr', 'precomputed'),
        ],
    )
    def test_unsupported_combinations_are_not_gpu_routed(self, assign_labels, affinity):
        model = GPUSpectralClustering(
            n_clusters=3, assign_labels=assign_labels, affinity=affinity
        )
        assert model._gpu_supported() is False


class TestGetClusteringModelNInit:
    def test_cpu_kmeans_honors_n_init(self):
        model = get_clustering_model('kmeans', {'n_clusters': 4}, use_gpu=False, n_init=3)
        assert model.n_init == 3

    def test_cpu_gmm_honors_n_init(self):
        model = get_clustering_model('gmm', {'n_components': 4}, use_gpu=False, n_init=3)
        assert model.n_init == 3

    def test_gpu_kmeans_honors_n_init(self):
        model = get_clustering_model('kmeans', {'n_clusters': 4}, use_gpu=True, n_init=3)
        assert model.n_init == 3

    def test_default_n_init_is_ten(self):
        model = get_clustering_model('kmeans', {'n_clusters': 4}, use_gpu=False)
        assert model.n_init == 10


class TestSplitOversizedClustersRouting:
    def test_gpu_branch_requests_kmeans_with_n_init_three(self, monkeypatch):
        from tasks import clustering_helper

        captured = {}

        class FakeKMeans:
            def fit_predict(self, X):
                return np.zeros(len(X), dtype=int)

        def fake_factory(method, params, use_gpu=False, n_init=10):
            captured['method'] = method
            captured['params'] = params
            captured['use_gpu'] = use_gpu
            captured['n_init'] = n_init
            return FakeKMeans()

        monkeypatch.setattr(clustering_helper, 'get_clustering_model', fake_factory)

        rng = np.random.default_rng(0)
        data = rng.standard_normal((600, 4))
        labels = np.zeros(600, dtype=int)
        result = clustering_helper._split_oversized_clusters(labels, data, use_gpu=True)

        assert captured['method'] == 'kmeans'
        assert captured['use_gpu'] is True
        assert captured['n_init'] == 3
        assert result.dtype == labels.dtype

    def test_cpu_branch_also_uses_factory_with_n_init_three(self, monkeypatch):
        from tasks import clustering_helper

        captured = {}

        class FakeKMeans:
            def fit_predict(self, X):
                return np.zeros(len(X), dtype=int)

        def fake_factory(method, params, use_gpu=False, n_init=10):
            captured['use_gpu'] = use_gpu
            captured['n_init'] = n_init
            return FakeKMeans()

        monkeypatch.setattr(clustering_helper, 'get_clustering_model', fake_factory)

        rng = np.random.default_rng(0)
        data = rng.standard_normal((600, 4))
        labels = np.zeros(600, dtype=int)
        clustering_helper._split_oversized_clusters(labels, data, use_gpu=False)

        assert captured['use_gpu'] is False
        assert captured['n_init'] == 3


class TestApplyClusteringModelRouting:
    def test_split_uses_cpu_when_gpu_wrapper_fell_back(self, monkeypatch):
        from tasks import clustering_helper

        monkeypatch.setattr(clustering_helper, 'USE_GPU_CLUSTERING', True)
        monkeypatch.setattr(clustering_helper, 'GPU_CLUSTERING_AVAILABLE', True)

        class FakeGPUDBSCAN:
            using_gpu = False

            def fit_predict(self, X):
                return np.zeros(len(X), dtype=int)

        monkeypatch.setattr(
            clustering_helper,
            'get_clustering_model',
            lambda method, params, use_gpu=False, n_init=10: FakeGPUDBSCAN(),
        )

        captured = {}

        def fake_split(labels, data, use_gpu=False):
            captured['use_gpu'] = use_gpu
            return labels

        monkeypatch.setattr(clustering_helper, '_split_oversized_clusters', fake_split)

        data = np.random.rand(50, 10)
        method_config = {'method': 'dbscan', 'params': {'eps': 0.5, 'min_samples': 3}}
        clustering_helper._apply_clustering_model(data, method_config, "Test", 1)

        assert captured['use_gpu'] is False

    def test_split_uses_gpu_when_wrapper_used_gpu(self, monkeypatch):
        from tasks import clustering_helper

        monkeypatch.setattr(clustering_helper, 'USE_GPU_CLUSTERING', True)
        monkeypatch.setattr(clustering_helper, 'GPU_CLUSTERING_AVAILABLE', True)

        class FakeGPUDBSCAN:
            using_gpu = True

            def fit_predict(self, X):
                return np.zeros(len(X), dtype=int)

        monkeypatch.setattr(
            clustering_helper,
            'get_clustering_model',
            lambda method, params, use_gpu=False, n_init=10: FakeGPUDBSCAN(),
        )

        captured = {}

        def fake_split(labels, data, use_gpu=False):
            captured['use_gpu'] = use_gpu
            return labels

        monkeypatch.setattr(clustering_helper, '_split_oversized_clusters', fake_split)

        data = np.random.rand(50, 10)
        method_config = {'method': 'dbscan', 'params': {'eps': 0.5, 'min_samples': 3}}
        clustering_helper._apply_clustering_model(data, method_config, "Test", 1)

        assert captured['use_gpu'] is True


class TestSpectralCpuFallback:
    def test_spectral_falls_back_to_cpu_when_gpu_unavailable(self, monkeypatch):
        from tasks import clustering_gpu

        monkeypatch.setattr(clustering_gpu, 'check_gpu_available', lambda: False)

        model = GPUSpectralClustering(
            n_clusters=3, assign_labels='kmeans', affinity='nearest_neighbors'
        )
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 8)).astype(np.float32)
        labels = model.fit_predict(X)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == 60
        assert model.using_gpu is False


class TestPcaInverseTransformFallback:
    def test_cpu_reconstruction_used_when_gpu_inverse_fails(self):
        pca = GPUPCA(n_components=2)
        pca.components_ = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
        pca.mean_ = np.array([1.0, 1.0, 0.0, 0.0])

        class FailingGpuModel:
            def inverse_transform(self, X):
                raise RuntimeError("boom")

        pca.model = FailingGpuModel()
        pca.using_gpu = True

        X = np.array([[1.0, 2.0]])
        out = pca.inverse_transform(X)
        expected = X @ pca.components_ + pca.mean_

        assert np.allclose(out, expected)
