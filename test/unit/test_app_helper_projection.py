# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Contract for the 2D projection fallback chain in app_helper.

A projection build must either produce real coordinates or fail loudly: an
all-zeros projection stored as success renders every point at the origin, which
is silent data corruption for the map and artist pages.

Main Features:
* UMAP failure falls back to PCA and still yields real coordinates
* When UMAP and PCA both fail the projection raises instead of returning zeros
* build_and_store_map_projection propagates projector failures to its caller
  instead of persisting a zero matrix as a successful build
"""

import numpy as np
import pytest

import app_helper
from tasks import alchemy_projections


class TestProjectMatrix2d:
    def test_umap_failure_falls_back_to_pca(self, monkeypatch):
        def umap_fail(_mat):
            raise RuntimeError('umap unavailable')

        monkeypatch.setattr(alchemy_projections, '_project_with_umap', umap_fail)
        monkeypatch.setattr(
            alchemy_projections,
            '_project_to_2d',
            lambda mat: np.ones((mat.shape[0], 2)),
        )

        out = app_helper._project_matrix_2d(np.zeros((3, 4), dtype=np.float32), 'test')

        assert out.shape == (3, 2)
        assert np.all(out == 1.0)

    def test_raises_when_umap_and_pca_both_fail(self, monkeypatch):
        def fail(_mat):
            raise RuntimeError('projector broken')

        monkeypatch.setattr(alchemy_projections, '_project_with_umap', fail)
        monkeypatch.setattr(alchemy_projections, '_project_to_2d', fail)

        with pytest.raises(RuntimeError, match='refusing to store'):
            app_helper._project_matrix_2d(np.zeros((3, 4), dtype=np.float32), 'test')

    def test_raises_when_no_projector_produces_output(self, monkeypatch):
        monkeypatch.setattr(alchemy_projections, '_project_with_umap', lambda mat: None)
        monkeypatch.setattr(alchemy_projections, '_project_to_2d', lambda mat: None)

        with pytest.raises(RuntimeError, match='refusing to store'):
            app_helper._project_matrix_2d(np.zeros((3, 4), dtype=np.float32), 'test')


class TestMapProjectionBuild:
    def test_map_projection_build_propagates_projector_failure(self, monkeypatch):
        from tasks import index_build_helpers

        def fail(_mat):
            raise RuntimeError('projector broken')

        monkeypatch.setattr(
            index_build_helpers,
            'stream_embeddings_to_buffer',
            lambda **_kw: (np.zeros((2, 4), dtype=np.float32), ['a', 'b']),
        )
        monkeypatch.setattr(alchemy_projections, '_project_with_umap', fail)
        monkeypatch.setattr(alchemy_projections, '_project_to_2d', fail)

        saved = []
        monkeypatch.setattr(app_helper, 'save_map_projection', lambda *a: saved.append(a))

        with pytest.raises(RuntimeError, match='refusing to store'):
            app_helper.build_and_store_map_projection('main_map')

        assert saved == []
