# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the saved-anchor embedding guard outside Song Alchemy.

Similar Song and Song Path seed a search with an anchor's stored centroid, so
they must refuse an anchor saved under another embedding model or dimension
exactly like Song Alchemy and the radios do.

Main Features:
* anchor_embedding_problem accepts a matching legacy or stamped anchor (also one
  stamped while the model file was unreadable) and names a wrong centroid size,
  a different model stamp or a different dimension.
* /api/similar_tracks answers 400 for a mismatched anchor without querying the
  index, and still queries it for a matching one.
* The Song Path anchor resolver returns None for a mismatched anchor.
"""

from unittest.mock import patch

import pytest
from flask import Flask

import app_ivf
import app_path
import config
from tasks.song_alchemy import anchor_embedding_problem, anchor_embedding_tag

DIM = config.EMBEDDING_DIMENSION


def _anchor(centroid_size=DIM, stamp=None):
    anchor = {'id': 7, 'name': 'Mix', 'centroid': [0.1] * centroid_size, 'exclusions': None}
    if stamp is not None:
        anchor['inclusions'] = {**stamp, 'points': []}
    return anchor


@pytest.fixture(autouse=True)
def readable_model_file(tmp_path, monkeypatch):
    model = tmp_path / 'musicnn_embedding.onnx'
    model.write_bytes(b'embedding-model-under-test')
    monkeypatch.setattr(config, 'EMBEDDING_MODEL_PATH', str(model))
    return model


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(app_ivf.ivf_bp)
    app.config['TESTING'] = True
    return app.test_client()


class TestAnchorEmbeddingProblem:
    def test_matching_legacy_anchor_has_no_problem(self):
        assert anchor_embedding_problem(_anchor()) is None

    def test_matching_stamped_anchor_has_no_problem(self):
        assert anchor_embedding_problem(_anchor(stamp=anchor_embedding_tag())) is None

    def test_wrong_centroid_size_is_a_problem(self):
        assert 'centroid' in anchor_embedding_problem(_anchor(centroid_size=DIM - 1))

    def test_other_model_stamp_is_a_problem(self):
        stamp = {**anchor_embedding_tag(), 'embedding_model_sha256': 'another-model'}
        assert 'saved with embedding' in anchor_embedding_problem(_anchor(stamp=stamp))

    def test_stamp_saved_while_the_model_was_unreadable_still_matches(self):
        stamp = {**anchor_embedding_tag(), 'embedding_model_sha256': None}
        assert anchor_embedding_problem(_anchor(stamp=stamp)) is None

    def test_other_dimension_stamp_is_a_problem_even_without_fingerprint(self):
        stamp = {'embedding_model_sha256': None, 'dimension': DIM + 1}
        assert 'saved with embedding' in anchor_embedding_problem(_anchor(stamp=stamp))


class TestSimilarTracksByAnchor:
    def test_mismatched_anchor_is_refused_without_querying(self, client):
        stamp = {**anchor_embedding_tag(), 'embedding_model_sha256': 'another-model'}
        with (
            patch('database.get_alchemy_anchor_by_id', return_value=_anchor(stamp=stamp)),
            patch.object(app_ivf, '_vector_neighbors_or_error') as neighbors,
        ):
            response = client.get('/api/similar_tracks', query_string={'anchor_id': 7})
        assert response.status_code == 400
        assert 're-save the anchor' in response.get_json()['error']
        neighbors.assert_not_called()

    def test_matching_anchor_queries_the_index(self, client):
        with (
            patch('database.get_alchemy_anchor_by_id', return_value=_anchor(stamp=anchor_embedding_tag())),
            patch.object(app_ivf, '_vector_neighbors_or_error', return_value=([], None)) as neighbors,
        ):
            response = client.get('/api/similar_tracks', query_string={'anchor_id': 7})
        assert response.status_code == 200
        neighbors.assert_called_once()


class TestPathAnchorResolver:
    def test_mismatched_anchor_resolves_to_nothing(self):
        with (
            patch('database.get_alchemy_anchor_by_id', return_value=_anchor(centroid_size=DIM + 1)),
            patch.object(app_path, '_find_nearest_song_excluding_vector') as nearest,
        ):
            assert app_path._resolve_anchor_to_song_id(7) is None
        nearest.assert_not_called()

    def test_matching_anchor_resolves_to_its_nearest_song(self):
        with (
            patch('database.get_alchemy_anchor_by_id', return_value=_anchor()),
            patch.object(app_path, '_find_nearest_song_excluding_vector', return_value='song-1'),
        ):
            assert app_path._resolve_anchor_to_song_id(7) == 'song-1'
