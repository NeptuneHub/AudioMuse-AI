# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for app_alchemy anchor create and rename validation.

Registers the alchemy blueprint and posts payloads to confirm the anchor
endpoints reject bad input before touching persistence.

Main Features:
* Whitespace-only anchor names return 400 on create and rename.
* Non-list and empty centroid payloads return 400.
* Malformed exclusions payloads return 400; valid ones reach persistence
  with parsed distances and omitted exclusions are stored as None.
* A centroid that could never be used (wrong size, non-finite, oversized number)
  returns 400 instead of saving an anchor every feature would ignore.
* Malformed inclusions payloads (wrong vector size, non-finite, oversized or
  non-numeric values, bad weight, non-boolean seed, bad group or signature)
  return 400; valid ones reach persistence with parsed weights, seed flags,
  groups and signatures, stamped with the current embedding model and
  dimension; inclusions from a run under another model are refused; omitted
  inclusions are stored as None.
* The legacy payload (name + centroid + exclusions, no inclusions) keeps working.
"""

import pytest
from unittest.mock import patch
from flask import Flask

import config
from app_alchemy import alchemy_bp
from tasks.song_alchemy import anchor_embedding_tag

DIM = config.EMBEDDING_DIMENSION
CENTROID = [0.1] * DIM


@pytest.fixture(autouse=True)
def readable_model_file(tmp_path, monkeypatch):
    model = tmp_path / 'musicnn_embedding.onnx'
    model.write_bytes(b'embedding-model-under-test')
    monkeypatch.setattr(config, 'EMBEDDING_MODEL_PATH', str(model))
    return model


@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(alchemy_bp)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestCreateAnchorValidation:
    @patch('database.save_alchemy_anchor')
    def test_whitespace_only_name_returns_400(self, mock_save, client):
        response = client.post('/api/anchors', json={'name': '   ', 'centroid': CENTROID})
        assert response.status_code == 400
        assert response.get_json()['error'] == 'Anchor name is required'
        assert response.get_json()['error_code'] == 1003
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_non_list_centroid_returns_400(self, mock_save, client):
        response = client.post('/api/anchors', json={'name': 'My Anchor', 'centroid': 'not-a-list'})
        assert response.status_code == 400
        assert response.get_json()['error'] == 'Anchor centroid is required and must be a list'
        assert response.get_json()['error_code'] == 1003
        mock_save.assert_not_called()

    @pytest.mark.parametrize(
        'centroid',
        [[0.1, 0.2], [0.1] * (DIM + 1), [float('nan')] + [0.1] * (DIM - 1), ['x'] + [0.1] * (DIM - 1), [10 ** 400] + [0.1] * (DIM - 1)],
    )
    @patch('database.save_alchemy_anchor')
    def test_centroid_that_cannot_be_used_returns_400(self, mock_save, client, centroid):
        response = client.post('/api/anchors', json={'name': 'My Anchor', 'centroid': centroid})
        assert response.status_code == 400
        assert response.get_json()['error'] == f'Anchor centroid must be a list of {DIM} finite numbers'
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_empty_list_centroid_returns_400(self, mock_save, client):
        response = client.post('/api/anchors', json={'name': 'My Anchor', 'centroid': []})
        assert response.status_code == 400
        assert response.get_json()['error'] == 'Anchor centroid is required and must be a list'
        assert response.get_json()['error_code'] == 1003
        mock_save.assert_not_called()


class TestCreateAnchorExclusionsValidation:
    @patch('database.save_alchemy_anchor')
    def test_non_list_exclusions_returns_400(self, mock_save, client):
        response = client.post(
            '/api/anchors',
            json={'name': 'A', 'centroid': CENTROID, 'exclusions': 'nope'},
        )
        assert response.status_code == 400
        assert response.get_json()['error'] == 'Anchor exclusions must be a list'
        assert response.get_json()['error_code'] == 1003
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_non_object_exclusion_entry_returns_400(self, mock_save, client):
        response = client.post(
            '/api/anchors',
            json={'name': 'A', 'centroid': CENTROID, 'exclusions': [[0.0, 1.0]]},
        )
        assert response.status_code == 400
        assert response.get_json()['error'] == 'Each anchor exclusion must be an object'
        assert response.get_json()['error_code'] == 1003
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_exclusion_entry_without_vector_returns_400(self, mock_save, client):
        response = client.post(
            '/api/anchors',
            json={'name': 'A', 'centroid': CENTROID, 'exclusions': [{'distance': 0.2}]},
        )
        assert response.status_code == 400
        assert response.get_json()['error'] == 'Each anchor exclusion needs a non-empty vector list'
        assert response.get_json()['error_code'] == 1003
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_exclusion_entry_with_non_numeric_distance_returns_400(self, mock_save, client):
        response = client.post(
            '/api/anchors',
            json={
                'name': 'A',
                'centroid': CENTROID,
                'exclusions': [{'vector': [0.0, 1.0], 'distance': 'far'}],
            },
        )
        assert response.status_code == 400
        assert response.get_json()['error'] == 'Anchor exclusion distance must be a number'
        assert response.get_json()['error_code'] == 1003
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_exclusion_entry_with_infinite_distance_returns_400(self, mock_save, client):
        response = client.post(
            '/api/anchors',
            json={
                'name': 'A',
                'centroid': CENTROID,
                'exclusions': [{'vector': [0.0, 1.0], 'distance': float('inf')}],
            },
        )
        assert response.status_code == 400
        assert response.get_json()['error'] == 'Anchor exclusion distance must be a finite number'
        assert response.get_json()['error_code'] == 1003
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_valid_exclusions_are_passed_to_save(self, mock_save, client):
        mock_save.return_value = {'id': 3, 'name': 'A'}
        response = client.post(
            '/api/anchors',
            json={
                'name': 'A',
                'centroid': CENTROID,
                'exclusions': [{'vector': [0.0, 1.0], 'distance': '0.3'}],
            },
        )
        assert response.status_code == 200
        assert mock_save.call_args.args == (
            'A',
            CENTROID,
            [{'vector': [0.0, 1.0], 'distance': 0.3}],
        )

    @patch('database.save_alchemy_anchor')
    def test_missing_exclusions_saves_none(self, mock_save, client):
        mock_save.return_value = {'id': 3, 'name': 'A'}
        response = client.post('/api/anchors', json={'name': 'A', 'centroid': CENTROID})
        assert response.status_code == 200
        assert mock_save.call_args.args == ('A', CENTROID, None)

    @patch('database.save_alchemy_anchor')
    def test_empty_exclusions_list_saves_none(self, mock_save, client):
        mock_save.return_value = {'id': 3, 'name': 'A'}
        response = client.post(
            '/api/anchors', json={'name': 'A', 'centroid': CENTROID, 'exclusions': []}
        )
        assert response.status_code == 200
        assert mock_save.call_args.args == ('A', CENTROID, None)


VECTOR_A = [1.0] + [0.0] * (DIM - 1)
VECTOR_B = [0.0, 1.0] + [0.0] * (DIM - 2)
SIZE_ERROR = f'Each anchor inclusion needs a vector of {DIM} finite numbers'


class TestCreateAnchorInclusionsValidation:
    @pytest.mark.parametrize(
        'inclusions, error',
        [
            ('nope', 'Anchor inclusions must be a list'),
            ([VECTOR_A], 'Each anchor inclusion must be an object'),
            ([{'weight': 1.0}], SIZE_ERROR),
            ([{'vector': [0.0, 1.0]}], SIZE_ERROR),
            ([{'vector': VECTOR_A + [0.0]}], SIZE_ERROR),
            ([{'vector': ['x'] + VECTOR_A[1:]}], SIZE_ERROR),
            ([{'vector': [True] + VECTOR_A[1:]}], SIZE_ERROR),
            ([{'vector': [float('nan')] + VECTOR_A[1:]}], SIZE_ERROR),
            ([{'vector': VECTOR_A, 'weight': 'heavy'}], 'Anchor inclusion weight must be a number'),
            (
                [{'vector': VECTOR_A, 'weight': -1.0}],
                'Anchor inclusion weight must be a finite number, 0 or greater',
            ),
            (
                [{'vector': VECTOR_A, 'weight': float('inf')}],
                'Anchor inclusion weight must be a finite number, 0 or greater',
            ),
            ([{'vector': VECTOR_A, 'seed': 'false'}], 'Anchor inclusion seed must be true or false'),
            ([{'vector': [10 ** 400] + VECTOR_A[1:]}], SIZE_ERROR),
            ([{'vector': VECTOR_A, 'weight': 10 ** 400}], 'Anchor inclusion weight must be a number'),
            ([{'vector': VECTOR_A, 'group': -1}], 'Anchor inclusion group must be a whole number, 0 or greater'),
            ([{'vector': VECTOR_A, 'group': True}], 'Anchor inclusion group must be a whole number, 0 or greater'),
            ([{'vector': VECTOR_A, 'signature': ['only title']}], 'Anchor inclusion signature must be a [title, artist] pair'),
        ],
    )
    @patch('database.save_alchemy_anchor')
    def test_malformed_inclusions_return_400(self, mock_save, client, inclusions, error):
        response = client.post(
            '/api/anchors',
            json={'name': 'A', 'centroid': CENTROID, 'inclusions': inclusions},
        )
        assert response.status_code == 400
        assert response.get_json()['error'] == error
        assert response.get_json()['error_code'] == 1003
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_valid_inclusions_are_stamped_and_passed_to_save(self, mock_save, client):
        mock_save.return_value = {'id': 3, 'name': 'A'}
        response = client.post(
            '/api/anchors',
            json={
                'name': 'A',
                'centroid': CENTROID,
                'inclusions': [
                    {'vector': VECTOR_A, 'weight': '2', 'seed': True, 'group': 0, 'signature': ['song', 'band']},
                    {'vector': VECTOR_B, 'group': 1},
                ],
                'inclusions_embedding': anchor_embedding_tag(),
            },
        )
        assert response.status_code == 200
        assert mock_save.call_args.args == ('A', CENTROID, None)
        assert mock_save.call_args.kwargs == {
            'inclusions': {
                **anchor_embedding_tag(),
                'points': [
                    {'vector': VECTOR_A, 'weight': 2.0, 'seed': True, 'group': 0, 'signature': ['song', 'band']},
                    {'vector': VECTOR_B, 'weight': 1.0, 'seed': False, 'group': 1, 'signature': None},
                ],
            }
        }

    @patch('database.save_alchemy_anchor')
    def test_inclusions_from_a_run_under_another_model_are_refused(self, mock_save, client):
        stale = {**anchor_embedding_tag(), 'embedding_model_sha256': 'another-model'}
        response = client.post(
            '/api/anchors',
            json={
                'name': 'A',
                'centroid': CENTROID,
                'inclusions': [{'vector': VECTOR_A}],
                'inclusions_embedding': stale,
            },
        )
        assert response.status_code == 400
        assert 'different embedding model' in response.get_json()['error']
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_missing_inclusions_saves_none(self, mock_save, client):
        mock_save.return_value = {'id': 3, 'name': 'A'}
        response = client.post('/api/anchors', json={'name': 'A', 'centroid': CENTROID})
        assert response.status_code == 200
        assert mock_save.call_args.kwargs == {'inclusions': None}


class TestRenameAnchorValidation:
    @patch('database.update_alchemy_anchor_name')
    def test_whitespace_only_name_returns_400(self, mock_update, client):
        response = client.put('/api/anchors/7', json={'name': '   '})
        assert response.status_code == 400
        assert response.get_json()['error'] == 'Anchor name is required'
        assert response.get_json()['error_code'] == 1003
        mock_update.assert_not_called()
