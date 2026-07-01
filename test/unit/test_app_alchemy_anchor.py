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
"""

import pytest
from unittest.mock import patch
from flask import Flask

from app_alchemy import alchemy_bp


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
        response = client.post('/api/anchors', json={'name': '   ', 'centroid': [0.1, 0.2]})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Anchor name is required'}
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_non_list_centroid_returns_400(self, mock_save, client):
        response = client.post('/api/anchors', json={'name': 'My Anchor', 'centroid': 'not-a-list'})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Anchor centroid is required and must be a list'}
        mock_save.assert_not_called()

    @patch('database.save_alchemy_anchor')
    def test_empty_list_centroid_returns_400(self, mock_save, client):
        response = client.post('/api/anchors', json={'name': 'My Anchor', 'centroid': []})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Anchor centroid is required and must be a list'}
        mock_save.assert_not_called()


class TestRenameAnchorValidation:
    @patch('database.update_alchemy_anchor_name')
    def test_whitespace_only_name_returns_400(self, mock_update, client):
        response = client.put('/api/anchors/7', json={'name': '   '})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Anchor name is required'}
        mock_update.assert_not_called()
