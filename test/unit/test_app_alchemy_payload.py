# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the app_alchemy alchemy-request payload validation.

Posts item payloads to the /api/alchemy endpoint to confirm operation and
id validation runs before the request reaches the alchemy engine.

Main Features:
* Payloads with no ADD operation and ADD items missing an id return 400.
* ADD items without ids are filtered out before dispatch.
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


class TestAlchemyApiPayloadValidation:
    def test_items_without_any_add_op_returns_400(self, client):
        response = client.post('/api/alchemy', json={'items': [{'id': 'song-1', 'op': 'SUBTRACT'}]})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Invalid request'}

    def test_add_item_missing_id_returns_400(self, client):
        response = client.post('/api/alchemy', json={'items': [{'op': 'ADD', 'type': 'song'}]})
        assert response.status_code == 400
        assert response.get_json() == {'error': 'Invalid request'}

    @patch('app_alchemy.song_alchemy')
    def test_add_items_without_id_are_filtered_before_dispatch(self, mock_alchemy, client):
        mock_alchemy.side_effect = ValueError('At least one item must be in the ADD set')
        response = client.post(
            '/api/alchemy', json={'items': [{'op': 'ADD'}, {'op': 'ADD', 'id': ''}]}
        )
        assert response.status_code == 400
        assert mock_alchemy.call_args.kwargs['add_items'] == []
        assert mock_alchemy.call_args.kwargs['subtract_items'] == []
