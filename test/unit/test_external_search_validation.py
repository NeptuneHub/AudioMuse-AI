# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Search-query validation on the external blueprint's /search route.

Drives app_external.external_bp with a Flask test client to check when a request
reaches the unified search backend versus short-circuiting to an empty list.

Main Features:
* Missing or explicitly empty search_query returns [] without calling the backend
* One-character and longer queries pass through to search_tracks_unified verbatim
* Legacy title/artist params are combined into a single "artist title" query
* /get_embedding decodes both stored vectors, so the row serializes to JSON
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


def _import_app_external():
    if 'app_external' in sys.modules:
        return sys.modules['app_external']
    fake_vm = types.ModuleType('tasks.ivf_manager')
    fake_vm.search_tracks_unified = MagicMock(return_value=[])
    stubs = {'tasks.ivf_manager': fake_vm}
    if 'tasks' not in sys.modules:
        stubs['tasks'] = types.ModuleType('tasks')
    with patch.dict(sys.modules, stubs):
        import app_external
    return app_external


@pytest.fixture
def ext():
    return _import_app_external()


@pytest.fixture
def client(ext):
    app = Flask(__name__)
    app.register_blueprint(ext.external_bp)
    app.config['TESTING'] = True
    return app.test_client()


class TestSearchQueryValidation:
    def test_missing_query_returns_empty_list(self, ext, client):
        with patch.object(ext, 'search_tracks_unified') as backend:
            resp = client.get('/search')
        assert resp.status_code == 200
        assert resp.get_json() == []
        backend.assert_not_called()

    def test_explicit_empty_query_returns_empty_list(self, ext, client):
        with patch.object(ext, 'search_tracks_unified') as backend:
            resp = client.get('/search', query_string={'search_query': ''})
        assert resp.status_code == 200
        assert resp.get_json() == []
        backend.assert_not_called()

    def test_one_char_query_reaches_backend(self, ext, client):
        results = [{'item_id': 'id-1', 'title': 'Song', 'author': 'Artist'}]
        with patch.object(ext, 'search_tracks_unified', return_value=results) as backend:
            resp = client.get('/search', query_string={'search_query': 'a'})
        assert resp.status_code == 200
        assert resp.get_json() == results
        assert backend.call_count == 1
        assert backend.call_args.args[0] == 'a'

    def test_valid_query_reaches_backend_and_returns_its_value(self, ext, client):
        results = [{'item_id': 'id-1', 'title': 'Song', 'author': 'Artist'}]
        with patch.object(ext, 'search_tracks_unified', return_value=results) as backend:
            resp = client.get('/search', query_string={'search_query': 'abc'})
        assert resp.status_code == 200
        assert resp.get_json() == results
        assert backend.call_count == 1
        assert backend.call_args.args[0] == 'abc'

    def test_legacy_title_artist_params_build_query(self, ext, client):
        with patch.object(ext, 'search_tracks_unified', return_value=[]) as backend:
            resp = client.get('/search', query_string={'title': 'Song 1', 'artist': 'Artist A'})
        assert resp.status_code == 200
        assert resp.get_json() == []
        assert backend.call_count == 1
        assert backend.call_args.args[0] == 'Artist A Song 1'


class TestGetEmbeddingDecodesBothVectors:
    def test_embedding_and_poincare_vectors_come_back_as_float_lists(self, ext, client):
        import numpy as np

        row = {
            'item_id': 'fp_2abc',
            'embedding': memoryview(np.asarray([0.5, -1.0], dtype=np.float32).tobytes()),
            'poincare_embedding': memoryview(np.asarray([0.25, 0.125], dtype=np.float32).tobytes()),
            'hyperbolic_radius': 0.3,
        }
        cursor = MagicMock()
        cursor.fetchone.return_value = row
        db = MagicMock()
        db.cursor.return_value.__enter__.return_value = cursor
        fake_database = types.ModuleType('database')
        fake_database.get_db = lambda: db
        fake_context = types.ModuleType('app_server_context')
        fake_context.provider_echo_id = lambda raw_id: raw_id
        fake_context.resolve_input_item_id = lambda raw_id: 'fp_2abc'
        with patch.dict(sys.modules, {'database': fake_database, 'app_server_context': fake_context}):
            resp = client.get('/get_embedding', query_string={'id': 'jellyfin-id'})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['embedding'] == [0.5, -1.0]
        assert body['poincare_embedding'] == [0.25, 0.125]
        assert body['item_id'] == 'jellyfin-id'
