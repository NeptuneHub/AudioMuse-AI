"""Validation tests for the external /search endpoint in app_external.

The heavy ``tasks.ivf_manager`` import is stubbed at module load so the
endpoint checks stay fast and hermetic; the backend search function is patched
per test so no database is touched.
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
        backend.assert_called_once_with('a')

    def test_valid_query_reaches_backend_and_returns_its_value(self, ext, client):
        results = [{'item_id': 'id-1', 'title': 'Song', 'author': 'Artist'}]
        with patch.object(ext, 'search_tracks_unified', return_value=results) as backend:
            resp = client.get('/search', query_string={'search_query': 'abc'})
        assert resp.status_code == 200
        assert resp.get_json() == results
        backend.assert_called_once_with('abc')

    def test_legacy_title_artist_params_build_query(self, ext, client):
        with patch.object(ext, 'search_tracks_unified', return_value=[]) as backend:
            resp = client.get('/search', query_string={'title': 'Hello', 'artist': 'Adele'})
        assert resp.status_code == 200
        assert resp.get_json() == []
        backend.assert_called_once_with('Adele Hello')
