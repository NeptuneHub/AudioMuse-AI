"""Request-validation coverage for the chat playlist endpoints.

Both /api/chatPlaylist and /api/chatPlaylistStream share the guard:
    if not isinstance(data, dict)
       or not isinstance(data.get('userInput'), str)
       or not data['userInput'].strip():
        return 400

These tests assert that a body which is NOT a dict (list, number, string,
null), a missing userInput key, a non-string userInput, and an empty /
whitespace-only userInput all return HTTP 400 with the missing-userInput
error -- for BOTH endpoints -- while a valid body proceeds past validation
into the (patched-out) pipeline.
"""
import sys
import types
import json
from unittest.mock import patch

import pytest
from flask import Flask


def _ensure_flasgger():
    try:
        import flasgger  # noqa: F401
        return
    except ImportError:
        pass
    fake = types.ModuleType('flasgger')

    def swag_from(*a, **k):
        def deco(f):
            return f
        return deco

    fake.swag_from = swag_from
    fake.Swagger = lambda *a, **k: None
    sys.modules['flasgger'] = fake


@pytest.fixture
def app_chat_mod():
    _ensure_flasgger()
    import app_chat
    return app_chat


@pytest.fixture
def client(app_chat_mod):
    app = Flask(__name__)
    app.register_blueprint(app_chat_mod.chat_bp)
    app.config['TESTING'] = True
    return app.test_client()


ENDPOINTS = ['/api/chatPlaylist', '/api/chatPlaylistStream']

# JSON bodies that are valid JSON but NOT a dict -> must 400.
NON_DICT_BODIES = [
    ('list', '[]'),
    ('number', '5'),
    ('string', '"make me a playlist"'),
    ('null', 'null'),
]


def _post_raw(client, path, raw_body):
    return client.post(path, data=raw_body, content_type='application/json')


def _assert_missing_user_input(resp):
    assert resp.status_code == 400
    payload = json.loads(resp.get_data(as_text=True))
    assert payload.get('error') == 'Missing userInput in request'


class TestNonDictBodyRejected:
    @pytest.mark.parametrize('path', ENDPOINTS)
    @pytest.mark.parametrize('label,raw', NON_DICT_BODIES)
    def test_non_dict_body_returns_400(self, client, path, label, raw):
        resp = _post_raw(client, path, raw)
        _assert_missing_user_input(resp)


class TestInvalidUserInputRejected:
    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_missing_user_input_key(self, client, path):
        resp = client.post(
            path, json={'ai_provider': 'NONE'},
            content_type='application/json')
        _assert_missing_user_input(resp)

    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_user_input_not_a_string(self, client, path):
        resp = client.post(
            path, json={'userInput': 123},
            content_type='application/json')
        _assert_missing_user_input(resp)

    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_user_input_empty_string(self, client, path):
        resp = client.post(
            path, json={'userInput': ''},
            content_type='application/json')
        _assert_missing_user_input(resp)

    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_user_input_whitespace_only(self, client, path):
        resp = client.post(
            path, json={'userInput': '   \t  '},
            content_type='application/json')
        _assert_missing_user_input(resp)


class TestValidBodyProceeds:
    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_valid_body_passes_validation(self, app_chat_mod, client, path):
        # Patch the pipeline so the real AI stack never runs; we only assert
        # that a valid body gets PAST the 400 guard and into the pipeline.
        called = {'run': 0}

        def _fake_run(data, log_messages):
            called['run'] += 1
            yield from ()  # generator marker; never yields
            return ({'message': 'stub', 'query_results': None}, 200)

        with patch.object(app_chat_mod, '_run_chat_pipeline', _fake_run):
            resp = client.post(
                path, json={'userInput': 'make me a playlist'},
                content_type='application/json')
            # The streaming endpoint builds a lazy generator; consume the body
            # inside the patch so _run_chat_pipeline actually runs.
            body = resp.get_data(as_text=True)

        assert resp.status_code != 400
        assert called['run'] == 1
        assert body is not None
