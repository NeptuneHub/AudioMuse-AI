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
        resp = client.post(path, json={'ai_provider': 'NONE'}, content_type='application/json')
        _assert_missing_user_input(resp)

    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_user_input_not_a_string(self, client, path):
        resp = client.post(path, json={'userInput': 123}, content_type='application/json')
        _assert_missing_user_input(resp)

    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_user_input_empty_string(self, client, path):
        resp = client.post(path, json={'userInput': ''}, content_type='application/json')
        _assert_missing_user_input(resp)

    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_user_input_whitespace_only(self, client, path):
        resp = client.post(path, json={'userInput': '   \t  '}, content_type='application/json')
        _assert_missing_user_input(resp)


class TestValidBodyProceeds:
    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_valid_body_passes_validation(self, app_chat_mod, client, path):
        called = {'run': 0}

        def _fake_run(data, log_messages):
            called['run'] += 1
            yield from ()
            return ({'message': 'stub', 'query_results': None}, 200)

        with patch.object(app_chat_mod, '_run_chat_pipeline', _fake_run):
            resp = client.post(
                path, json={'userInput': 'make me a playlist'}, content_type='application/json'
            )
            body = resp.get_data(as_text=True)

        assert resp.status_code != 400
        assert called['run'] == 1
        assert body is not None
