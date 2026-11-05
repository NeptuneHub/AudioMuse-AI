# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for chat-endpoint request-body validation.

Posts malformed and valid bodies to the chat routes to confirm the shared
user-input validation rejects bad requests and lets valid ones proceed.

Main Features:
* Non-dict bodies and missing, non-string, empty, or blank user input return 400.
* User input longer than INSTANT_PLAYLIST_MAX_INPUT_CHARS returns 400 before the
  pipeline runs; the chat page gets the same limit for its textarea.
* A well-formed body passes validation into the handler.
"""

import pathlib
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


def _counting_pipeline(called):
    def _fake_run(data, log_messages):
        called['run'] += 1
        yield from ()
        return ({'message': 'stub', 'query_results': None}, 200)

    return _fake_run


class TestUserInputLengthLimit:
    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_input_at_the_limit_passes(self, app_chat_mod, client, path, monkeypatch):
        monkeypatch.setattr(app_chat_mod.config, 'INSTANT_PLAYLIST_MAX_INPUT_CHARS', 20)
        called = {'run': 0}
        with patch.object(app_chat_mod, '_run_chat_pipeline', _counting_pipeline(called)):
            resp = client.post(path, json={'userInput': 'x' * 20})
            resp.get_data(as_text=True)

        assert resp.status_code != 400
        assert called['run'] == 1

    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_input_over_the_limit_is_rejected(self, app_chat_mod, client, path, monkeypatch):
        monkeypatch.setattr(app_chat_mod.config, 'INSTANT_PLAYLIST_MAX_INPUT_CHARS', 20)
        called = {'run': 0}
        with patch.object(app_chat_mod, '_run_chat_pipeline', _counting_pipeline(called)):
            resp = client.post(path, json={'userInput': 'x' * 21})

        assert resp.status_code == 400
        payload = json.loads(resp.get_data(as_text=True))
        assert payload.get('error') == 'userInput is too long (max 20 characters)'
        assert called['run'] == 0

    @pytest.mark.parametrize('path', ENDPOINTS)
    def test_crafted_slow_input_is_rejected_before_the_pipeline(self, app_chat_mod, client, path):
        # ~205 KB of key phrases: the payload of the quadratic hint-parsing DoS report.
        called = {'run': 0}
        with patch.object(app_chat_mod, '_run_chat_pipeline', _counting_pipeline(called)):
            resp = client.post(path, json={'userInput': 'A major ' * 26_000})

        assert resp.status_code == 400
        assert called['run'] == 0

    def test_default_limit_fits_a_long_real_prompt(self, app_chat_mod):
        # The longest real prompt found in the wild (a Spotify "Prompted Playlist") is
        # ~1,560 characters; the default must leave room for it.
        assert app_chat_mod.config.INSTANT_PLAYLIST_MAX_INPUT_CHARS >= 1600

    def test_chat_page_gets_the_limit(self, app_chat_mod, monkeypatch):
        captured = {}
        monkeypatch.setattr(app_chat_mod, 'render_template', lambda name, **kw: captured.update(kw) or '')
        monkeypatch.setattr(app_chat_mod.config, 'INSTANT_PLAYLIST_MAX_INPUT_CHARS', 1234)
        app_chat_mod.chat_home()
        assert captured['instant_playlist_max_input_chars'] == 1234

    def test_chat_page_textarea_and_submit_enforce_the_limit(self):
        page = (pathlib.Path(__file__).resolve().parents[2] / 'templates' / 'chat.html').read_text(
            encoding='utf-8'
        )
        assert 'maxlength="{{ instant_playlist_max_input_chars }}"' in page
        assert 'const maxInputChars = {{ instant_playlist_max_input_chars }};' in page
        assert 'userInput.value.length > maxInputChars' in page
