# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Setup-wizard Plex account-linking (plex.tv/link) proxy routes.

Covers the two thin proxy endpoints that let the wizard mint a Plex PIN and
poll for the resulting account token without the browser hitting plex.tv
directly (plex.tv sends no CORS headers).

Main Features:
* PIN creation forwards client_id as X-Plex-Client-Identifier and returns id/code
* Missing client_id and non-numeric PIN ids are rejected before any outbound call
* Poll returns the authToken when linked and null while still pending
* plex.tv failures surface as a 502 rather than leaking the exception
"""

import app_setup


class _FakeResponse:
    def __init__(self, payload, raise_exc=None):
        self._payload = payload
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self):
        return self._payload


def _create(json_body):
    with app_setup.app.test_request_context(
        '/api/setup/plex/pin', method='POST', json=json_body
    ):
        resp, status = app_setup.setup_plex_pin_create()
        return resp.get_json(), status


def _poll(pin_id, query=''):
    with app_setup.app.test_request_context(f'/api/setup/plex/pin/{pin_id}{query}'):
        result = app_setup.setup_plex_pin_poll(pin_id)
        resp, status = result if isinstance(result, tuple) else (result, 200)
        return resp.get_json(), status


class TestCreatePin:
    def test_returns_id_and_code(self, monkeypatch):
        captured = {}

        def fake_post(url, headers=None, data=None, timeout=None):
            captured['url'] = url
            captured['headers'] = headers
            return _FakeResponse({'id': 42, 'code': 'AB4C'})

        monkeypatch.setattr(app_setup.requests, 'post', fake_post)
        body, status = _create({'client_id': 'audiomuse-xyz'})

        assert status == 200
        assert body == {'id': 42, 'code': 'AB4C'}
        assert captured['url'] == app_setup.PLEX_PIN_API_BASE
        assert captured['headers']['X-Plex-Client-Identifier'] == 'audiomuse-xyz'

    def test_missing_client_id_is_400(self, monkeypatch):
        def fail_post(*a, **k):
            raise AssertionError('should not call plex.tv without a client_id')

        monkeypatch.setattr(app_setup.requests, 'post', fail_post)
        body, status = _create({})
        assert status == 400
        assert 'client_id' in body['error']

    def test_plex_unreachable_is_502(self, monkeypatch):
        def boom_post(*a, **k):
            raise RuntimeError('connection refused')

        monkeypatch.setattr(app_setup.requests, 'post', boom_post)
        body, status = _create({'client_id': 'x'})
        assert status == 502
        assert 'error' in body


class TestPollPin:
    def test_returns_token_when_linked(self, monkeypatch):
        def fake_get(url, headers=None, timeout=None):
            assert url.endswith('/123')
            return _FakeResponse({'authToken': 'plex-token-123'})

        monkeypatch.setattr(app_setup.requests, 'get', fake_get)
        body, status = _poll('123', '?client_id=x')
        assert status == 200
        assert body == {'token': 'plex-token-123'}

    def test_token_null_while_pending(self, monkeypatch):
        monkeypatch.setattr(
            app_setup.requests, 'get',
            lambda *a, **k: _FakeResponse({'authToken': None}),
        )
        body, status = _poll('123', '?client_id=x')
        assert status == 200
        assert body == {'token': None}

    def test_non_numeric_pin_is_400(self, monkeypatch):
        def fail_get(*a, **k):
            raise AssertionError('should not call plex.tv for a non-numeric id')

        monkeypatch.setattr(app_setup.requests, 'get', fail_get)
        body, status = _poll('../secrets', '?client_id=x')
        assert status == 400

    def test_missing_client_id_is_400(self, monkeypatch):
        def fail_get(*a, **k):
            raise AssertionError('should not call plex.tv without a client_id')

        monkeypatch.setattr(app_setup.requests, 'get', fail_get)
        body, status = _poll('123')
        assert status == 400
