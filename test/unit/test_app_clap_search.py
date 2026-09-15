# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""CLAP search blueprint: validation of the requested result limit.

Main Features:
* A missing or null limit falls back to CLAP_SEARCH_DEFAULT_LIMIT
* A non-numeric limit (a word, a list, an object, an overflowing float)
  answers 400 with the invalid-request code, never 500
* A numeric limit is passed through, floored at 1
"""

import pytest
from flask import Flask

from app_clap_search import clap_search_bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(clap_search_bp)
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture
def search_calls(monkeypatch):
    import app_helper
    import app_server_context
    import config
    import tasks.clap_text_search as clap_text_search

    calls = []

    def fake_search(query, limit=None, steering=None):
        calls.append(limit)
        return [{'item_id': 'song-1', 'title': 'T', 'author': 'A', 'similarity': 0.9}]

    monkeypatch.setattr(config, 'CLAP_ENABLED', True)
    monkeypatch.setattr(config, 'CLAP_SEARCH_DEFAULT_LIMIT', 37)
    monkeypatch.setattr(app_server_context, 'resolve_request_server_id', lambda data=None: None)
    monkeypatch.setattr(
        app_server_context,
        'scope_results',
        lambda rows, requested_n=None, id_key='item_id', translate=True: rows,
    )
    monkeypatch.setattr(app_helper, 'attach_song_features', lambda rows, id_key='item_id': rows)
    monkeypatch.setattr(clap_text_search, 'is_clap_cache_loaded', lambda: True)
    monkeypatch.setattr(clap_text_search, 'search_by_text', fake_search)
    return calls


@pytest.mark.parametrize('body', [{'query': 'upbeat'}, {'query': 'upbeat', 'limit': None}])
def test_missing_or_null_limit_uses_the_default(client, search_calls, body):
    response = client.post('/api/clap/search', json=body)
    assert response.status_code == 200
    assert search_calls == [37]


@pytest.mark.parametrize('limit', ['abc', [5], {'n': 5}, 1e400])
def test_non_numeric_limit_answers_400_not_500(client, search_calls, limit):
    response = client.post('/api/clap/search', json={'query': 'upbeat', 'limit': limit})
    assert response.status_code == 400
    assert response.get_json()['error_code'] == 1003
    assert search_calls == []


@pytest.mark.parametrize('limit, expected', [(12, 12), ('8', 8), (0, 1)])
def test_numeric_limit_is_passed_through_and_floored_at_one(client, search_calls, limit, expected):
    response = client.post('/api/clap/search', json={'query': 'upbeat', 'limit': limit})
    assert response.status_code == 200
    assert search_calls == [expected]
