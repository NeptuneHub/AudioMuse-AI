# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The lyrics search endpoints refuse a bad body instead of answering 500.

A live fuzz of the running instance found POST /api/lyrics/search/text answering
500 for a query that was not a string: the handler did `(data.get('query') or
'').strip()`, and an int or a list is truthy, so .strip() raised AttributeError
into the generic 500. Its sibling, the axis search, already guarded the same
input with isinstance, and the limit right below it already answered 400, so the
500 was the one gap in a handler that otherwise keeps the contract.

Main Features:
* A query that is not a string is a 400 with a clear message, never a 500
* A missing, empty or whitespace-only query keeps answering 400
* A real string query still reaches the search
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app_lyrics import lyrics_search_bp  # noqa: E402


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(lyrics_search_bp)
    app.config['TESTING'] = True
    return app.test_client()


@pytest.fixture
def searched(monkeypatch):
    import config

    monkeypatch.setattr(config, 'LYRICS_ENABLED', True, raising=False)
    calls = []

    def _search(query, limit=10):
        calls.append((query, limit))
        return [{'item_id': 'a', 'title': 'T', 'author': 'A'}]

    module = pytest.importorskip('tasks.lyrics_manager')
    monkeypatch.setattr(module, 'search_by_text', _search, raising=False)
    return calls


class TestTheTextSearchBodyContract:
    @pytest.mark.parametrize('query', [5, [], ['a'], {'a': 1}, 3.5, True])
    def test_a_query_that_is_not_a_string_is_refused_with_400(self, client, searched, query):
        with patch('app_server_context.resolve_request_server_id', return_value=None):
            response = client.post('/api/lyrics/search/text', json={'query': query})

        assert response.status_code == 400, (
            'a non-string query used to reach .strip() and raise into the generic '
            '500 handler; a bad body must be refused, not crash the request'
        )
        assert 'query' in response.get_json()['error'].lower()
        assert not searched, 'a refused body must never reach the search'

    @pytest.mark.parametrize('query', [None, '', '   '])
    def test_an_empty_query_is_still_refused_with_400(self, client, searched, query):
        with patch('app_server_context.resolve_request_server_id', return_value=None):
            response = client.post('/api/lyrics/search/text', json={'query': query})

        assert response.status_code == 400
        assert not searched

    def test_a_real_query_still_reaches_the_search(self, client, searched):
        with patch('app_server_context.resolve_request_server_id', return_value=None):
            response = client.post('/api/lyrics/search/text', json={'query': '  love song  '})

        assert response.status_code == 200
        assert searched and searched[0][0] == 'love song', (
            'the guard must not change what a valid query means: it is still stripped'
        )
