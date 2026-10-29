# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""CLAP search blueprint: request validation of the text and song searches.

Main Features:
* A missing or null limit falls back to CLAP_SEARCH_DEFAULT_LIMIT
* A non-numeric limit (a word, a list, an object, an overflowing float)
  answers 400 with the invalid-request code, never 500
* A numeric limit is passed through, floored at 1
* Song search: the caller's provider id is resolved before the search, a
  missing id or a body that is not an object is 400, the optional artist limit
  and radius switches are passed through like /api/similar_tracks (omitted means
  the server default), a seed without an embedding and a
  seed with nothing similar left are distinct 404s, an unloaded index is 503,
  an internal error is 500 rather than a client error, and no canonical fp_
  id ever appears in the answer
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


@pytest.fixture
def song_calls(monkeypatch):
    import app_helper
    import app_server_context
    import config
    import tasks.clap_text_search as clap_text_search

    calls = []
    state = {'loaded': True, 'results': None, 'raise': None, 'scoped_away': False}

    def fake_song_search(item_id, limit=None, eliminate_duplicates=None, radius_similarity=None):
        calls.append((item_id, limit, eliminate_duplicates, radius_similarity))
        if state['raise'] is not None:
            raise state['raise']
        if state['results'] is not None:
            return state['results']
        return [
            {'item_id': 'fp_0123456789abcdef', 'title': 'T1', 'author': 'A', 'similarity': 0.9},
            {'item_id': 'fp_fedcba9876543210', 'title': 'T2', 'author': 'B', 'similarity': 0.8},
        ]

    def translating_scope(rows, requested_n=None, id_key='item_id', translate=True):
        if state['scoped_away']:
            return []
        out = [dict(r, item_id='provider-' + r['item_id'][3:7]) for r in rows]
        return out if requested_n is None else out[:requested_n]

    monkeypatch.setattr(config, 'CLAP_ENABLED', True)
    monkeypatch.setattr(config, 'CLAP_SEARCH_DEFAULT_LIMIT', 37)
    monkeypatch.setattr(app_server_context, 'resolve_request_server_id', lambda data=None: None)
    monkeypatch.setattr(
        app_server_context,
        'resolve_input_item_id',
        lambda raw_id, data=None: 'fp_seed' if raw_id == 'provider-seed' else raw_id,
    )
    monkeypatch.setattr(app_server_context, 'scope_results', translating_scope)
    monkeypatch.setattr(app_helper, 'attach_song_features', lambda rows, id_key='item_id': rows)
    monkeypatch.setattr(clap_text_search, 'is_clap_cache_loaded', lambda: state['loaded'])
    monkeypatch.setattr(clap_text_search, 'search_by_song', fake_song_search)
    return calls, state


def test_song_search_resolves_the_provider_id_and_uses_the_default_limit(client, song_calls):
    calls, _ = song_calls
    response = client.post('/api/clap/search_by_song', json={'item_id': 'provider-seed'})
    assert response.status_code == 200
    assert calls == [('fp_seed', 37, None, None)]
    body = response.get_json()
    assert body['count'] == 2
    assert [r['similarity'] for r in body['results']] == [0.9, 0.8]


def test_no_response_of_the_song_search_ever_carries_a_canonical_id(client, song_calls):
    response = client.post('/api/clap/search_by_song', json={'item_id': 'provider-seed', 'limit': 5})
    assert response.status_code == 200
    assert 'fp_' not in response.get_data(as_text=True)


@pytest.mark.parametrize('body', [{}, {'item_id': ''}, {'item_id': '   '}, {'item_id': 7}, {'item_id': None}])
def test_song_search_without_a_seed_answers_400(client, song_calls, body):
    calls, _ = song_calls
    response = client.post('/api/clap/search_by_song', json=body)
    assert response.status_code == 400
    assert response.get_json()['error_code'] == 1003
    assert calls == []


@pytest.mark.parametrize('body', [['provider-seed'], 'provider-seed', 5, True])
def test_song_search_with_a_body_that_is_not_an_object_answers_400_not_500(client, song_calls, body):
    calls, _ = song_calls
    response = client.post('/api/clap/search_by_song', json=body)
    assert response.status_code == 400
    assert response.get_json()['error_code'] == 1003
    assert calls == []


@pytest.mark.parametrize('limit', ['abc', [5], {'n': 5}, 1e400])
def test_song_search_non_numeric_limit_answers_400_not_500(client, song_calls, limit):
    calls, _ = song_calls
    response = client.post('/api/clap/search_by_song', json={'item_id': 'provider-seed', 'limit': limit})
    assert response.status_code == 400
    assert calls == []


@pytest.mark.parametrize('limit, expected', [(12, 12), ('8', 8), (0, 1)])
def test_song_search_limit_is_passed_through_and_floored_at_one(client, song_calls, limit, expected):
    calls, _ = song_calls
    response = client.post('/api/clap/search_by_song', json={'item_id': 'provider-seed', 'limit': limit})
    assert response.status_code == 200
    assert calls == [('fp_seed', expected, None, None)]


@pytest.mark.parametrize('value, expected', [(True, True), (False, False), ('false', False), (' TRUE ', True)])
def test_song_search_passes_the_artist_limit_switch_like_similar_tracks(client, song_calls, value, expected):
    calls, _ = song_calls
    response = client.post(
        '/api/clap/search_by_song', json={'item_id': 'provider-seed', 'eliminate_duplicates': value}
    )
    assert response.status_code == 200
    assert calls == [('fp_seed', 37, expected, None)]


@pytest.mark.parametrize('value', ['maybe', 1, [True], {'on': True}])
def test_song_search_with_an_invalid_artist_limit_switch_answers_400(client, song_calls, value):
    calls, _ = song_calls
    response = client.post(
        '/api/clap/search_by_song', json={'item_id': 'provider-seed', 'eliminate_duplicates': value}
    )
    assert response.status_code == 400
    assert calls == []


@pytest.mark.parametrize('value, expected', [(True, True), (False, False), ('false', False), (' TRUE ', True)])
def test_song_search_passes_the_radius_switch_like_similar_tracks(client, song_calls, value, expected):
    calls, _ = song_calls
    response = client.post(
        '/api/clap/search_by_song', json={'item_id': 'provider-seed', 'radius_similarity': value}
    )
    assert response.status_code == 200
    assert calls == [('fp_seed', 37, None, expected)]


@pytest.mark.parametrize('value', ['maybe', 1, [True], {'on': True}])
def test_song_search_with_an_invalid_radius_switch_answers_400(client, song_calls, value):
    calls, _ = song_calls
    response = client.post(
        '/api/clap/search_by_song', json={'item_id': 'provider-seed', 'radius_similarity': value}
    )
    assert response.status_code == 400
    assert '"radius_similarity" must be true or false' in response.get_json()['error']
    assert calls == []


def test_a_seed_without_a_dclap_embedding_answers_404_saying_so(client, song_calls):
    from tasks.clap_text_search import SongSeedNotFound

    _, state = song_calls
    state['raise'] = SongSeedNotFound('no embedding')
    response = client.post('/api/clap/search_by_song', json={'item_id': 'provider-seed'})
    assert response.status_code == 404
    assert 'no DCLAP embedding' in response.get_json()['error']
    assert response.get_json()['results'] == []


@pytest.mark.parametrize('scoped_away', [False, True], ids=['nothing-similar', 'nothing-on-this-server'])
def test_a_seed_with_no_similar_song_left_answers_404_without_blaming_the_analysis(client, song_calls, scoped_away):
    _, state = song_calls
    state['results'] = [] if not scoped_away else None
    state['scoped_away'] = scoped_away
    response = client.post('/api/clap/search_by_song', json={'item_id': 'provider-seed'})
    assert response.status_code == 404
    assert 'embedding' not in response.get_json()['error']
    assert response.get_json()['results'] == []


def test_song_search_before_the_index_is_loaded_answers_503(client, song_calls):
    calls, state = song_calls
    state['loaded'] = False
    response = client.post('/api/clap/search_by_song', json={'item_id': 'provider-seed'})
    assert response.status_code == 503
    assert calls == []


def test_an_index_that_unloads_during_the_search_answers_503(client, song_calls):
    from tasks.clap_text_search import ClapIndexUnavailable

    _, state = song_calls
    state['raise'] = ClapIndexUnavailable('gone')
    response = client.post('/api/clap/search_by_song', json={'item_id': 'provider-seed'})
    assert response.status_code == 503


def test_an_internal_value_error_is_a_server_error_not_a_client_error(client, song_calls):
    _, state = song_calls
    state['raise'] = ValueError('shapes (512,) and (256,) not aligned')
    response = client.post('/api/clap/search_by_song', json={'item_id': 'provider-seed'})
    assert response.status_code == 500


def test_song_search_with_clap_disabled_answers_400(client, song_calls, monkeypatch):
    import config

    calls, _ = song_calls
    monkeypatch.setattr(config, 'CLAP_ENABLED', False)
    response = client.post('/api/clap/search_by_song', json={'item_id': 'provider-seed'})
    assert response.status_code == 400
    assert calls == []
