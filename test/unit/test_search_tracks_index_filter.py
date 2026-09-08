# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Song picker autocomplete: the optional index filter of /api/search_tracks.

Main Features:
* Without an index parameter every analysed song of the selected server is
  searched (no id filter reaches the backend)
* index=neural restricts the search to the songs that carry a neural
  fingerprint through a database-side clause (never a list of ids, which at a
  million songs would be shipped on every keystroke) and answers an empty
  list while that index is not loaded, never falling back to every song
* index=sem_grove filters on the SemGrove index's id set
* A filter that fails to load answers an empty list, not an error
"""

from unittest.mock import patch

import pytest
from flask import Flask

import app_ivf


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(app_ivf.ivf_bp)
    app.config['TESTING'] = True
    return app.test_client()


def _search(client, backend, **extra):
    with patch.object(app_ivf, 'search_tracks_unified', return_value=[]) as backend_call:
        response = client.get('/api/search_tracks', query_string={'search_query': 'act', 'start': 0, 'end': 20, **extra})
    backend.update(called=backend_call.called, kwargs=backend_call.call_args.kwargs if backend_call.called else None)
    return response


def test_without_an_index_parameter_no_id_filter_reaches_the_backend(client):
    backend = {}
    response = _search(client, backend)
    assert response.status_code == 200
    assert backend['called'] is True
    assert backend['kwargs']['item_id_filter'] is None


def test_index_neural_filters_in_the_database_and_never_ships_an_id_list(client, monkeypatch):
    from tasks import neural_fingerprint_index

    monkeypatch.setattr(neural_fingerprint_index, 'picker_where', lambda: neural_fingerprint_index.PICKER_WHERE)
    backend = {}
    response = _search(client, backend, index='neural')
    assert response.status_code == 200
    assert backend['kwargs']['item_id_filter'] is None
    assert backend['kwargs']['extra_where'] == neural_fingerprint_index.PICKER_WHERE
    assert 'neural_fingerprint IS NOT NULL' in backend['kwargs']['extra_where'][0]


def test_index_neural_answers_nothing_while_the_neural_index_is_not_loaded(client, monkeypatch):
    from tasks import neural_fingerprint_index

    monkeypatch.setattr(neural_fingerprint_index, 'picker_where', lambda: None)
    backend = {}
    response = _search(client, backend, index='neural')
    assert response.status_code == 200
    assert response.get_json() == []
    assert backend['called'] is False


def test_index_sem_grove_still_filters_on_the_sem_grove_index(client, monkeypatch):
    from tasks import sem_grove_manager

    monkeypatch.setattr(sem_grove_manager, 'get_sem_grove_item_ids', lambda: {'fp_s'})
    backend = {}
    response = _search(client, backend, index='sem_grove')
    assert response.status_code == 200
    assert backend['kwargs']['item_id_filter'] == {'fp_s'}


def test_a_filter_that_fails_to_load_answers_an_empty_list(client, monkeypatch):
    from tasks import neural_fingerprint_index

    def broken():
        raise RuntimeError('pack unreadable')

    monkeypatch.setattr(neural_fingerprint_index, 'picker_where', broken)
    backend = {}
    response = _search(client, backend, index='neural')
    assert response.status_code == 200
    assert response.get_json() == []
    assert backend['called'] is False
