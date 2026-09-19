# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The read-only external API: score, embedding and search by provider id.

Third-party clients address tracks by their media-server id; the endpoints
echo that id back, never the catalogue id, and decode the stored embedding
blobs into plain lists.

Main Features:
* /external/get_score and /external/get_embedding answer for a provider id
  and echo it
* the embedding is a 200-float list with its hyperbolic companions
* a missing id is 400, an unknown id 404, and search returns provider ids
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids

pytestmark = pytest.mark.e2e

SCORE_KEYS = ('title', 'tempo', 'key', 'scale', 'mood_vector', 'energy')


def test_get_score(stack, api, library, analyzed_library):
    pid = library.pid('A03')
    body = api.json('GET', f'/external/get_score?id={pid}')
    assert_no_fp_ids(body)
    assert body['item_id'] == pid
    for key in SCORE_KEYS:
        assert key in body, (key, body)
    assert body['title'] == library.track('A03').title


def test_get_embedding(stack, api, library, analyzed_library):
    pid = library.pid('A03')
    body = api.json('GET', f'/external/get_embedding?id={pid}')
    assert_no_fp_ids(body)
    assert body['item_id'] == pid
    assert isinstance(body['embedding'], list) and len(body['embedding']) == 200
    assert all(isinstance(v, (int, float)) for v in body['embedding'])
    assert 'poincare_embedding' in body and 'hyperbolic_radius' in body


def test_errors(stack, api, analyzed_library):
    assert api.get('/external/get_score').status_code == 400
    assert api.get('/external/get_embedding').status_code == 400
    assert api.get('/external/get_score?id=nope').status_code == 404
    assert api.get('/external/get_embedding?id=nope').status_code == 404


def test_search(stack, api, library, analyzed_library):
    results = api.json('GET', '/external/search?search_query=Variatio')
    assert_no_fp_ids(results)
    assert results, 'search must find the Goldberg clips'
    ids = {r['item_id'] for r in results}
    assert library.pid('A02') in ids or library.pid('A01') in ids or library.pid('F01') in ids
    for row in results:
        assert row.get('title') and row['item_id']
    assert api.json('GET', '/external/search') == []
