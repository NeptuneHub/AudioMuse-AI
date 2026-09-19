# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The hyperbolic explorer: similar, roots, niche, journey and the tree.

The Poincare projection built at index time answers similarity in three
modes, a geodesic journey between two clips, and a browsable tree whose leaves
are the library's tracks, all with provider ids.

Main Features:
* /api/hyperbolic/similar in the three modes, with input validation
* /api/hyperbolic/journey starts and ends on the requested clips
* /api/hyperbolic/tree walks from the root down to track leaves
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids

pytestmark = pytest.mark.e2e


@pytest.fixture(scope='module')
def warmed(stack, api, analyzed_library):
    body = api.json('POST', '/api/hyperbolic/warmup', timeout=300)
    assert body.get('loaded') is True, body
    return body


def test_similar_modes(stack, api, library, warmed):
    seed = library.pid('A03')
    for mode in ('similar', 'roots', 'niche'):
        body = api.json('POST', '/api/hyperbolic/similar', json={'item_id': seed, 'limit': 5, 'mode': mode})
        assert_no_fp_ids(body)
        assert body['mode'] == mode
        assert body['seed_item_id'] == seed
        assert body['count'] == len(body['results']) <= 5
        for row in body['results']:
            assert row['item_id'] != seed
            assert row.get('title')
            assert 'distance' in row
            assert 'hyperbolic_radius' in row


def test_similar_validation(stack, api, library, warmed):
    seed = library.pid('A03')
    assert api.post('/api/hyperbolic/similar', json={}).status_code == 400
    assert api.post('/api/hyperbolic/similar', json={'item_id': seed, 'mode': 'x'}).status_code == 400
    assert api.post('/api/hyperbolic/similar', json={'item_id': seed, 'radial_spread': 2}).status_code == 400


def test_journey(stack, api, library, warmed):
    start, end = library.pid('A03'), library.pid('E03')
    body = api.json('POST', '/api/hyperbolic/journey', json={'start_item_id': start, 'end_item_id': end, 'length': 5})
    assert_no_fp_ids(body)
    results = body['results']
    assert results[0]['item_id'] == start, results
    assert results[-1]['item_id'] == end, results
    assert body['start_item_id'] == start
    assert body['end_item_id'] == end
    assert api.post('/api/hyperbolic/journey', json={'start_item_id': start, 'end_item_id': start, 'length': 5}).status_code == 400


def test_tree_walk_reaches_tracks(stack, api, warmed):
    body = api.json('GET', '/api/hyperbolic/tree')
    assert_no_fp_ids(body)
    node = body['node']
    assert node['id'] == 'root'
    seen_track = False
    for _ in range(8):
        items = node.get('items') or []
        if any(item.get('type') == 'track' for item in items):
            seen_track = True
            break
        folder = next((item for item in items if item.get('type') == 'folder'), None)
        assert folder is not None, node
        node = api.json('GET', f'/api/hyperbolic/tree?node_id={folder["id"]}')['node']
    assert seen_track, 'no track leaf reached'
    assert api.get('/api/hyperbolic/tree?node_id=bogus').status_code == 400
