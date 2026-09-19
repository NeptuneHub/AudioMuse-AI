# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Music map (ALGORITHM.md section 9): the 2D projection of the whole catalogue.

The full map lists every catalogue row once with its 2D coordinates and
features, a percentage returns a subset, the precomputed cache reports its
buckets, and a synchronous rebuild leaves the same map behind.

Main Features:
* /api/map?percent=100 covers every catalogue row with provider ids
* smaller percentages return a non-empty subset
* the cache status is ok and the rebuild keeps the item count
* an unknown server selection is 400
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids

pytestmark = pytest.mark.e2e


def test_full_map(stack, api, library, analyzed_library):
    body = api.json('GET', '/api/map?percent=100')
    assert_no_fp_ids(body)
    items = body['items']
    assert len(items) == stack.catalogue_rows, len(items)
    assert len({i['item_id'] for i in items}) == len(items)
    for item in items:
        assert len(item['embedding_2d']) == 2
        assert item.get('title')
        assert 'artist' in item
        assert 'mood_vector' in item
    assert body.get('projection')


def test_subset_and_cache(stack, api, library, analyzed_library):
    subset = api.json('GET', '/api/map?percent=25')
    assert 0 < len(subset['items']) <= stack.catalogue_rows
    status = api.json('GET', '/api/map_cache_status')
    assert status['ok'] is True, status
    assert status['buckets'], status
    rebuilt = api.json('POST', '/api/rebuild_map_cache', timeout=300)
    assert rebuilt['ok'] is True, rebuilt
    again = api.json('GET', '/api/map?percent=100')
    assert len(again['items']) == stack.catalogue_rows
    assert api.get('/api/map?percent=100&server=nope').status_code == 400
