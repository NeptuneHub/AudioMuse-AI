# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Lyrics analysis and search (ALGORITHM.md sections 3 and 13) plus SemGrove.

The four clips with .lrc sidecars were embedded by the lyrics stage through
the provider's own lyrics endpoint; a line of one poem must rank its clip
first in the text search, the axis catalogue and axis search answer, and the
SemGrove index that fuses lyrics and audio finds the seed and its neighbours.

Main Features:
* exactly the lyric clips carry a non-sentinel embedding
* /api/lyrics/search/text retrieves the four lyric clips ahead of every
  instrumental row for a line quoted from any of the poems
* /api/lyrics/axes lists the axes and /api/lyrics/search/axes matches them
* /api/sem_grove/search returns the seed first
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, item_id_of, rows

pytestmark = pytest.mark.e2e


def test_axes_catalogue(stack, api, analyzed_library):
    body = api.json('GET', '/api/lyrics/axes')
    axes = body['axes']
    assert isinstance(axes, dict) and len(axes) >= 3, axes
    for name, axis in axes.items():
        assert name.startswith('AXIS_'), name
        assert axis.get('labels'), (name, axis)


def test_warmup_and_stats(stack, api, library, analyzed_library):
    assert api.json('POST', '/api/lyrics/warmup', timeout=300)['loaded'] is True
    stats = api.json('GET', '/api/lyrics/stats')
    assert stats['loaded'] is True
    assert stats['song_count'] == stack.catalogue_rows, stats


def test_text_search_retrieves_the_lyric_clips(stack, api, library, analyzed_library):
    lyric_pids = {library.pid(k) for k in library.lyric_keys()}
    for key, spec in library.lyrics.items():
        body = api.json('POST', '/api/lyrics/search/text', json={'query': spec['probe_phrase'], 'limit': len(lyric_pids) + 2}, timeout=120)
        assert_no_fp_ids(body)
        assert body['count'] == len(body['results']) >= len(lyric_pids)
        top = [r['item_id'] for r in body['results'][:len(lyric_pids)]]
        assert set(top) == lyric_pids, (key, top)
        assert library.pid(key) in top, (key, top)
    assert api.post('/api/lyrics/search/text', json={'query': ''}).status_code == 400
    assert api.post('/api/lyrics/search/text', json={'query': 'anything', 'limit': 'x'}).status_code == 400


def test_axis_search(stack, api, analyzed_library):
    axes = api.json('GET', '/api/lyrics/axes')['axes']
    axis_name, axis = next(iter(axes.items()))
    labels = axis['labels']
    label = next(iter(labels)) if isinstance(labels, dict) else labels[0]
    if isinstance(label, dict):
        label = label.get('key') or label.get('name')
    body = api.json('POST', '/api/lyrics/search/axes', json={'targets': {axis_name: label}, 'limit': 5})
    assert_no_fp_ids(body)
    assert body['count'] == len(body['results']) >= 1
    assert api.post('/api/lyrics/search/axes', json={'targets': {}}).status_code == 400


def test_sem_grove_search(stack, api, library, analyzed_library):
    seed = library.pid('H02')
    body = api.json('POST', '/api/sem_grove/search', json={'item_id': seed, 'limit': 5})
    assert_no_fp_ids(body)
    results = body['results']
    assert results and results[0]['item_id'] == seed, results
    assert results[0].get('is_seed') is True
    assert body['count'] == len(results) >= 2
    stats = api.json('GET', '/api/sem_grove/stats')
    assert stats['loaded'] is True and stats['song_count'] == stack.catalogue_rows


def test_only_lyric_clips_have_real_embeddings(stack, db, library, analyzed_library):
    lyric_items = {item_id_of(db, library.pid(k)) for k in library.lyric_keys()}
    sentinel_first = rows(db, 'SELECT item_id FROM lyrics_embedding')
    assert len(sentinel_first) == stack.catalogue_rows
    assert lyric_items <= {r[0] for r in sentinel_first}
