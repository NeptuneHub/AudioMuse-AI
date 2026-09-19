# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Similarity indexes (ALGORITHM.md section 4) after the analysis's final rebuild.

Every index the worker builds is loaded by the web process and reports the
catalogue size: the CLAP, lyrics and SemGrove caches, the map cache, the
hyperbolic tree, and the wizard's coverage bands; the cache refresh endpoints
reload without changing the counts.

Main Features:
* clap, lyrics and sem_grove stats are loaded with song_count == catalogue rows
* map cache and hyperbolic tree are ready; every coverage band is the top one
* the three cache refresh endpoints succeed and keep the counts
* the persisted index directory rows exist for every index
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, rows

pytestmark = pytest.mark.e2e

INDEX_NAMES = (
    'music_library', 'clap_index', 'lyrics_index', 'lyrics_axes_index',
    'sem_grove_index', 'artist_similarity_index', 'neural_fingerprint_index',
)

TOP_COVERAGE_BAND = 5

STATS = (
    ('/api/clap/stats', 'embedding_dimension', 512),
    ('/api/lyrics/stats', 'embedding_dimension', 768),
    ('/api/sem_grove/stats', 'audio_dim', 200),
)


def test_stats_report_the_catalogue(stack, api, library, analyzed_library):
    for path, dim_key, dim in STATS:
        stats = api.json('GET', path)
        assert_no_fp_ids(stats)
        assert stats['loaded'] is True, (path, stats)
        assert stats['song_count'] == stack.catalogue_rows, (path, stats)
        if dim_key in stats:
            assert stats[dim_key] == dim, (path, stats)


def test_map_and_hyperbolic_ready(stack, api, library, analyzed_library):
    assert api.json('GET', '/api/map_cache_status')['ok'] is True
    warm = api.json('POST', '/api/hyperbolic/warmup', timeout=300)
    assert warm.get('loaded') is True, warm
    status = api.json('GET', '/api/hyperbolic/cache_status')
    assert status.get('ok') is True, status
    if 'track_count' in status:
        assert status['track_count'] == stack.catalogue_rows, status


def test_coverage_bands_are_full(stack, api, analyzed_library):
    coverage = api.json('GET', '/api/setup')['model_coverage']
    assert coverage, coverage
    assert all(band == TOP_COVERAGE_BAND for band in coverage.values()), coverage


def test_cache_refresh_keeps_counts(stack, api, library, analyzed_library):
    for path in ('/api/clap/cache/refresh', '/api/lyrics/cache/refresh', '/api/sem_grove/cache/refresh'):
        body = api.json('POST', path, timeout=300)
        assert body.get('success') is True or body.get('loaded') is True, (path, body)
    for path, _key, _dim in STATS:
        assert api.json('GET', path)['song_count'] == stack.catalogue_rows, path


def test_index_rows_persisted(stack, db, analyzed_library):
    names = {r[0] for r in rows(db, 'SELECT name FROM ivf_dir')}
    for index in INDEX_NAMES:
        assert f'{index}__ivf_dir' in names, (index, sorted(names))
    assert any(name.startswith('hyperbolic_') for name in names), sorted(names)
    assert rows(db, "SELECT count(*) FROM map_projection_data WHERE index_name = 'main_map'")[0][0] == 1
