# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Dashboard summary and the library browser on the analyzed catalogue.

The summary is refreshed by a timer thread in the web process, so the test
polls it until the content snapshot reflects the analysis; the browser lists
songs, artists and albums with counts derived from the manifest, exposes the
duplicate copy and the unanalyzable file, and validates its parameters.

Main Features:
* /api/dashboard/summary reaches the catalogue counts and lists the servers
* browse by kind: songs, artists, albums, unanalyzable, with a text filter
* the duplicates filter needs a server and shows the merged copy
"""

import time
import urllib.parse

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids

pytestmark = pytest.mark.e2e


def _summary_with_content(api, expected_songs, timeout=150):
    deadline = time.monotonic() + timeout
    summary = api.json('GET', '/api/dashboard/summary')
    while (summary.get('content') or {}).get('total_songs') != expected_songs:
        assert time.monotonic() < deadline, summary.get('content')
        time.sleep(5)
        summary = api.json('GET', '/api/dashboard/summary')
    return summary


def test_summary_reflects_the_catalogue(stack, api, library, analyzed_library):
    summary = _summary_with_content(api, stack.catalogue_rows)
    assert_no_fp_ids(summary)
    for key in ('generated_at', 'workers', 'queue_backlog', 'recent_tasks', 'content'):
        assert key in summary, key
    content = summary['content']
    assert content['total_songs'] == stack.catalogue_rows
    servers = content.get('music_servers') or []
    assert servers
    assert servers[0].get('name')


def test_browse_kinds(stack, api, library, analyzed_library):
    songs = api.json('GET', '/api/dashboard/browse?kind=songs')
    assert_no_fp_ids(songs)
    assert songs['kind'] == 'songs'
    assert songs['filter'] == 'all'
    seen = len(songs['results'])
    page = 1
    while songs['has_more']:
        page += 1
        songs = api.json('GET', f'/api/dashboard/browse?kind=songs&page={page}')
        seen += len(songs['results'])
        assert page < 50
    assert seen == stack.catalogue_rows, seen
    artists = api.json('GET', '/api/dashboard/browse?kind=artists')
    assert len(artists['results']) >= library.counts['clip_artists']
    albums = api.json('GET', '/api/dashboard/browse?kind=albums')
    assert len(albums['results']) >= library.counts['album_folders'] - len(library.unanalyzable) - 1 + len(stack.seed.albums)
    broken = api.json('GET', '/api/dashboard/browse?kind=unanalyzable')
    assert len(broken['results']) == len(library.unanalyzable), broken['results']
    fragment = library.track('C02').title[:8]
    filtered = api.json('GET', '/api/dashboard/browse?kind=songs&q=' + urllib.parse.quote(fragment))
    assert filtered['results']
    assert all(fragment.lower() in r['title'].lower() for r in filtered['results'])


def test_duplicates_filter(stack, api, library, analyzed_library):
    assert api.get('/api/dashboard/browse?kind=songs&filter=duplicates').status_code == 400
    server_name = api.json('GET', '/api/servers')['servers'][0]['name']
    dups = api.json('GET', '/api/dashboard/browse?kind=songs&filter=duplicates&server=' + urllib.parse.quote(server_name))
    assert_no_fp_ids(dups)
    merged = [c for c in library.copies.values() if c['expect'] == 'merged']
    assert len(dups['results']) == len(merged), dups['results']
    assert dups['results'][0].get('copies') == 2, dups['results']
    assert api.get('/api/dashboard/browse?kind=songs&server=bogus').status_code == 400
