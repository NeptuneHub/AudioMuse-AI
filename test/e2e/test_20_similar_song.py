# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Playlist from a similar song (ALGORITHM.md section 6) against the live IVF index.

Search finds a fixture clip by title, the similarity endpoint ranks its
neighbours by distance without echoing the seed, the mood centroid and
farthest-track helpers answer, and the resulting playlist is really created on
Navidrome with the posted tracks in order.

Main Features:
* /api/search_tracks and /api/similar_tracks answer with provider ids only
* validation: missing arguments 400, unknown id 404, unknown server 400
* /api/create_playlist writes the playlist on Navidrome, entries in order
"""

import urllib.parse

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, create_playlist, scalar, unique_name

pytestmark = pytest.mark.e2e


def _search(api, text):
    results = api.json('GET', '/api/search_tracks?search_query=' + urllib.parse.quote(text))
    assert_no_fp_ids(results)
    return results


def test_search_finds_a_clip(stack, api, library, analyzed_library):
    track = library.track('B01')
    results = _search(api, track.title[:12])
    assert results, 'search returned nothing'
    ids = {r['item_id'] for r in results}
    assert library.pid('B01') in ids, results
    for row in results:
        assert row.get('title') and 'author' in row


def test_similar_tracks_ranking(stack, api, library, analyzed_library):
    seed = library.pid('B01')
    results = api.json('GET', f'/api/similar_tracks?item_id={seed}&n=5')
    assert_no_fp_ids(results)
    assert 1 <= len(results) <= 5, results
    assert seed not in {r['item_id'] for r in results}
    distances = [r['distance'] for r in results]
    assert all(isinstance(d, (int, float)) and d >= 0 for d in distances), distances
    assert distances[0] == min(distances), distances
    for row in results:
        assert row.get('title') and 'author' in row
    track = library.track('B01')
    by_title = api.json(
        'GET',
        '/api/similar_tracks?n=5&title=' + urllib.parse.quote(track.title)
        + '&artist=' + urllib.parse.quote(track.artist),
    )
    assert by_title and by_title[0]['item_id'] == results[0]['item_id']
    capped = api.json('GET', f'/api/similar_tracks?item_id={seed}&n=12&eliminate_duplicates=true')
    per_author = {}
    for row in capped:
        per_author[row['author']] = per_author.get(row['author'], 0) + 1
    assert all(count <= 3 for count in per_author.values()), per_author


def test_similar_tracks_validation(stack, api, library, analyzed_library):
    seed = library.pid('B01')
    assert api.get('/api/similar_tracks').status_code == 400
    assert api.get('/api/similar_tracks?item_id=nope').status_code == 404
    assert api.get(f'/api/similar_tracks?item_id={seed}&server=nope').status_code == 400
    server_name = api.json('GET', '/api/servers')['servers'][0]['name']
    scoped = api.get(f'/api/similar_tracks?item_id={seed}&n=3&server=' + urllib.parse.quote(server_name))
    assert scoped.status_code == 200, scoped.text


def test_mood_centroids_and_max_distance(stack, api, library, analyzed_library):
    centroids = api.json('GET', '/api/mood_centroids')
    assert isinstance(centroids, dict) and centroids
    mood = next(iter(centroids))
    by_mood = api.json('GET', f'/api/similar_tracks?mood={urllib.parse.quote(mood)}&centroid_index=0&n=3')
    assert_no_fp_ids(by_mood)
    assert isinstance(by_mood, list) and by_mood
    farthest = api.json('GET', f'/api/max_distance?item_id={library.pid("B01")}')
    assert_no_fp_ids(farthest)
    assert farthest['max_distance'] > 0
    assert farthest['farthest_item_id']


def test_create_playlist_on_navidrome(stack, api, db, library, navidrome, analyzed_library):
    seed = library.pid('B01')
    results = api.json('GET', f'/api/similar_tracks?item_id={seed}&n=4')
    ids = [seed] + [r['item_id'] for r in results]
    name = unique_name('similar')
    before = scalar(db, 'SELECT count(*) FROM playlist')
    created = create_playlist(api, name, ids)
    try:
        assert created['playlist_id']
        assert created['mapped'] == len(ids) and created['skipped'] == 0, created
        remote = navidrome.playlist(created['playlist_id'])
        assert remote['name'].startswith(name), remote['name']
        assert navidrome.playlist_entry_ids(created['playlist_id']) == ids
        assert scalar(db, 'SELECT count(*) FROM playlist') == before
    finally:
        navidrome.delete_playlist(created['playlist_id'])
    assert api.post('/api/create_playlist', json={'playlist_name': name, 'track_ids': []}).status_code == 400
    assert api.post('/api/create_playlist', json={'track_ids': ids}).status_code == 400
