# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Song Alchemy (ALGORITHM.md section 8): blends, saved anchors and radios.

An ADD/SUBTRACT blend of two clips answers with results and the 2D preview
points, an anchor is saved from a real embedding and reused by the similarity
and path endpoints, a radio on that anchor is created, run against Navidrome
(the playlist appears under the anchor's name and the rerun reuses it), then
both are deleted. The artist projection endpoints are exercised too.

Main Features:
* POST /api/alchemy with ADD and SUBTRACT items, seeds never in the results
* anchors CRUD with a 200-float centroid, wrong length refused
* radios CRUD, one radio per anchor, /api/radios/run writes the playlist
* /api/artist_projections lists components and the rebuild succeeds
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, scalar, unique_name

pytestmark = pytest.mark.e2e

RESPONSE_KEYS = ('results', 'filtered_out', 'add_points', 'sub_points', 'projection')


def _blend(api, add, subtract, n=5):
    body = api.json(
        'POST', '/api/alchemy',
        json={'items': [{'id': add, 'op': 'ADD'}, {'id': subtract, 'op': 'SUBTRACT'}], 'n': n, 'temperature': 1.0},
    )
    assert_no_fp_ids(body)
    return body


def test_blend_two_clips(stack, api, library, analyzed_library):
    add, subtract = library.pid('A03'), library.pid('E03')
    body = _blend(api, add, subtract)
    for key in RESPONSE_KEYS:
        assert key in body, (key, list(body))
    results = body['results']
    assert 1 <= len(results) <= 5, results
    ids = {r['item_id'] for r in results}
    assert add not in ids and subtract not in ids
    for row in results:
        assert row.get('title') and 'author' in row
    assert api.post('/api/alchemy', json={'items': [], 'n': 5}).status_code == 400


def test_blend_with_an_artist_item(stack, api, library, analyzed_library):
    artists = api.json('GET', '/api/search_artists?query=Airmen')
    assert artists, 'artist search found nothing'
    artist = artists[0]
    artist_ref = artist.get('artist_id') or artist.get('artist') or artist.get('name')
    body = api.json(
        'POST', '/api/alchemy',
        json={'items': [{'id': artist_ref, 'op': 'ADD', 'type': 'artist'}], 'n': 4, 'temperature': 1.0},
    )
    assert_no_fp_ids(body)
    assert body['results']


def test_anchor_and_radio_lifecycle(stack, api, db, library, navidrome, analyzed_library):
    centroid = api.json('GET', f'/external/get_embedding?id={library.pid("A03")}')['embedding']
    assert len(centroid) == 200
    name = unique_name('anchor')
    assert api.post('/api/anchors', json={'name': name, 'centroid': centroid[:10]}).status_code == 400
    assert api.post('/api/anchors', json={'centroid': centroid}).status_code == 400
    created = api.json('POST', '/api/anchors', json={'name': name, 'centroid': centroid})
    anchor_id = created['anchor']['id']
    assert created['anchor']['name'] == name
    radio_id = None
    playlist_name = None
    try:
        listed = api.json('GET', '/api/anchors')['anchors']
        assert any(a['id'] == anchor_id for a in listed), listed
        assert scalar(db, 'SELECT jsonb_array_length(centroid::jsonb) FROM alchemy_anchors WHERE id = %s', (anchor_id,)) == 200
        renamed_name = name + '-b'
        api.json('PUT', f'/api/anchors/{anchor_id}', json={'name': renamed_name})
        similar = api.json('GET', f'/api/similar_tracks?anchor_id={anchor_id}&n=3')
        assert_no_fp_ids(similar)
        assert similar
        path = api.json('GET', f'/api/find_path?start_anchor={anchor_id}&end_song_id={library.pid("E03")}&max_steps=5')
        assert path['path'] and path['path'][-1]['item_id'] == library.pid('E03')
        blend = api.json('POST', '/api/alchemy', json={'items': [{'id': anchor_id, 'op': 'ADD', 'type': 'anchor'}], 'n': 3, 'temperature': 1.0})
        assert blend['results']

        radio = api.json('POST', '/api/radios', json={'anchor_id': anchor_id, 'temperature': 1.0, 'n_results': 5})['radio']
        radio_id = radio['id']
        assert radio['anchor_id'] == anchor_id and radio.get('enabled') is True
        assert api.post('/api/radios', json={'anchor_id': anchor_id, 'temperature': 1.0, 'n_results': 5}).status_code == 400
        radios = api.json('GET', '/api/radios')['radios']
        assert any(r['id'] == radio_id for r in radios), radios
        api.json('PUT', f'/api/radios/{radio_id}', json={'n_results': 4})

        summary = api.json('POST', '/api/radios/run', timeout=180)
        assert summary.get('playlists_created', 0) >= 1, summary
        assert not summary.get('failed'), summary
        playlist_name = renamed_name
        remote = navidrome.playlist_by_name(playlist_name)
        assert remote is not None, [p.get('name') for p in navidrome.playlists()]
        assert 1 <= len(navidrome.playlist_entry_ids(remote['id'])) <= 4
        api.json('POST', '/api/radios/run', timeout=180)
        again = navidrome.playlist_by_name(playlist_name)
        assert again['id'] == remote['id']
        assert sum(1 for p in navidrome.playlists() if p.get('name') == playlist_name) == 1
    finally:
        if radio_id is not None:
            api.json('DELETE', f'/api/radios/{radio_id}')
            assert api.delete(f'/api/radios/{radio_id}').status_code == 404
        api.json('DELETE', f'/api/anchors/{anchor_id}')
        if playlist_name:
            navidrome.delete_playlists_named(lambda n: n == playlist_name)
    assert scalar(db, 'SELECT count(*) FROM alchemy_radios WHERE anchor_id = %s', (anchor_id,)) == 0
    assert scalar(db, 'SELECT count(*) FROM alchemy_anchors WHERE id = %s', (anchor_id,)) == 0


def test_artist_projections(stack, api, analyzed_library):
    body = api.json('GET', '/api/artist_projections')
    assert_no_fp_ids(body)
    assert body['count'] > 0 and body['components']
    component = body['components'][0]
    assert 'artist_name' in component and len(component['projection']) == 2
    rebuilt = api.json('POST', '/api/build_artist_projection', timeout=180)
    assert rebuilt.get('status') == 'success', rebuilt
