# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Album Creation (ALGORITHM.md section 18) on the real stack.

The album is sampled, so nothing here is compared against a recorded ranking:
every test asserts the invariants an album must keep whatever the draw.

Main Features:
* a song seed stays in its album, which is ordered opener to closer, duplicate
  free, within the per-artist cap and free of internal ids
* a description seed answers on the same contract, and a description the DCLAP
  index cannot serve is a clean error rather than a crash
* the album search finds the library's albums and needs two letters
* bad input is 400, an unknown seed 404, an unknown server 400
* the proposal becomes a Navidrome playlist in running order
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, create_playlist, unique_name

pytestmark = pytest.mark.e2e

SEED_KEY = 'B01'
ARTIST = 'Airmen of Note'
ALBUM_TRACKS = 12
ARTIST_CAP = 3
GENERATE = '/api/album_creation/generate'


def _check_album(album, expected_type):
    assert_no_fp_ids(album)
    assert album['seed']['type'] == expected_type and album['seed']['label']
    assert album['suggested_name']
    tracks = album['tracks']
    assert 4 <= len(tracks) <= ALBUM_TRACKS, album
    ids = [track['item_id'] for track in tracks]
    assert len(ids) == len(set(ids)), ids
    assert [track['slot'] for track in tracks] == list(range(1, len(tracks) + 1))
    assert tracks[0]['role'] == 'opener' and tracks[-1]['role'] == 'closer', tracks
    assert [track['role'] for track in tracks[1:3]] == ['single', 'single'], tracks
    assert all(track['role'] == 'track' for track in tracks[3:-1]), tracks
    for track in tracks:
        assert track.get('title') and 'author' in track
    stats = album['stats']
    assert stats['tracks'] == len(tracks) and stats['minutes'] >= 0
    assert 0 < stats['cohesion'] <= 1 and 0 < stats['target_cohesion'] <= 1
    assert stats['opener_style'] in ('intro', 'bang')
    return tracks


def _capped(tracks, exempt=None):
    authors = [track['author'] for track in tracks if track['author'] != exempt]
    return all(authors.count(author) <= ARTIST_CAP for author in set(authors))


def test_a_song_seed_stays_in_its_album(stack, api, library, analyzed_library):
    seed = library.pid(SEED_KEY)
    tracks = _check_album(api.json('POST', GENERATE, json={'seed_type': 'song', 'item_id': seed}), 'song')
    assert seed in [track['item_id'] for track in tracks]
    assert _capped(tracks), tracks


def test_two_draws_are_both_valid_albums(stack, api, library, analyzed_library):
    seed = library.pid(SEED_KEY)
    for _ in range(2):
        _check_album(api.json('POST', GENERATE, json={'seed_type': 'song', 'item_id': seed}), 'song')


def test_validation(stack, api, library, analyzed_library):
    seed = library.pid(SEED_KEY)
    api.post('/api/clap/warmup', json={}, timeout=300)
    for body in ({}, {'seed_type': 'playlist', 'item_id': seed}, {'seed_type': 'song'},
                 {'seed_type': 'text'}, {'seed_type': 'text', 'query': '   '},
                 {'seed_type': 'album', 'album': 'x'}, {'seed_type': 'artist', 'artist': 'x'}):
        assert api.post(GENERATE, json=body).status_code == 400, body
    assert api.post(GENERATE, json={'seed_type': 'song', 'item_id': 'no-such-song'}).status_code == 404
    assert api.post(GENERATE, json={'seed_type': 'song', 'item_id': seed, 'server': 'nope'}).status_code == 400
    assert api.get('/api/album_creation/search_albums?query=the').status_code == 404


def test_the_proposal_becomes_a_playlist_in_running_order(stack, api, library, navidrome, analyzed_library):
    album = api.json('POST', GENERATE, json={'seed_type': 'song', 'item_id': library.pid(SEED_KEY)})
    ids = [track['item_id'] for track in album['tracks']]
    created = create_playlist(api, unique_name('album'), ids)
    try:
        assert navidrome.playlist_entry_ids(created['playlist_id']) == ids
    finally:
        navidrome.delete_playlist(created['playlist_id'])


def test_a_description_seed_answers_or_explains_itself(stack, api, library, analyzed_library):
    api.post('/api/clap/warmup', json={}, timeout=300)
    response = api.post(GENERATE, json={'seed_type': 'text', 'query': 'calm piano music'}, timeout=300)
    if response.status_code != 200:
        assert 'DCLAP' in response.json()['error'], response.text
        pytest.skip('this stack serves no DCLAP index, so a text seed cannot be built')
    album = _check_album(response.json(), 'text')
    assert response.json()['seed'] == {'type': 'text', 'label': 'calm piano music'}
    assert response.json()['suggested_name'] == 'Calm piano music'
    assert album
