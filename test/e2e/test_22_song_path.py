# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Song path (ALGORITHM.md section 7): a smooth sequence between two clips.

The path starts and ends on the requested songs, every hop is a catalogue
track answered with its provider id, a mood may replace one endpoint, the
lyrics space works between two clips that carry lyrics, and the path becomes
a real playlist on Navidrome.

Main Features:
* /api/find_path returns the endpoints first and last with a finite distance
* identical endpoints and an unknown mood are 400
* path_space=lyrics runs on the SemGrove index for lyric clips
* the path is created as a playlist on Navidrome
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, create_playlist, unique_name

pytestmark = pytest.mark.e2e


def _path(api, query, expect=200):
    body = api.json('GET', '/api/find_path?' + query, expect=expect)
    assert_no_fp_ids(body)
    return body


def test_path_between_two_clips(stack, api, library, analyzed_library):
    start, end = library.pid('A03'), library.pid('E03')
    body = _path(api, f'start_song_id={start}&end_song_id={end}&max_steps=5')
    path = body['path']
    assert path[0]['item_id'] == start and path[-1]['item_id'] == end, path
    assert 2 <= len(path) <= 7, len(path)
    assert isinstance(body['total_distance'], (int, float)) and body['total_distance'] >= 0
    for hop in path:
        assert hop.get('title') and 'author' in hop
        assert isinstance(hop.get('embedding_vector'), list)
        assert 'top_genre' in hop


def test_path_validation_and_mood_endpoint(stack, api, library, analyzed_library):
    start = library.pid('A03')
    assert api.get(f'/api/find_path?start_song_id={start}&end_song_id={start}&max_steps=5').status_code == 400
    assert api.get(f'/api/find_path?start_mood=bogus&end_song_id={start}&max_steps=5').status_code == 400
    body = _path(api, f'start_mood=happy&end_song_id={start}&max_steps=5')
    assert body['path'] and body['path'][-1]['item_id'] == start


def test_path_in_lyrics_space(stack, api, library, analyzed_library):
    start, end = library.pid('H02'), library.pid('H03')
    body = _path(api, f'start_song_id={start}&end_song_id={end}&max_steps=5&path_space=lyrics')
    assert body['path'][0]['item_id'] == start and body['path'][-1]['item_id'] == end


def test_path_becomes_a_playlist(stack, api, library, navidrome, analyzed_library):
    start, end = library.pid('A03'), library.pid('E03')
    path = _path(api, f'start_song_id={start}&end_song_id={end}&max_steps=5')['path']
    ids = [hop['item_id'] for hop in path]
    name = unique_name('path')
    created = create_playlist(api, name, ids)
    try:
        assert navidrome.playlist_entry_ids(created['playlist_id']) == ids
    finally:
        navidrome.delete_playlist(created['playlist_id'])
