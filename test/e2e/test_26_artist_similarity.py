# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Artist similarity (ALGORITHM.md section 11) on the five fixture performers.

Each album is one performer, so the artist index built at analysis time holds
every performer; search finds them by a name fragment, similarity ranks the
others without echoing the query artist, and the per-artist track list carries
provider ids.

Main Features:
* /api/search_artists finds a performer by a fragment with its track count
* /api/similar_artists ranks the other performers, honours n, and answers 404
  for an unknown artist and 400 without a query
* /api/artist_tracks lists the performer's clips with provider ids
"""

import urllib.parse

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, scalar

pytestmark = pytest.mark.e2e

ARTIST = 'Airmen of Note'


def test_search_artists(stack, api, library, analyzed_library):
    results = api.json('GET', '/api/search_artists?query=Airm')
    assert_no_fp_ids(results)
    assert results, results
    match = next((r for r in results if r.get('artist') == ARTIST or r.get('name') == ARTIST), None)
    assert match is not None, results
    if 'track_count' in match:
        assert match['track_count'] == 3, match


def test_similar_artists(stack, api, library, analyzed_library):
    results = api.json('GET', '/api/similar_artists?artist=' + urllib.parse.quote(ARTIST) + '&n=3')
    assert_no_fp_ids(results)
    assert 1 <= len(results) <= 3, results
    names = [r.get('artist') or r.get('name') for r in results]
    assert ARTIST not in names, names
    assert all('divergence' in r or 'distance' in r or 'score' in r for r in results), results
    with_components = api.json(
        'GET', '/api/similar_artists?artist=' + urllib.parse.quote(ARTIST) + '&n=2&include_component_matches=true'
    )
    assert_no_fp_ids(with_components)
    assert with_components
    assert api.get('/api/similar_artists?artist=Nobody%20Here&n=3').status_code == 404
    assert api.get('/api/similar_artists').status_code == 400


def test_artist_tracks(stack, api, library, analyzed_library):
    tracks = api.json('GET', '/api/artist_tracks?artist=' + urllib.parse.quote(ARTIST))
    assert_no_fp_ids(tracks)
    ids = {t['item_id'] for t in tracks}
    assert ids == {library.pid(k) for k in ('E01', 'E02', 'E03')}, tracks


def test_artist_index_tables(stack, db, library, analyzed_library):
    assert scalar(db, 'SELECT count(*) FROM artist_server_map') >= library.counts['clip_artists']
    assert scalar(db, 'SELECT count(*) FROM artist_metadata_data') >= 1
