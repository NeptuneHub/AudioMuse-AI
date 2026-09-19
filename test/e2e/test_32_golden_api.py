# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Exact answers of every deterministic read endpoint, recorded once and compared.

Each call below is made exactly as the UI makes it and its whole answer is
compared with the one recorded in seed/golden/test_32_golden_api.json after
normalization (provider ids become song, album, artist or server names,
generated ids and timestamps are dropped, floats are rounded). The score
rows the analysis wrote for the real clips are compared value by value as
well. Clustering, temperature-driven alchemy and the 2D map coordinates (the
UMAP projection is not seeded) are the only random outputs and stay out;
alchemy at temperature zero is deterministic and is in. The dashboard's
mood, genre and tempo buckets are checked for shape only: a real clip whose
scores sit on a bucket edge lands on one side or the other depending on the
CPU and thread count the models ran with.

Main Features:
* track, search, similar songs, mood centroids, farthest song, song path
* alchemy at temperature zero, map membership, artists, text search
* lyrics axes and searches, SemGrove, recording search by track
* hyperbolic neighbours, journey and tree root; external API; mobile sync
* dashboard summary and browser, setup coverage, plugins, servers
* score rows of every real clip the analysis processed
"""

import urllib.parse

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, item_id_of, rows

pytestmark = pytest.mark.e2e

REAL_CLIPS = (
    'A01', 'A02', 'A03', 'B01', 'B02', 'B03', 'C01', 'C02', 'C03', 'D01', 'D02', 'D03', 'E01', 'E02', 'E03', 'F02',
    'H01', 'H02', 'H03', 'H04', 'H05',
)
SCORE_COLUMNS = 'title, author, album_artist, tempo, key, scale, energy, mood_vector, other_features, duration'
MOOD_BUCKETS = {'relaxed', 'danceable', 'happy', 'sad', 'party', 'aggressive'}


@pytest.fixture(scope='module', autouse=True)
def _warm(stack, api, analyzed_library):
    api.wait_idle(180)
    api.json('POST', '/api/hyperbolic/warmup', timeout=300)
    api.json('POST', '/api/clap/warmup', timeout=300)
    api.json('POST', '/api/lyrics/warmup', timeout=300)
    api.json('POST', '/api/recording_search/warmup', timeout=300)


def _get(api, golden, key, path, **kwargs):
    body = api.json('GET', path, **kwargs)
    assert_no_fp_ids(body)
    golden.check(key, body)
    return body


def _post(api, golden, key, path, payload, **kwargs):
    body = api.json('POST', path, json=payload, **kwargs)
    assert_no_fp_ids(body)
    golden.check(key, body)
    return body


def test_tracks_and_similarity(stack, api, library, golden):
    a01, a03, b01, b02 = (library.pid(k) for k in ('A01', 'A03', 'B01', 'B02'))
    _get(api, golden, 'track A01', f'/api/track?item_id={a01}')
    _get(api, golden, 'search_tracks B01', '/api/search_tracks?search_query=' + urllib.parse.quote(library.track('B01').title[:12]))
    _get(api, golden, 'similar_tracks A03 n5', f'/api/similar_tracks?item_id={a03}&n=5')
    _get(api, golden, 'similar_tracks A03 n12 eliminate_duplicates', f'/api/similar_tracks?item_id={a03}&n=12&eliminate_duplicates=true')
    _get(api, golden, 'similar_tracks B02 n5', f'/api/similar_tracks?item_id={b02}&n=5')
    centroids = _get(api, golden, 'mood_centroids', '/api/mood_centroids')
    mood = next(iter(centroids))
    _get(api, golden, 'similar_tracks by first mood centroid', f'/api/similar_tracks?mood={urllib.parse.quote(mood)}&centroid_index=0&n=3')
    _get(api, golden, 'max_distance B01', f'/api/max_distance?item_id={b01}')


def test_path_alchemy_and_map(stack, api, library, golden):
    a01, b01, e03 = (library.pid(k) for k in ('A01', 'B01', 'E03'))
    _get(api, golden, 'find_path A01 to E03 max 5', f'/api/find_path?start_song_id={a01}&end_song_id={e03}&max_steps=5')
    _post(api, golden, 'alchemy ADD A01 SUBTRACT B01 temperature 0 n5', '/api/alchemy',
          {'items': [{'id': a01, 'op': 'ADD'}, {'id': b01, 'op': 'SUBTRACT'}], 'n': 5, 'temperature': 0})
    body = api.json('GET', '/api/map?percent=100')
    assert_no_fp_ids(body)
    items = body.get('items') if isinstance(body, dict) else body
    golden.check('map percent 100 members', sorted(golden.resolver.token(str(item['item_id'])) for item in items))


def test_artists_and_text_search(stack, api, library, golden):
    artist = library.track('E01').album_artist
    _get(api, golden, 'search_artists Airm', '/api/search_artists?query=Airm')
    _get(api, golden, 'similar_artists E n3', '/api/similar_artists?artist=' + urllib.parse.quote(artist) + '&n=3')
    _get(api, golden, 'similar_artists E n2 components', '/api/similar_artists?artist=' + urllib.parse.quote(artist) + '&n=2&include_component_matches=true')
    _get(api, golden, 'artist_tracks E', '/api/artist_tracks?artist=' + urllib.parse.quote(artist))
    _post(api, golden, 'clap search probe limit 8', '/api/clap/search', {'query': library.clap_probe['query'], 'limit': 8}, timeout=300)
    _get(api, golden, 'clap concepts', '/api/clap/concepts')
    _get(api, golden, 'clap stats', '/api/clap/stats')


def test_lyrics_semgrove_and_recording(stack, api, library, golden):
    h02 = library.pid('H02')
    axes = _get(api, golden, 'lyrics axes', '/api/lyrics/axes')['axes']
    axis_name, axis = next(iter(axes.items()))
    labels = axis['labels']
    label = next(iter(labels)) if isinstance(labels, dict) else labels[0]
    if isinstance(label, dict):
        label = label.get('key') or label.get('name')
    phrase = library.lyrics['H02']['probe_phrase']
    _post(api, golden, 'lyrics text search H02 phrase limit 6', '/api/lyrics/search/text', {'query': phrase, 'limit': 6}, timeout=300)
    _post(api, golden, 'lyrics axes search first axis first label', '/api/lyrics/search/axes', {'targets': {axis_name: label}, 'limit': 5})
    _get(api, golden, 'lyrics stats', '/api/lyrics/stats')
    _post(api, golden, 'sem_grove search H02 limit 5', '/api/sem_grove/search', {'item_id': h02, 'limit': 5})
    _get(api, golden, 'sem_grove stats', '/api/sem_grove/stats')
    _post(api, golden, 'recording_search by_track A02 n5', '/api/recording_search/by_track', {'item_id': library.pid('A02'), 'n_results': 5})


def test_hyperbolic(stack, api, library, golden):
    a01, a03, e03 = (library.pid(k) for k in ('A01', 'A03', 'E03'))
    for mode in ('similar', 'roots', 'niche'):
        _post(api, golden, f'hyperbolic {mode} A03 limit 5', '/api/hyperbolic/similar', {'item_id': a03, 'limit': 5, 'mode': mode})
    _post(api, golden, 'hyperbolic journey A01 to E03 length 5', '/api/hyperbolic/journey', {'start_item_id': a01, 'end_item_id': e03, 'length': 5})
    root = api.json('GET', '/api/hyperbolic/tree')['node']
    assert_no_fp_ids(root)
    golden.check('hyperbolic tree root items', [
        {'type': item.get('type'), 'name': item.get('name') or item.get('title'), 'count': item.get('count') or item.get('track_count')}
        for item in root.get('items') or []
    ])
    _get(api, golden, 'hyperbolic cache_status', '/api/hyperbolic/cache_status')


def test_external_sync_dashboard_and_setup(stack, api, library, golden):
    a03 = library.pid('A03')
    _get(api, golden, 'external get_score A03', f'/external/get_score?id={a03}')
    _get(api, golden, 'external get_embedding A03', f'/external/get_embedding?id={a03}')
    _get(api, golden, 'external search Variatio', '/external/search?search_query=Variatio')
    _get(api, golden, 'sync A03 without embeddings', f'/api/sync?ids={a03}&include_embeddings=false')
    index = api.json('GET', '/api/sync?fields=index&limit=500')
    golden.check('sync index totals', {k: v for k, v in index.items() if k != 'tracks'})
    summary = api.json('GET', '/api/dashboard/summary')
    assert_no_fp_ids(summary)
    content = dict(summary['content'])
    moods = content.pop('moods_coverage')
    genres = content.pop('top_genre')
    tempos = content.pop('tempo_profile')
    golden.check('dashboard summary content', content)
    assert {m['label'] for m in moods} <= MOOD_BUCKETS, moods
    assert 0 < sum(m['count'] for m in moods) <= content['total_songs'], moods
    assert genres and sum(g['count'] for g in genres) <= content['total_songs'], genres
    assert sum(tempos[k] for k in ('slow', 'medium', 'fast', 'very_fast')) == content['total_songs'], tempos
    _get(api, golden, 'browse songs page 1', '/api/dashboard/browse?kind=songs')
    _get(api, golden, 'browse artists', '/api/dashboard/browse?kind=artists')
    _get(api, golden, 'browse albums', '/api/dashboard/browse?kind=albums')
    _get(api, golden, 'browse unanalyzable', '/api/dashboard/browse?kind=unanalyzable')
    server_name = api.json('GET', '/api/servers')['servers'][0]['name']
    _get(api, golden, 'browse duplicates', '/api/dashboard/browse?kind=songs&filter=duplicates&server=' + urllib.parse.quote(server_name))
    setup = api.json('GET', '/api/setup')
    golden.check('setup coverage', {'setup_saved': setup.get('setup_saved'), 'model_coverage': setup.get('model_coverage')})
    _get(api, golden, 'plugins installed', '/api/plugins/installed')
    _get(api, golden, 'plugins repos', '/api/plugins/repos')
    servers = api.json('GET', '/api/servers')
    golden.check('servers', [{'name': s.get('name'), 'server_type': s.get('server_type'), 'is_default': s.get('is_default')} for s in servers['servers']])


def test_score_rows_of_the_real_clips(stack, db, library, golden):
    item_ids = sorted({item_id_of(db, library.pid(key)) for key in REAL_CLIPS})
    assert all(item_ids), item_ids
    table = rows(db, f'SELECT {SCORE_COLUMNS} FROM score WHERE item_id = ANY(%s) ORDER BY title, author', (item_ids,))
    assert len(table) == len(item_ids), (len(table), len(item_ids))
    golden.check('score rows of the real clips', [dict(zip(SCORE_COLUMNS.split(', '), row)) for row in table])
