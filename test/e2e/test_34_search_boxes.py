# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every search box of the UI follows one standard, checked in a real browser.

Each song, artist and playlist picker is operated like a person does: one
letter is typed, the first page of 20 suggestions must be requested and
shown, and scrolling the list to its end must request and append the next 20
without repeating a row. The keyboard then picks a suggestion and the page
must record it. The pickers all run on static/autocomplete.js, so this module
is the functional proof that no page drifted from the shared contract.

Main Features:
* one typed letter searches (start=0&end=20) on every picker of every page
* the unfiltered song pickers load the next page on scroll, with no duplicate,
  driven by the endpoint's X-Search-Has-More header
* artist and playlist pickers send the same paged request
* ArrowDown + Enter picks a suggestion, closes the list and fills the page
* an alchemy pick refused as a duplicate leaves the list open for another pick
* the map Search button searches the text typed after an earlier pick
* no console error, page error, failed request or HTTP 500 on the way
"""

import time
import urllib.parse

import pytest

import test.e2e.test_33_ui_flows as ui

pytestmark = [pytest.mark.e2e, pytest.mark.browser]

flow = ui.flow
LETTER = 's'
PAGE = 20
TRACKS = '/api/search_tracks'
ARTISTS = '/api/search_artists'
PLAYLISTS = '/api/search_playlists'
ALCHEMY_CARD = '.alchemy-card >> nth=0'
CYCLE = ALCHEMY_CARD + ' >> .type-cycle-btn'

PICKERS = [
    dict(name='similarity', path='/similarity', ready='#similarity-form', input='#search_query',
         box='#autocomplete-results', endpoint=TRACKS, paged=True, picked=('value', '#selected_item_id')),
    dict(name='path-start', path='/path', ready='#path-form', input='#start_search',
         box='#start-autocomplete-results', endpoint=TRACKS, paged=True, picked=('value', '#start_song_id')),
    dict(name='path-end', path='/path', ready='#path-form', input='#end_search',
         box='#end-autocomplete-results', endpoint=TRACKS, paged=True, picked=('value', '#end_song_id')),
    dict(name='alchemy-song', path='/alchemy', ready='#alchemy-form', input=ALCHEMY_CARD + ' >> input.song',
         box=ALCHEMY_CARD + ' >> .autocomplete-results-song', endpoint=TRACKS, paged=True,
         picked=('value', ALCHEMY_CARD + ' >> .song-id')),
    dict(name='alchemy-artist', path='/alchemy', ready='#alchemy-form', input=ALCHEMY_CARD + ' >> input.song',
         clicks=[CYCLE], box=ALCHEMY_CARD + ' >> .autocomplete-results-artist', endpoint=ARTISTS, paged=False,
         picked=('value', ALCHEMY_CARD + ' >> .song-id')),
    dict(name='alchemy-playlist', path='/alchemy', ready='#alchemy-form', input=ALCHEMY_CARD + ' >> input.song',
         clicks=[CYCLE] * 4, box=ALCHEMY_CARD + ' >> .autocomplete-results-playlist', endpoint=PLAYLISTS,
         paged=False, picked=('value', ALCHEMY_CARD + ' >> .song-id')),
    dict(name='hyperbolic', path='/hyperbolic', ready='#hyper-similar-form', input='#hyper-song-input',
         box='#hyper-autocomplete-results', endpoint=TRACKS, paged=True, picked=('changed', '#hyper-song-input')),
    dict(name='hyperbolic-journey-start', path='/hyperbolic', ready='#hyper-similar-form',
         clicks=['.tab-btn[data-tab="journey"]'], input='#journey-start-input', box='#journey-start-results',
         endpoint=TRACKS, paged=True, picked=('changed', '#journey-start-input')),
    dict(name='hyperbolic-journey-end', path='/hyperbolic', ready='#hyper-similar-form',
         clicks=['.tab-btn[data-tab="journey"]'], input='#journey-end-input', box='#journey-end-results',
         endpoint=TRACKS, paged=True, picked=('changed', '#journey-end-input')),
    dict(name='map', path='/map', ready='#map_size', input='#search_query_map', box='#map_autocomplete_results',
         endpoint=TRACKS, paged=True, picked=('changed', '#search_query_map')),
    dict(name='artist-similarity', path='/artist_similarity', ready='#artist-similarity-form', input='#artist_search',
         box='#autocomplete-results', endpoint=ARTISTS, paged=False, picked=('changed', '#artist_search')),
    dict(name='album-creation', path='/album_creation', ready='#album-creation-form', input='#seed_query',
         box='#seed-suggestions', endpoint=TRACKS, paged=True, picked=('text', '#seed-selected')),
    dict(name='lyrics-by-song', path='/lyrics_search', ready='#axis-form', clicks=['.tab-btn[data-tab="song"]'],
         input='#sg-search-query', box='#sg-autocomplete-results', endpoint=TRACKS, paged=False,
         params={'index': 'sem_grove'}, picked=('value', '#sg-selected-item-id')),
    dict(name='recording-by-song', path='/recording_search', ready='#recording-panel',
         clicks=['.tab-btn[data-mode="song"]'], input='#song-query', box='#song-suggestions', endpoint=TRACKS,
         paged=False, params={'index': 'neural'}, picked=('text', '#song-selected'), optional=True),
]


def _window(start):
    return {'start': str(start), 'end': str(start + PAGE)}


def _asks(endpoint, start):
    expected = _window(start)

    def matches(response):
        parts = urllib.parse.urlsplit(response.url)
        query = dict(urllib.parse.parse_qsl(parts.query))
        return parts.path == endpoint and all(query.get(key) == value for key, value in expected.items())

    return matches


def _rows(box):
    return box.locator('[role="option"]')


def _wait_rows(box, expected):
    deadline = time.monotonic() + ui.RENDER_TIMEOUT_S
    while _rows(box).count() != expected and time.monotonic() < deadline:
        time.sleep(0.2)
    return _rows(box).count()


def _identity(row):
    return row.get('item_id') or row.get('id') or row.get('artist')


@pytest.mark.parametrize('picker', PICKERS, ids=[p['name'] for p in PICKERS])
def test_one_letter_searches_and_scrolling_loads_the_next_page(flow, picker):
    page, problems = flow
    ui._open(page, picker['path'], picker['ready'])
    for selector in picker.get('clicks', []):
        target = page.locator(selector).first
        if picker.get('optional') and not target.count():
            pytest.skip(f"{picker['name']}: {selector} is not rendered on this instance")
        target.wait_for(state='visible', timeout=ui.PICK_TIMEOUT_MS)
        target.click()
    field = page.locator(picker['input']).first
    box = page.locator(picker['box']).first
    field.wait_for(state='visible', timeout=ui.PICK_TIMEOUT_MS)
    if field.is_disabled():
        pytest.skip(f"{picker['name']}: the picker is disabled until its index is loaded")

    with page.expect_response(_asks(picker['endpoint'], 0), timeout=ui.PICK_TIMEOUT_MS) as first:
        field.fill(LETTER)
    sent = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(first.value.url).query))
    assert sent.get('search_query', sent.get('query')) == LETTER, sent
    for key, value in picker.get('params', {}).items():
        assert sent.get(key) == value, sent
    rows = first.value.json()
    assert isinstance(rows, list), rows
    assert len(rows) <= PAGE, len(rows)
    if rows:
        assert _wait_rows(box, len(rows)) == len(rows)
        assert box.is_visible()

    if picker['paged']:
        assert len(rows) == PAGE, f"{picker['name']}: the seeded catalogue must fill the first page"
        assert first.value.headers.get('x-search-has-more') == '1', first.value.headers
        with page.expect_response(_asks(picker['endpoint'], PAGE), timeout=ui.PICK_TIMEOUT_MS) as second:
            box.evaluate('b => { b.scrollTop = b.scrollHeight; }')
        more = second.value.json()
        assert more, f"{picker['name']}: the second page is empty"
        assert _wait_rows(box, len(rows) + len(more)) == len(rows) + len(more)
        seen = [_identity(row) for row in rows + more]
        assert len(seen) == len(set(seen)), f"{picker['name']}: a row came back twice across pages"

    if rows:
        field.press('ArrowDown')
        assert 'active' in (_rows(box).first.get_attribute('class') or '')
        before = field.input_value()
        field.press('Enter')
        page.wait_for_timeout(300)
        assert not box.is_visible(), f"{picker['name']}: the list stayed open after the pick"
        kind, selector = picker['picked']
        target = page.locator(selector).first
        if kind == 'value':
            assert target.input_value(), f"{picker['name']}: the pick was not recorded"
        elif kind == 'changed':
            assert target.input_value() != before, f"{picker['name']}: the pick did not fill the input"
        else:
            assert target.inner_text().strip(), f"{picker['name']}: the pick was not shown"
    ui._clean(problems, picker['path'])


def _first_suggestion(page, field, box):
    with page.expect_response(_asks(TRACKS, 0), timeout=ui.PICK_TIMEOUT_MS):
        field.fill(LETTER)
    row = _rows(box).first
    row.wait_for(state='visible', timeout=ui.PICK_TIMEOUT_MS)
    return row


def test_a_duplicate_alchemy_pick_keeps_the_list_open(flow):
    page, problems = flow
    ui._open(page, '/alchemy', '#alchemy-form')
    cards = page.locator('.alchemy-card')
    cards.nth(1).wait_for(state='attached', timeout=ui.PICK_TIMEOUT_MS)
    first, second = cards.nth(0), cards.nth(1)
    _first_suggestion(page, first.locator('input.song'), first.locator('.autocomplete-results-song')).click()
    picked = first.locator('.song-id').input_value()
    assert picked
    box = second.locator('.autocomplete-results-song')
    _first_suggestion(page, second.locator('input.song'), box).click()
    page.wait_for_timeout(300)
    assert second.locator('.song-id').input_value() == ''
    assert box.is_visible(), 'a refused duplicate must leave the suggestions open for another pick'
    assert _rows(box).count() >= 1
    ui._clean(problems, '/alchemy')


def test_the_map_search_button_follows_the_text_typed_after_a_pick(flow):
    page, problems = flow
    ui._open(page, '/map', '#map_size')
    field = page.locator('#search_query_map')
    _first_suggestion(page, field, page.locator('#map_autocomplete_results')).click()
    picked_text = field.input_value()
    assert picked_text
    with page.expect_response(_asks(TRACKS, 0), timeout=ui.PICK_TIMEOUT_MS):
        field.fill(LETTER)
    with page.expect_response(_asks(TRACKS, 0), timeout=ui.PICK_TIMEOUT_MS) as searched:
        page.locator('#map_search_btn').click()
    sent = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(searched.value.url).query))
    assert sent.get('search_query') == LETTER, sent
    ui._clean(problems, '/map')
