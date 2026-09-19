# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""UI flows: the main forms operated in a real browser, and the songs they show.

Each flow opens a page in headless Chromium and uses it the way a person
does: types a title into the search box, waits for the autocomplete, clicks
the suggestion, sets the options, presses the button. It then checks three
things. The rows on screen are exactly the rows of the answer the page
received, in the same order, so the rendering neither drops nor reorders
songs. The songs on screen equal the recorded expectation in
seed/golden/test_33_ui_flows.json, so a change in what the user sees fails
with the expected and the actual song. Where the page sends the same request
a recorded API answer exists for, the songs equal that API answer too. No
JavaScript error, failed request or HTTP 500 may occur on the way. One flow
also creates a playlist from the page and reads it back from Navidrome.

Main Features:
* similar songs, song path, alchemy at temperature zero, text search, lyrics
  text search, similar artists, hyperbolic neighbours, sonic fingerprint and
  the library browser with its search box
* rows on screen equal the page's own API answer, in order
* the songs shown equal the recorded expectation (re-record with
  AUDIOMUSE_E2E_RECORD_GOLDEN=1 bash test/e2e/run_local.sh -k ui_flows)
* a playlist created from the similarity page exists on Navidrome
"""

import json
import os
import re
import time
import urllib.parse

import pytest

from test.e2e.e2e_helpers import unique_name
from test.e2e.golden import GOLDEN_DIR, RECORD_ENV, Golden, Resolver
from test.e2e.stack import library as fixture_library
from test.e2e.stack import paths
from test.e2e.stack.env import NAVIDROME_ADMIN_PASSWORD, NAVIDROME_ADMIN_USER

pytestmark = [pytest.mark.e2e, pytest.mark.browser]

ATTACH_ENV = 'AUDIOMUSE_E2E_BASE_URL'
API_GOLDEN = os.path.join(GOLDEN_DIR, 'test_32_golden_api.json')
VIEWPORT = {'viewport': {'width': 1366, 'height': 900}}
IGNORED_FAILURES = ('net::ERR_ABORTED',)
ROW_PREFIX = re.compile(r'^\d+\.\s*')
UI_PLAYLIST_STEM = 'ui-similarity'
UI_PLAYLIST_PREFIX = 'e2e-' + UI_PLAYLIST_STEM
PICK_TIMEOUT_MS = 20000
ANSWER_TIMEOUT_MS = 300000
RENDER_TIMEOUT_S = 15


@pytest.fixture(scope='module')
def lib():
    return fixture_library.load()


@pytest.fixture(scope='module')
def ui_golden():
    recorder = Golden('test_33_ui_flows', Resolver(), bool(os.environ.get(RECORD_ENV, '').strip()))
    yield recorder
    recorder.flush()


@pytest.fixture(scope='module')
def api_golden():
    if not os.path.isfile(API_GOLDEN):
        return {}
    with open(API_GOLDEN, encoding='utf-8') as handle:
        return json.load(handle)


@pytest.fixture
def flow(browser, page_base_url, request):
    os.makedirs(paths.PLAYWRIGHT_DIR, exist_ok=True)
    context = browser.new_context(base_url=page_base_url, **VIEWPORT)
    context.tracing.start(screenshots=True, snapshots=True)
    page = context.new_page()
    origin = urllib.parse.urlsplit(page_base_url).netloc
    problems = []

    def same_origin(url):
        return urllib.parse.urlsplit(url).netloc == origin

    page.on('console', lambda msg: problems.append(f'console.{msg.type}: {msg.text}') if msg.type == 'error' else None)
    page.on('pageerror', lambda exc: problems.append(f'pageerror: {exc}'))
    page.on(
        'requestfailed',
        lambda req: problems.append(f'requestfailed: {req.url} {req.failure}')
        if same_origin(req.url) and not any(code in str(req.failure) for code in IGNORED_FAILURES)
        else None,
    )
    page.on(
        'response',
        lambda res: problems.append(f'http {res.status}: {res.url}')
        if same_origin(res.url) and res.status >= 500
        else None,
    )
    yield page, problems
    failed = getattr(request.node, 'rep_call', None) is not None and request.node.rep_call.failed
    stem = os.path.join(paths.PLAYWRIGHT_DIR, re.sub(r'[^A-Za-z0-9._-]+', '_', request.node.name)[:120])
    if failed:
        try:
            page.screenshot(path=stem + '.png', full_page=True)
        except Exception:
            pass
        context.tracing.stop(path=stem + '.zip')
    else:
        context.tracing.stop()
    context.close()


def _open(page, path, ready):
    response = page.goto(path, wait_until='domcontentloaded')
    assert response is not None and response.ok, f'{path}: {response and response.status}'
    page.locator(ready).first.wait_for(state='attached', timeout=PICK_TIMEOUT_MS)


def _pick(page, field, suggestions, title):
    field.fill(title)
    item = suggestions.filter(has_text=title).first
    item.wait_for(state='visible', timeout=PICK_TIMEOUT_MS)
    item.click()


def _submit(page, button, api_path, method):
    with page.expect_response(
        lambda res: urllib.parse.urlsplit(res.url).path == api_path and res.request.method == method,
        timeout=ANSWER_TIMEOUT_MS,
    ) as answer:
        page.locator(button).first.click()
    response = answer.value
    assert response.ok, f'{api_path}: {response.status} {response.text()[:300]}'
    return response.json()


def _pairs(rows):
    return [[row.get('title') or 'Unknown', row.get('author') or row.get('artist') or ''] for row in rows]


def _shown(page, container, expected_count):
    rows = page.locator(f'{container} .result-item')
    deadline = time.monotonic() + RENDER_TIMEOUT_S
    while rows.count() != expected_count and time.monotonic() < deadline:
        page.wait_for_timeout(200)
    shown = []
    for index in range(rows.count()):
        row = rows.nth(index)
        title = ROW_PREFIX.sub('', row.locator('.result-title').inner_text().strip())
        shown.append([title, row.locator('.result-artist').inner_text().strip()])
    return shown


def _recorded(api_golden, key):
    payload = api_golden.get(key)
    if payload is None:
        return None
    rows = payload.get('results') if isinstance(payload, dict) else payload
    return _pairs(rows)


def _clean(problems, where):
    assert not problems, f'{where}:\n' + '\n'.join(problems)


def test_similar_songs_and_playlist(flow, lib, ui_golden, request):
    page, problems = flow
    seed = lib.track('A03')
    _open(page, '/similarity', '#similarity-form')
    _pick(page, page.locator('#search_query'), page.locator('#autocomplete-results .autocomplete-item'), seed.title)
    assert page.locator('#selected_item_id').input_value(), 'the suggestion did not select a track'
    page.fill('#n', '5')
    page.uncheck('#eliminate_duplicates')
    page.uncheck('#radius_similarity')
    answer = _submit(page, '#similarity-form button[type=submit]', '/api/similar_tracks', 'GET')
    shown = _shown(page, '#results-table-wrapper', len(answer))
    assert shown == _pairs(answer), (shown, _pairs(answer))
    assert len(shown) == 5 and [seed.title, seed.artist] not in shown
    ui_golden.check('similarity: Aria da Capo e Fine, 5 songs, no duplicate elimination, no radius', shown)
    if not os.environ.get(ATTACH_ENV, '').strip():
        navidrome = request.getfixturevalue('navidrome')
        navidrome.delete_playlists_named(lambda candidate: candidate.startswith(UI_PLAYLIST_PREFIX))
        name = unique_name(UI_PLAYLIST_STEM)
        page.fill('#playlist_name', name)
        created = _submit(page, '#playlist-form button[type=submit]', '/api/create_playlist', 'POST')
        try:
            assert created.get('playlist_id'), created
            made = [p for p in navidrome.playlists() if (p.get('name') or '').startswith(name)]
            assert len(made) == 1, [p.get('name') for p in navidrome.playlists()]
            entries = navidrome.playlist_entry_ids(made[0]['id'])
            assert len(entries) >= len(shown), (len(entries), len(shown))
        finally:
            navidrome.delete_playlists_named(lambda candidate: candidate.startswith(UI_PLAYLIST_PREFIX))
    _clean(problems, '/similarity')


def test_song_path(flow, lib, ui_golden):
    page, problems = flow
    start, end = lib.track('B01'), lib.track('E03')
    _open(page, '/path', '#path-form')
    _pick(page, page.locator('#start_search'), page.locator('#start-autocomplete-results .autocomplete-item'), start.title)
    _pick(page, page.locator('#end_search'), page.locator('#end-autocomplete-results .autocomplete-item'), end.title)
    assert page.locator('#start_song_id').input_value() and page.locator('#end_song_id').input_value()
    page.fill('#max_steps', '5')
    answer = _submit(page, '#path-form button[type=submit]', '/api/find_path', 'GET')
    path = answer['path']
    shown = _shown(page, '#results-table-wrapper', len(path))
    assert shown == _pairs(path), (shown, _pairs(path))
    assert shown[0] == [start.title, start.artist] and shown[-1] == [end.title, end.artist], shown
    ui_golden.check('path: Figaro overture to the last Airmen of Note clip, 5 steps', shown)
    _clean(problems, '/path')


def test_alchemy_at_temperature_zero(flow, lib, ui_golden):
    page, problems = flow
    add, subtract = lib.track('A03'), lib.track('B01')
    _open(page, '/alchemy', '#alchemy-form')
    cards = page.locator('.alchemy-card')
    cards.nth(1).wait_for(state='attached', timeout=PICK_TIMEOUT_MS)
    for index, track in enumerate((add, subtract)):
        card = cards.nth(index)
        _pick(page, card.locator('input.song'), card.locator('.autocomplete-results .autocomplete-item'), track.title)
        assert card.locator('.song-id').input_value(), track.title
    cards.nth(1).locator('.op-sub').click()
    page.fill('#n_results', '5')
    page.fill('#temperature', '0')
    answer = _submit(page, '#run-alchemy', '/api/alchemy', 'POST')
    results = answer['results']
    shown = _shown(page, '#results-table-wrapper', len(results))
    assert shown == _pairs(results), (shown, _pairs(results))
    assert 1 <= len(shown) <= 5
    ui_golden.check('alchemy: ADD Aria da Capo e Fine, SUBTRACT Figaro overture, temperature 0, 5 songs', shown)
    _clean(problems, '/alchemy')


def test_text_search(flow, lib, ui_golden, api_golden):
    page, problems = flow
    query = lib.clap_probe['query']
    _open(page, '/clap_search', '#search-form')
    page.locator('#search-query').wait_for(state='visible', timeout=PICK_TIMEOUT_MS)
    page.fill('#search-query', query)
    page.fill('#limit', '8')
    answer = _submit(page, '#search-form button[type=submit]', '/api/clap/search', 'POST')
    shown = _shown(page, '#results-list', len(answer['results']))
    assert shown == _pairs(answer['results']), (shown, _pairs(answer['results']))
    recorded = _recorded(api_golden, 'clap search probe limit 8')
    assert recorded is None or shown == recorded, (shown, recorded)
    ui_golden.check('text search: solo piano, 8 songs', shown)
    _clean(problems, '/clap_search')


def test_lyrics_text_search(flow, lib, ui_golden, api_golden):
    page, problems = flow
    phrase = lib.lyrics['H02']['probe_phrase']
    _open(page, '/lyrics_search', '#axis-form')
    page.locator('.tab-btn[data-tab="text"]').click()
    page.locator('#tab-text #search-query').wait_for(state='visible', timeout=PICK_TIMEOUT_MS)
    page.fill('#tab-text #search-query', phrase)
    page.fill('#text-limit', '6')
    answer = _submit(page, '#tab-text button[type=submit]', '/api/lyrics/search/text', 'POST')
    results = [row for row in answer['results'] if not row.get('is_seed')]
    shown = _shown(page, '#results-list', len(results))
    assert shown == _pairs(results), (shown, _pairs(results))
    sung = lib.track('H02')
    assert [sung.title, sung.artist] in shown[:5], shown
    recorded = _recorded(api_golden, 'lyrics text search H02 phrase limit 6')
    assert recorded is None or shown == recorded, (shown, recorded)
    ui_golden.check('lyrics text search: a line of Missing Person, 6 songs', shown)
    _clean(problems, '/lyrics_search')


def test_similar_artists(flow, lib, ui_golden, api_golden):
    page, problems = flow
    artist = lib.track('E01').album_artist
    _open(page, '/artist_similarity', '#artist-similarity-form')
    _pick(page, page.locator('#artist_search'), page.locator('#autocomplete-results .autocomplete-item'), artist)
    page.fill('#n', '3')
    page.uncheck('#show_component_matches')
    answer = _submit(page, '#find-artists-btn', '/api/similar_artists', 'GET')
    names = [row.get('artist') or row.get('name') for row in answer]
    rows = page.locator('#results-table-wrapper tr.artist-row')
    deadline = time.monotonic() + RENDER_TIMEOUT_S
    while rows.count() != len(names) and time.monotonic() < deadline:
        page.wait_for_timeout(200)
    shown = [rows.nth(i).locator('td').first.inner_text().strip() for i in range(rows.count())]
    assert shown == names, (shown, names)
    assert artist not in shown and 1 <= len(shown) <= 3
    recorded = api_golden.get('similar_artists E n3')
    assert recorded is None or shown == [row.get('artist') for row in recorded], (shown, recorded)
    ui_golden.check('similar artists: Airmen of Note, 3 artists', shown)
    _clean(problems, '/artist_similarity')


def test_hyperbolic_neighbours(flow, lib, ui_golden, api_golden):
    page, problems = flow
    seed = lib.track('A03')
    _open(page, '/hyperbolic', '#hyper-similar-form')
    _pick(page, page.locator('#hyper-song-input'), page.locator('#hyper-autocomplete-results .autocomplete-item'), seed.title)
    page.fill('#hyper-limit', '5')
    answer = _submit(page, '#hyper-similar-form button[type=submit]', '/api/hyperbolic/similar', 'POST')
    shown = _shown(page, '#results-table-wrapper', len(answer['results']))
    assert shown == _pairs(answer['results']), (shown, _pairs(answer['results']))
    recorded = _recorded(api_golden, 'hyperbolic similar A03 limit 5')
    assert recorded is None or shown == recorded, (shown, recorded)
    ui_golden.check('hyperbolic neighbours: Aria da Capo e Fine, similar mode, 5 songs', shown)
    _clean(problems, '/hyperbolic')


def test_sonic_fingerprint(flow):
    page, problems = flow
    _open(page, '/sonic_fingerprint', '#fingerprint-form')
    page.locator('#navidrome_user').wait_for(state='visible', timeout=PICK_TIMEOUT_MS)
    page.fill('#navidrome_user', NAVIDROME_ADMIN_USER)
    page.fill('#navidrome_password', NAVIDROME_ADMIN_PASSWORD)
    page.fill('#n', '6')
    answer = _submit(page, '#fingerprint-form button[type=submit]', '/api/sonic_fingerprint/generate', 'POST')
    shown = _shown(page, '#results-table-wrapper', len(answer))
    assert shown == _pairs(answer), (shown, _pairs(answer))
    assert 1 <= len(shown) <= 6
    page.locator('#fingerprint-radar').wait_for(state='visible', timeout=PICK_TIMEOUT_MS)
    _clean(problems, '/sonic_fingerprint')


def test_library_browser_and_its_search(flow, lib, ui_golden):
    page, problems = flow
    with page.expect_response(
        lambda res: urllib.parse.urlsplit(res.url).path == '/api/dashboard/browse', timeout=ANSWER_TIMEOUT_MS,
    ) as first:
        _open(page, '/browse', '#browse-title')
    listing = first.value.json()
    rows = page.locator('#browse-tbody tr')
    deadline = time.monotonic() + RENDER_TIMEOUT_S
    while rows.count() != len(listing['results']) and time.monotonic() < deadline:
        page.wait_for_timeout(200)
    titles = [rows.nth(i).locator('td').nth(1).inner_text().strip() for i in range(rows.count())]
    assert titles == [row.get('title') for row in listing['results']], titles[:5]
    ui_golden.check('library browser: first page of songs', titles)
    wanted = lib.track('A02')
    fragment = wanted.title[:8]
    with page.expect_response(
        lambda res: urllib.parse.urlsplit(res.url).path == '/api/dashboard/browse' and 'q=' in res.url,
        timeout=ANSWER_TIMEOUT_MS,
    ) as searched:
        page.fill('#browse-q', fragment)
    found = searched.value.json()['results']
    deadline = time.monotonic() + RENDER_TIMEOUT_S
    while rows.count() != len(found) and time.monotonic() < deadline:
        page.wait_for_timeout(200)
    filtered = [rows.nth(i).locator('td').nth(1).inner_text().strip() for i in range(rows.count())]
    assert filtered == [row.get('title') for row in found], filtered
    assert wanted.title in filtered and len(filtered) < len(titles), filtered
    ui_golden.check('library browser: search for the start of a Goldberg variation title', filtered)
    _clean(problems, '/browse')
