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
* the Scheduled Tasks page warns, without blocking the save, when two enabled
  schedules share a minute (cron's either-day rule included), then every
  schedule is restored
"""

import json
import os
import re
import time
import urllib.parse

import pytest

from test.e2e.e2e_helpers import unique_name
from test.e2e.golden import ASR_TOLERANCE, GOLDEN_DIR, RECORD_ENV, TOLERANCE, Golden, Resolver, differences
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
ALBUM_OF_THE_WEEK_PLAYLIST = 'Album of the Week by AudioMuse-AI'
CRON_TICK_TIMEOUT_S = 150
CRON_PLAYLIST_TIMEOUT_S = 120
CRON_QUIET_S = 75
CRON_ROW_PREFIXES = ('analysis', 'clustering', 'sonic-fingerprint', 'album-of-the-week', 'alchemy-radio')
CRON_ROW_KEYS = ('id', 'name', 'task_type', 'cron_expr', 'enabled', 'options')
CRON_OVERLAP_RISK = 'at your own risk'


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
    assert response is not None, f'{path}: {response and response.status}'
    assert response.ok, f'{path}: {response and response.status}'
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


def _ranking(shown, rows, score_key):
    assert len(shown) == len(rows), (shown, rows)
    return [
        {'item_id': f'{title} | {artist}', score_key: row.get(score_key)}
        for (title, artist), row in zip(shown, rows)
    ]


def _same_as_the_recorded_api_answer(api_golden, key, ranking, score_key, tolerance=TOLERANCE):
    payload = api_golden.get(key)
    if payload is None:
        return
    rows = payload.get('results') if isinstance(payload, dict) else payload
    recorded = [
        {'item_id': f"{row.get('title') or 'Unknown'} | {row.get('author') or row.get('artist') or ''}", score_key: row.get(score_key)}
        for row in rows if not row.get('is_seed')
    ]
    lines = differences(recorded, ranking, tolerance=tolerance)
    assert not lines, f'the page shows a different ranking than the recorded API answer {key!r}:\n' + '\n'.join(lines)


def _clean(problems, where):
    assert not problems, f'{where}:\n' + '\n'.join(problems)


def _similar_songs(page, seed):
    _open(page, '/similarity', '#similarity-form')
    _pick(page, page.locator('#search_query'), page.locator('#autocomplete-results .autocomplete-item'), seed.title)
    assert page.locator('#selected_item_id').input_value(), 'the suggestion did not select a track'
    page.fill('#n', '5')
    page.uncheck('#eliminate_duplicates')
    page.uncheck('#radius_similarity')
    answer = _submit(page, '#similarity-form button[type=submit]', '/api/similar_tracks', 'GET')
    shown = _shown(page, '#results-table-wrapper', len(answer))
    assert shown == _pairs(answer), (shown, _pairs(answer))
    return shown, answer


def test_similar_songs(flow, lib, ui_golden):
    page, problems = flow
    seed = lib.track('A03')
    shown, answer = _similar_songs(page, seed)
    assert len(shown) == 5
    assert [seed.title, seed.artist] not in shown
    ui_golden.check('similarity: Aria da Capo e Fine, 5 songs, no duplicate elimination, no radius', _ranking(shown, answer, 'distance'))
    _clean(problems, '/similarity')


@pytest.mark.skipif(bool(os.environ.get(ATTACH_ENV, '').strip()), reason='needs the harness Navidrome client, absent when attached to a held stack')
def test_playlist_created_from_the_similarity_page(flow, lib, navidrome):
    page, problems = flow
    shown, _answer = _similar_songs(page, lib.track('A03'))
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
    assert page.locator('#start_song_id').input_value()
    assert page.locator('#end_song_id').input_value()
    page.fill('#max_steps', '5')
    answer = _submit(page, '#path-form button[type=submit]', '/api/find_path', 'GET')
    path = answer['path']
    shown = _shown(page, '#results-table-wrapper', len(path))
    assert shown == _pairs(path), (shown, _pairs(path))
    assert shown[0] == [start.title, start.artist], shown
    assert shown[-1] == [end.title, end.artist], shown
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
    ui_golden.check('alchemy: ADD Aria da Capo e Fine, SUBTRACT Figaro overture, temperature 0, 5 songs', _ranking(shown, results, 'distance'))
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
    ranking = _ranking(shown, answer['results'], 'similarity')
    _same_as_the_recorded_api_answer(api_golden, 'clap search probe limit 8', ranking, 'similarity')
    ui_golden.check('text search: solo piano, 8 songs', ranking)
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
    ranking = _ranking(shown, results, 'similarity')
    assert ranking[0]['item_id'] == f'{sung.title} | {sung.artist}', ranking
    _same_as_the_recorded_api_answer(api_golden, 'lyrics text search H02 phrase limit 6', ranking, 'similarity', ASR_TOLERANCE)
    ui_golden.check('lyrics text search: a line of Missing Person, 6 songs', ranking, tolerance=ASR_TOLERANCE)
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
    assert artist not in shown
    assert 1 <= len(shown) <= 3
    ranking = [{'artist': name, 'divergence': row.get('divergence')} for name, row in zip(shown, answer)]
    recorded = api_golden.get('similar_artists E n3')
    if recorded is not None:
        lines = differences([{'artist': row.get('artist'), 'divergence': row.get('divergence')} for row in recorded], ranking)
        assert not lines, lines
    ui_golden.check('similar artists: Airmen of Note, 3 artists', ranking)
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
    ranking = _ranking(shown, answer['results'], 'distance')
    _same_as_the_recorded_api_answer(api_golden, 'hyperbolic similar A03 limit 5', ranking, 'distance')
    ui_golden.check('hyperbolic neighbours: Aria da Capo e Fine, similar mode, 5 songs', ranking)
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


def test_album_creation(flow, lib):
    page, problems = flow
    seed = lib.track('B01')
    _open(page, '/album_creation', '#album-creation-form')
    assert page.locator('#create-album-btn').is_disabled()
    page.locator('.seed-types button[data-seed="text"]').click()
    assert page.locator('#seed-query-label').inner_text().strip() == 'Text search:'
    assert not page.locator('#seed-hint').is_hidden()
    page.locator('.seed-types button[data-seed="song"]').click()
    assert page.locator('.seed-types button[data-seed="song"]').get_attribute('aria-pressed') == 'true'
    _pick(page, page.locator('#seed_query'), page.locator('#seed-suggestions .autocomplete-item'), seed.title)
    assert seed.title in page.locator('#seed-selected').inner_text()
    assert page.locator('#create-album-btn').is_enabled()
    answer = _submit(page, '#create-album-btn', '/api/album_creation/generate', 'POST')
    tracks = answer['tracks']
    shown = _shown(page, '#results-table-wrapper', len(tracks))
    assert shown == _pairs(tracks), (shown, _pairs(tracks))
    assert [seed.title, seed.artist] in shown, shown
    badges = page.locator('#results-table-wrapper .similarity-badge').all_inner_texts()
    assert len(badges) == len(tracks), badges
    assert badges[:3] == ['Opener', 'Single', 'Single'], badges
    assert badges[-1] == 'Closer', badges
    assert set(badges[3:-1]) <= {'Track'}, badges
    assert page.locator('#stat-tracks').inner_text().strip() == str(len(tracks))
    page.locator('#playlist-creator').wait_for(state='visible', timeout=PICK_TIMEOUT_MS)
    assert page.locator('#playlist_name').input_value() == answer['suggested_name']
    _clean(problems, '/album_creation')


@pytest.mark.skipif(bool(os.environ.get(ATTACH_ENV, '').strip()), reason='needs the harness Navidrome client, absent when attached to a held stack')
def test_playlist_created_from_the_album_creation_page(flow, lib, navidrome):
    page, problems = flow
    _open(page, '/album_creation', '#album-creation-form')
    _pick(page, page.locator('#seed_query'), page.locator('#seed-suggestions .autocomplete-item'), lib.track('B01').title)
    answer = _submit(page, '#create-album-btn', '/api/album_creation/generate', 'POST')
    page.locator('#playlist-creator').wait_for(state='visible', timeout=PICK_TIMEOUT_MS)
    navidrome.delete_playlists_named(lambda candidate: candidate.startswith('e2e-ui-album'))
    name = unique_name('ui-album')
    page.fill('#playlist_name', name)
    created = _submit(page, '#playlist-form button[type=submit]', '/api/create_playlist', 'POST')
    try:
        assert created.get('playlist_id'), created
        assert navidrome.playlist_entry_ids(created['playlist_id']) == [track['item_id'] for track in answer['tracks']]
        page.locator('#playlist-status.status-success').wait_for(state='visible', timeout=PICK_TIMEOUT_MS)
    finally:
        navidrome.delete_playlists_named(lambda candidate: candidate.startswith('e2e-ui-album'))
    _clean(problems, '/album_creation')


def _cron_row(api, task_type):
    return next((row for row in api.json('GET', '/api/cron') if row['task_type'] == task_type), None)


def _save_schedules(page, saved):
    page.wait_for_function("!document.getElementById('save-btn').disabled", timeout=PICK_TIMEOUT_MS)
    before = len(saved)
    page.click('#save-btn')
    deadline = time.monotonic() + RENDER_TIMEOUT_S * 2
    while len(saved) == before and time.monotonic() < deadline:
        page.wait_for_timeout(200)
    assert saved[before:] == ['Saved'], saved


@pytest.mark.skipif(bool(os.environ.get(ATTACH_ENV, '').strip()), reason='needs the harness API and Navidrome clients, absent when attached to a held stack')
def test_album_of_the_week_is_scheduled_and_disabled_from_the_scheduled_tasks_page(flow, api, navidrome):
    page, problems = flow
    saved = []
    page.on('dialog', lambda dialog: (saved.append(dialog.message), dialog.accept()))
    api.wait_idle(180)
    navidrome.delete_playlists_named(lambda name: name == ALBUM_OF_THE_WEEK_PLAYLIST)
    stamped_before = (_cron_row(api, 'album_of_the_week') or {}).get('last_run')
    try:
        _open(page, '/cron', '#album-of-the-week-cron')
        page.wait_for_function("!document.getElementById('save-btn').disabled", timeout=PICK_TIMEOUT_MS)
        assert not page.is_checked('#album-of-the-week-enabled')
        page.fill('#album-of-the-week-cron', '* * * * *')
        page.check('#album-of-the-week-enabled')
        _save_schedules(page, saved)
        row = _cron_row(api, 'album_of_the_week')
        assert row['enabled'] is True, row
        assert row['cron_expr'] == '* * * * *', row
        assert row['name'] == 'Album of the Week', row

        deadline = time.monotonic() + CRON_TICK_TIMEOUT_S
        while _cron_row(api, 'album_of_the_week').get('last_run') in (None, stamped_before):
            assert time.monotonic() < deadline, 'the cron loop never fired the row scheduled from the page'
            time.sleep(5)

        page.reload(wait_until='domcontentloaded')
        page.wait_for_function("!document.getElementById('save-btn').disabled", timeout=PICK_TIMEOUT_MS)
        assert page.is_checked('#album-of-the-week-enabled'), 'the page does not show the schedule it saved'
        assert page.input_value('#album-of-the-week-cron') == '* * * * *'
        page.uncheck('#album-of-the-week-enabled')
        page.fill('#album-of-the-week-cron', '30 0 * * 6')
        _save_schedules(page, saved)
        row = _cron_row(api, 'album_of_the_week')
        assert row['enabled'] is False, row
        assert row['cron_expr'] == '30 0 * * 6', row

        deadline = time.monotonic() + CRON_PLAYLIST_TIMEOUT_S
        playlist = navidrome.playlist_by_name(ALBUM_OF_THE_WEEK_PLAYLIST)
        while playlist is None and time.monotonic() < deadline:
            time.sleep(2)
            playlist = navidrome.playlist_by_name(ALBUM_OF_THE_WEEK_PLAYLIST)
        assert playlist is not None, [p.get('name') for p in navidrome.playlists()]
        api.wait_idle(300)
        entries = navidrome.playlist_entry_ids(playlist['id'])
        assert 4 <= len(entries) <= 12, entries
        assert len(entries) == len(set(entries)), entries

        stamped_after = _cron_row(api, 'album_of_the_week')['last_run']
        time.sleep(CRON_QUIET_S)
        assert _cron_row(api, 'album_of_the_week')['last_run'] == stamped_after, 'the row kept firing after it was disabled from the page'
    finally:
        row = _cron_row(api, 'album_of_the_week')
        if row and row['enabled']:
            api.json('POST', '/api/cron', json={'id': row['id'], 'name': row['name'], 'task_type': row['task_type'], 'cron_expr': '30 0 * * 6', 'enabled': False})
        navidrome.delete_playlists_named(lambda name: name == ALBUM_OF_THE_WEEK_PLAYLIST)
    _clean(problems, '/cron')


def _cron_rows(api):
    return {row['task_type']: row for row in api.json('GET', '/api/cron')}


def _row_state(row):
    return {key: row[key] for key in CRON_ROW_KEYS}


def _far_away_month():
    return (time.localtime().tm_mon + 5) % 12 + 1


def _schedule(page, prefix, cron_expr, enabled=True):
    page.fill(f'#{prefix}-cron', cron_expr)
    page.set_checked(f'#{prefix}-enabled', enabled)


def _overlap_warning(page, visible):
    page.locator('#cron-overlap-warning').wait_for(state='visible' if visible else 'hidden', timeout=RENDER_TIMEOUT_S * 1000)


def _warns_about(page, *names):
    _overlap_warning(page, True)
    text = page.locator('#cron-overlap-text').inner_text()
    assert all(name in text for name in names), (names, text)
    assert CRON_OVERLAP_RISK in text, text


def _restore_schedules(api, before, opened):
    for task_type, row in _cron_rows(api).items():
        previous = before.get(task_type) or dict(row, cron_expr=opened.get(task_type, row['cron_expr']), enabled=False)
        api.json('POST', '/api/cron', json=_row_state(previous))


@pytest.mark.skipif(bool(os.environ.get(ATTACH_ENV, '').strip()), reason='needs the harness API client, absent when attached to a held stack')
def test_the_scheduled_tasks_page_warns_when_two_schedules_share_a_minute_and_still_saves(flow, api):
    page, problems = flow
    saved = []
    page.on('dialog', lambda dialog: (saved.append(dialog.message), dialog.accept()))
    before = _cron_rows(api)
    opened = {}
    month = _far_away_month()
    try:
        _open(page, '/cron', '#cron-overlap-warning')
        page.wait_for_function("!document.getElementById('save-btn').disabled", timeout=PICK_TIMEOUT_MS)
        opened = {prefix.replace('-', '_'): page.input_value(f'#{prefix}-cron') for prefix in CRON_ROW_PREFIXES}
        boxes = page.locator('#main-content-inner input[type=checkbox]')
        for index in range(boxes.count()):
            boxes.nth(index).uncheck()
        _overlap_warning(page, False)

        _schedule(page, 'analysis', '0 2 * * *')
        _schedule(page, 'clustering', '0 2 * * 6')
        _warns_about(page, 'Analysis', 'Clustering')
        _schedule(page, 'clustering', '5 2 * * *')
        _overlap_warning(page, False)
        _schedule(page, 'analysis', '0 2 * * 1')
        _schedule(page, 'clustering', '0 2 * * 2')
        _overlap_warning(page, False)
        _schedule(page, 'analysis', '0 2 1 * 1')
        _warns_about(page, 'Analysis', 'Clustering')
        page.uncheck('#clustering-enabled')
        _overlap_warning(page, False)

        _schedule(page, 'analysis', f'0 2 * {month} *')
        _schedule(page, 'clustering', f'0 2 * {month} 6')
        _warns_about(page, 'Analysis', 'Clustering')
        _save_schedules(page, saved)
        stored = _cron_rows(api)
        assert (stored['analysis']['cron_expr'], stored['analysis']['enabled']) == (f'0 2 * {month} *', True), stored
        assert (stored['clustering']['cron_expr'], stored['clustering']['enabled']) == (f'0 2 * {month} 6', True), stored
        assert not [task_type for task_type, row in stored.items() if row['enabled'] and task_type not in ('analysis', 'clustering')], stored
        _warns_about(page, 'Analysis', 'Clustering')
    finally:
        _restore_schedules(api, before, opened)
    restored = _cron_rows(api)
    assert {task_type: _row_state(restored[task_type]) for task_type in before} == {task_type: _row_state(row) for task_type, row in before.items()}
    assert not [task_type for task_type, row in restored.items() if row['enabled'] and task_type not in before], restored
    fired = [task_type for task_type, row in restored.items() if not (before.get(task_type) or {}).get('enabled') and row['last_run'] != (before.get(task_type) or {}).get('last_run')]
    assert not fired, (fired, restored)
    _clean(problems, '/cron')


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
    assert wanted.title in filtered, filtered
    assert len(filtered) < len(titles), filtered
    ui_golden.check('library browser: search for the start of a Goldberg variation title', filtered)
    _clean(problems, '/browse')
