# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Scheduled tasks (ALGORITHM.md section 16): the cron rows and one real tick.

Rows for every schedulable task type are created, listed, renamed, upserted
by type and validated. Then two rows are enabled every minute and the web
process's cron loop is left to fire: only ONE online row runs per tick (the
alchemy radio, first in the fixed order), the sonic fingerprint due in the
same minute is skipped unclaimed, and it runs on a later tick once the radio
is disabled; each writes its playlist on Navidrome and each row then shows a
last_run newer than the one it had (a kept state carries the stamp of the
run before). The stamp is written when the tick claims the row, before the
task runs, so each playlist is then awaited on Navidrome for a bounded time.
The run that finishes second trims the first one's recap row from
task_status, so the fingerprint is judged by the playlist it wrote; a recap
row that is still there must say SUCCESS.

Main Features:
* POST /api/cron creates and updates rows, GET /api/cron lists them
* an enabled row with a bad expression and a non-object options are 400
* one cron tick runs ONE online row inline in Flask; the other due in the
  same minute is skipped unclaimed and runs on a later tick
* one cron tick runs the album of the week inline into ONE fixed playlist
* every row is disabled again at the end so no later tick fires
"""

import time

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, rows, unique_name

pytestmark = pytest.mark.e2e

TASK_TYPES = ('analysis', 'clustering', 'sonic_fingerprint', 'album_of_the_week', 'alchemy_radio')
NIGHTLY = '0 3 * * *'
EVERY_MINUTE = '* * * * *'
TICK_TIMEOUT = 150
PLAYLIST_TIMEOUT = 120
TICK_TASK_TYPES = ('alchemy_radio', 'sonic_fingerprint')
CRON_SONIC_PLAYLIST = 'Sonic Fingerprint by AudioMuse-AI'
CRON_ALBUM_PLAYLIST = 'Album of the Week by AudioMuse-AI'


@pytest.fixture
def disable_all_rows(api):
    yield
    for entry in api.json('GET', '/api/cron'):
        api.json(
            'POST', '/api/cron',
            json={
                'id': entry['id'], 'name': entry['name'], 'task_type': entry['task_type'],
                'cron_expr': entry['cron_expr'], 'enabled': False, 'options': entry.get('options') or {},
            },
        )


def _by_type(api):
    entries = api.json('GET', '/api/cron')
    assert_no_fp_ids(entries)
    return {e['task_type']: e for e in entries}


def _last_runs(api):
    entries = _by_type(api)
    return {task_type: (entries.get(task_type) or {}).get('last_run') for task_type in TICK_TASK_TYPES}


def _wait_for_playlist(navidrome, name):
    deadline = time.monotonic() + PLAYLIST_TIMEOUT
    while True:
        playlist = navidrome.playlist_by_name(name)
        if playlist is not None or time.monotonic() >= deadline:
            return playlist
        time.sleep(2)


def _save(api, task_type, name, cron_expr, enabled, row_id=None):
    payload = {'name': name, 'task_type': task_type, 'cron_expr': cron_expr, 'enabled': enabled}
    if row_id is not None:
        payload['id'] = row_id
    assert api.json('POST', '/api/cron', json=payload) == {'message': 'saved'}


def test_rows_round_trip(stack, api, db, disable_all_rows):
    for task_type in TASK_TYPES:
        _save(api, task_type, f'e2e {task_type}', NIGHTLY, False)
    entries = _by_type(api)
    assert set(TASK_TYPES) <= set(entries), entries
    for task_type in TASK_TYPES:
        entry = entries[task_type]
        assert entry['name'] == f'e2e {task_type}'
        assert entry['cron_expr'] == NIGHTLY
        assert entry['enabled'] is False
        assert isinstance(entry.get('options'), dict)
        assert isinstance(entry['id'], int)
    assert rows(db, 'SELECT count(*) FROM cron WHERE task_type = ANY(%s)', (list(TASK_TYPES),))[0][0] == len(TASK_TYPES)


def test_upsert_by_type_and_rename_by_id(stack, api, db, disable_all_rows):
    _save(api, 'analysis', 'e2e first', NIGHTLY, False)
    before = rows(db, "SELECT count(*) FROM cron WHERE task_type = 'analysis'")[0][0]
    _save(api, 'analysis', 'e2e second', '30 4 * * *', False)
    after = rows(db, "SELECT count(*) FROM cron WHERE task_type = 'analysis'")[0][0]
    assert before == after == 1
    entry = _by_type(api)['analysis']
    assert entry['name'] == 'e2e second'
    assert entry['cron_expr'] == '30 4 * * *'
    _save(api, 'analysis', 'e2e renamed', NIGHTLY, False, row_id=entry['id'])
    renamed = _by_type(api)['analysis']
    assert renamed['id'] == entry['id']
    assert renamed['name'] == 'e2e renamed'


def test_validation(stack, api, disable_all_rows):
    bad_expr = api.post('/api/cron', json={'name': 'e2e bad', 'task_type': 'clustering', 'cron_expr': '99 3 * * *', 'enabled': True})
    assert bad_expr.status_code == 400, bad_expr.text
    bad_options = api.post('/api/cron', json={'name': 'e2e bad', 'task_type': 'clustering', 'cron_expr': NIGHTLY, 'enabled': False, 'options': 'x'})
    assert bad_options.status_code == 400, bad_options.text
    assert api.json('GET', '/api/cron/plugin_tasks') == []


def _wait_for_fire(api, stamped_before, task_type):
    deadline = time.monotonic() + TICK_TIMEOUT
    while True:
        entries = _by_type(api)
        if entries[task_type].get('last_run') not in (None, stamped_before[task_type]):
            return entries
        assert time.monotonic() < deadline, f'the cron loop did not fire {task_type} within {TICK_TIMEOUT}s: {entries}'
        time.sleep(5)


def test_one_tick_runs_one_online_row_and_the_other_runs_once_it_is_alone(stack, api, db, library, navidrome, analyzed_library, disable_all_rows):
    api.wait_idle(180)
    centroid = api.json('GET', f'/external/get_embedding?id={library.pid("A03")}')['embedding']
    anchor_name = unique_name('cron-anchor')
    anchor_id = api.json('POST', '/api/anchors', json={'name': anchor_name, 'centroid': centroid})['anchor']['id']
    radio_id = api.json('POST', '/api/radios', json={'anchor_id': anchor_id, 'temperature': 1.0, 'n_results': 5})['radio']['id']
    for key in ('B01', 'B02', 'B03'):
        navidrome.scrobble(library.pid(key), submission=True)
    navidrome.delete_playlists_named(lambda name: name in (anchor_name, CRON_SONIC_PLAYLIST))
    stamped_before = _last_runs(api)
    try:
        _save(api, 'alchemy_radio', 'e2e radio tick', EVERY_MINUTE, True)
        _save(api, 'sonic_fingerprint', 'e2e sonic tick', EVERY_MINUTE, True)
        entries = _wait_for_fire(api, stamped_before, 'alchemy_radio')
        assert entries['sonic_fingerprint'].get('last_run') == stamped_before['sonic_fingerprint'], (
            'one online run per minute: the fingerprint due in the same minute as the '
            'radio is skipped unclaimed, so its last_run must not move'
        )
        _save(api, 'alchemy_radio', 'e2e radio tick', NIGHTLY, False, row_id=entries['alchemy_radio']['id'])
        entries = _wait_for_fire(api, stamped_before, 'sonic_fingerprint')
        _save(api, 'sonic_fingerprint', 'e2e sonic tick', NIGHTLY, False, row_id=entries['sonic_fingerprint']['id'])
        radio_playlist = _wait_for_playlist(navidrome, anchor_name)
        assert radio_playlist is not None, [p.get('name') for p in navidrome.playlists()]
        sonic_playlist = _wait_for_playlist(navidrome, CRON_SONIC_PLAYLIST)
        assert sonic_playlist is not None, [p.get('name') for p in navidrome.playlists()]
        api.wait_idle(300)
        assert 1 <= len(navidrome.playlist_entry_ids(radio_playlist['id'])) <= 5
        assert navidrome.playlist_entry_ids(sonic_playlist['id'])
        sonic_rows = rows(db, "SELECT status FROM task_status WHERE task_type = 'sonic_fingerprint' ORDER BY timestamp DESC LIMIT 1")
        assert all(row[0] == 'SUCCESS' for row in sonic_rows), sonic_rows
        assert rows(db, 'SELECT count(*) FROM cron_retry')[0][0] == 0
    finally:
        api.json('DELETE', f'/api/radios/{radio_id}')
        api.json('DELETE', f'/api/anchors/{anchor_id}')
        navidrome.delete_playlists_named(lambda name: name in (anchor_name, CRON_SONIC_PLAYLIST))


def test_no_retry_rows_pending(stack, db):
    assert rows(db, 'SELECT count(*) FROM cron_retry')[0][0] == 0


def test_one_tick_runs_the_album_of_the_week_and_the_next_refills_it(stack, api, db, navidrome, analyzed_library, disable_all_rows):
    api.wait_idle(180)
    navidrome.delete_playlists_named(lambda name: name == CRON_ALBUM_PLAYLIST)
    before = (_by_type(api).get('album_of_the_week') or {}).get('last_run')
    try:
        _save(api, 'album_of_the_week', 'e2e album tick', EVERY_MINUTE, True)
        deadline = time.monotonic() + TICK_TIMEOUT
        while True:
            entry = _by_type(api)['album_of_the_week']
            if entry.get('last_run') not in (None, before):
                break
            assert time.monotonic() < deadline, f'the cron loop did not fire within {TICK_TIMEOUT}s: {entry}'
            time.sleep(5)
        playlist = _wait_for_playlist(navidrome, CRON_ALBUM_PLAYLIST)
        assert playlist is not None, [p.get('name') for p in navidrome.playlists()]
        api.wait_idle(300)
        entries = navidrome.playlist_entry_ids(playlist['id'])
        assert 4 <= len(entries) <= 12 and len(entries) == len(set(entries)), entries

        fired_once = _by_type(api)['album_of_the_week'].get('last_run')
        deadline = time.monotonic() + TICK_TIMEOUT
        while True:
            entry = _by_type(api)['album_of_the_week']
            if entry.get('last_run') not in (None, fired_once):
                break
            assert time.monotonic() < deadline, f'the second tick never fired: {entry}'
            time.sleep(5)
        _save(api, 'album_of_the_week', 'e2e album tick', NIGHTLY, False, row_id=entry['id'])
        api.wait_idle(300)
        named = [p for p in navidrome.playlists() if p.get('name') == CRON_ALBUM_PLAYLIST]
        assert len(named) == 1, [p.get('name') for p in navidrome.playlists()]
        refilled = navidrome.playlist_entry_ids(named[0]['id'])
        assert 4 <= len(refilled) <= 12 and len(refilled) == len(set(refilled)), refilled
        album_rows = rows(db, "SELECT status FROM task_status WHERE task_type = 'album_of_the_week' ORDER BY timestamp DESC LIMIT 1")
        assert all(row[0] == 'SUCCESS' for row in album_rows), album_rows
        assert rows(db, 'SELECT count(*) FROM cron_retry')[0][0] == 0
    finally:
        navidrome.delete_playlists_named(lambda name: name == CRON_ALBUM_PLAYLIST)
