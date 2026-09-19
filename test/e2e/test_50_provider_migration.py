# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Provider migration from one Navidrome instance to another, and back.

The wizard's whole API sequence runs against the second Navidrome instance
that serves the same files under different ids: start a session, probe the
target, list its libraries, run the dry-run planner on the queue, read the
report, finalize the counts, execute with the confirmation phrase, wait for
the job (whose restart handshake completes through the harness control
plane), and assert the default server now points at the second instance with
every mapping repointed at its ids. The same steps then migrate back, so the
rest of the suite sees the original server.

Main Features:
* session start validation: unsupported target, incomplete credentials, a
  target already registered as a secondary server
* dry-run matches every catalogue row by path with zero orphans
* execute repoints track_server_map and music_servers, keeps the catalogue,
  and the queued alignment leaves the same rows bound to the new ids
* migrating back restores the original default server and ids
"""

import re
import time

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, rows, scalar
from test.e2e.stack.env import NAVIDROME_ADMIN_PASSWORD, NAVIDROME_ADMIN_USER

pytestmark = pytest.mark.e2e

CONFIRMATION = 'I want to migrate to navidrome and unbind unmatched tracks'
FP_ID = re.compile(r'\bfp_[0-9a-f]{40,}\b')
PLANNER_TIMEOUT = 300
EXECUTE_TIMEOUT = 600


def _creds(url):
    return {'url': url, 'user': NAVIDROME_ADMIN_USER, 'password': NAVIDROME_ADMIN_PASSWORD, 'api_key': ''}


def _default_url(db):
    creds = rows(db, 'SELECT creds FROM music_servers WHERE is_default')[0][0]
    return (creds or {}).get('url', '').rstrip('/')


def _wait_status(api, task_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = api.json('GET', f'/api/migration/status/{task_id}')
        if body.get('status') in ('SUCCESS', 'FAIL', 'REVOKED'):
            return body
        time.sleep(2)
    raise AssertionError(f'migration job {task_id} not terminal after {timeout}s: {body}')


def _migrate(stack, api, db, target, expected_rows, golden, label):
    api.wait_idle(300)
    session = api.json('POST', '/api/migration/session/start', json={'target_type': 'navidrome', 'target_creds': _creds(target.base_url)})
    session_id = session['session_id']
    probe = api.json('POST', '/api/migration/probe/test', json={'type': 'navidrome', 'creds': _creds(target.base_url)})
    assert probe['ok'] is True, probe
    libraries = api.json('POST', '/api/migration/libraries', json={'session_id': session_id})
    assert isinstance(libraries['libraries'], list)
    selected = api.json('POST', '/api/migration/libraries/select', json={'session_id': session_id, 'libraries': None})
    assert selected['ok'] is True
    planner = api.json('POST', '/api/migration/dry-run', json={'session_id': session_id})
    assert planner.get('async') is True and planner['task_id']
    planned = _wait_status(api, planner['task_id'], PLANNER_TIMEOUT)
    assert planned['status'] == 'SUCCESS', planned
    report = api.get(f'/api/migration/dry-run-report/{session_id}')
    assert report.status_code == 200, report.text[:300]
    assert 'text/csv' in report.headers.get('Content-Type', ''), report.headers
    assert report.text.startswith('old_id,old_artist,old_album'), report.text[:200]
    assert not FP_ID.search(report.text), 'the report must list provider ids, never catalogue ids'
    final = api.json('POST', '/api/migration/finalize-dry-run', json={'session_id': session_id})
    golden.check(f'finalize dry run {label}', final)
    assert final['matched'] == expected_rows, final
    assert final['orphans'] == 0 and final['collisions'] == 0, final
    refused = api.post('/api/migration/execute', json={'session_id': session_id, 'backup_confirmed': True, 'confirmation_text': 'nope'})
    assert refused.status_code == 400, refused.text
    unconfirmed = api.post('/api/migration/execute', json={'session_id': session_id, 'backup_confirmed': False, 'confirmation_text': CONFIRMATION})
    assert unconfirmed.status_code == 400, unconfirmed.text
    execute = api.json('POST', '/api/migration/execute', json={'session_id': session_id, 'backup_confirmed': True, 'confirmation_text': CONFIRMATION})
    done = _wait_status(api, execute['task_id'], EXECUTE_TIMEOUT)
    assert done['status'] == 'SUCCESS', done
    api.wait_idle(300)
    for worker in stack.workers.values():
        worker.wait_ready(180)
    session_row = api.json('GET', f'/api/migration/session/{session_id}')
    assert session_row.get('status') == 'completed', session_row
    return session_id


def _assert_bound_to(db, library, target_client, expected_rows):
    songs = {s['id']: s for s in target_client.all_songs()}
    mapped = rows(db, 'SELECT item_id, provider_track_id, file_path FROM track_server_map WHERE server_id = (SELECT server_id FROM music_servers WHERE is_default)')
    assert len({m[0] for m in mapped}) == expected_rows, len({m[0] for m in mapped})
    assert all(m[1] in songs for m in mapped), 'a mapping points at an id the target does not serve'
    for key in library.clip_keys():
        title = library.track(key).title
        assert any(songs[m[1]].get('title') == title for m in mapped if m[1] in songs), key


def test_session_start_validation(stack, api, analyzed_library):
    assert api.post('/api/migration/session/start', json={'target_type': 'nope', 'target_creds': {}}).status_code == 400
    incomplete = api.post('/api/migration/session/start', json={'target_type': 'navidrome', 'target_creds': {'url': stack.navidrome2.base_url}})
    assert incomplete.status_code == 400, incomplete.text
    registered = api.json('POST', '/api/servers', expect=201, json={'name': 'e2e-mig-block', 'server_type': 'navidrome', 'creds': _creds(stack.navidrome2.base_url)})
    try:
        api.wait_for_task(registered['sweep_task_id'], timeout=300)
        conflict = api.post('/api/migration/session/start', json={'target_type': 'navidrome', 'target_creds': _creds(stack.navidrome2.base_url)})
        assert conflict.status_code == 409, conflict.text
    finally:
        api.json('DELETE', f"/api/servers/{registered['server_id']}")
    api.wait_idle(120)


def test_migrate_to_second_instance_and_back(stack, api, db, library, analyzed_library, golden):
    original_url = _default_url(db)
    assert original_url == stack.navidrome.base_url
    rows_before = scalar(db, 'SELECT count(*) FROM score')
    expected_rows = stack.catalogue_rows
    assert rows_before == expected_rows

    _migrate(stack, api, db, stack.navidrome2, expected_rows, golden, 'to the second instance')
    assert _default_url(db) == stack.navidrome2.base_url
    assert scalar(db, 'SELECT count(*) FROM score') == rows_before
    _assert_bound_to(db, library, stack.subsonic2, expected_rows)
    assert scalar(db, 'SELECT count(*) FROM music_servers') == 1
    assert api.json('GET', '/api/servers')['servers'][0]['creds']['url'].rstrip('/') == stack.navidrome2.base_url
    similar = api.json('GET', f"/api/search_tracks?search_query={library.track('B01').title[:12]}")
    assert_no_fp_ids(similar)
    assert similar, 'the catalogue must still be searchable after the migration'

    _migrate(stack, api, db, stack.navidrome, expected_rows, golden, 'back to the first instance')
    assert _default_url(db) == original_url
    assert scalar(db, 'SELECT count(*) FROM score') == rows_before
    stack.library.bind_provider_ids(stack.subsonic.all_songs())
    _assert_bound_to(db, library, stack.subsonic, expected_rows)
    assert api.json('GET', f"/api/track?item_id={library.pid('A03')}")['title'] == library.track('A03').title
