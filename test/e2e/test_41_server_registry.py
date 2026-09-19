# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The music-server registry with two real Navidrome instances.

The second Navidrome serves the same files under its own ids, so registering
it must make the alignment sweep map every analyzed file onto the catalogue
rows the default server already owns (docs/MULTI_SERVER.md), the per-server
endpoints answer for it, promoting it to default publishes a worker restart
that the control plane acknowledges, and deleting it cascades every row bound
to its id while the default server can never be deleted.

Main Features:
* POST /api/servers/test answers ok for the real credentials and not ok for
  a wrong password; /api/servers/libraries lists the Navidrome folder
* adding the second instance enqueues a real sweep that binds every catalogue
  row to its ids
* align and per-server sweep run as queue tasks and succeed
* the default swap round-trips with an acknowledged worker restart
* delete cascades track_server_map, artist_server_map, chromaprint and playlist
"""

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, item_id_of, provider_ids_for, rows, scalar
from test.e2e.stack.env import NAVIDROME_ADMIN_PASSWORD, NAVIDROME_ADMIN_USER

pytestmark = pytest.mark.e2e

SWEEP_TIMEOUT = 300


def _creds(url, password=NAVIDROME_ADMIN_PASSWORD):
    return {'url': url, 'user': NAVIDROME_ADMIN_USER, 'password': password, 'api_key': ''}


def _default_id(db):
    return scalar(db, 'SELECT server_id FROM music_servers WHERE is_default')


def test_default_server_listed(stack, api, analyzed_library):
    payload = api.json('GET', '/api/servers')
    assert_no_fp_ids(payload)
    servers = payload['servers']
    assert len(servers) == 1
    assert servers[0]['server_type'] == 'navidrome'
    assert servers[0]['is_default'] is True
    assert servers[0]['server_id'] == payload['default_id']


def test_probe_endpoints(stack, api, analyzed_library):
    ok = api.json('POST', '/api/servers/test', json={'server_type': 'navidrome', 'creds': _creds(stack.navidrome2.base_url)})
    assert ok['ok'] is True, ok
    assert ok.get('sample_count', 0) >= 1, ok
    bad = api.json('POST', '/api/servers/test', json={'server_type': 'navidrome', 'creds': _creds(stack.navidrome2.base_url, 'wrong')})
    assert bad['ok'] is False, bad
    assert bad.get('auth_failed') is True, bad
    libraries = api.json('POST', '/api/servers/libraries', json={'server_type': 'navidrome', 'creds': _creds(stack.navidrome2.base_url)})
    assert isinstance(libraries.get('libraries'), list)
    unsupported = api.post('/api/servers/test', json={'server_type': 'nope', 'creds': {}})
    assert unsupported.status_code == 400


def test_second_instance_lifecycle(stack, api, db, library, analyzed_library):
    api.wait_idle(120)
    original_default = _default_id(db)
    created = api.json(
        'POST', '/api/servers', expect=201,
        json={'name': 'e2e-second', 'server_type': 'navidrome', 'creds': _creds(stack.navidrome2.base_url)},
    )
    assert_no_fp_ids(created)
    server_id = created['server_id']
    assert created['is_default'] is False
    assert created['sweep_task_id']
    second_songs = {s['id'] for s in stack.subsonic2.all_songs()}
    try:
        final = api.wait_for_task(created['sweep_task_id'], timeout=SWEEP_TIMEOUT)
        assert final['task_type_from_db'] == 'server_sweep', final
        mapped = rows(db, 'SELECT item_id, provider_track_id FROM track_server_map WHERE server_id = %s', (server_id,))
        assert stack.catalogue_rows <= len(mapped) <= stack.analyzable_files, (len(mapped), final)
        assert len({m[0] for m in mapped}) == stack.catalogue_rows, final
        assert all(m[1] in second_songs for m in mapped), 'the second server must be bound to its own ids'
        default_items = {r[0] for r in rows(db, 'SELECT item_id FROM track_server_map WHERE server_id = %s', (original_default,))}
        assert {m[0] for m in mapped} == default_items
        assert scalar(db, 'SELECT count(*) FROM artist_server_map WHERE server_id = %s', (server_id,)) >= library.counts['clip_artists']
        assert scalar(db, 'SELECT track_count FROM music_servers WHERE server_id = %s', (server_id,)) == stack.expected_files

        listed = api.json('GET', '/api/servers')
        assert len(listed['servers']) == 2
        assert listed.get('multi_server_enabled') is True

        aligned = api.json('POST', '/api/servers/align', expect=202)
        assert aligned['enqueued'] is True
        api.wait_for_task(aligned['task_id'], timeout=SWEEP_TIMEOUT)
        api.wait_idle(60)
        swept = api.json('POST', f'/api/servers/{server_id}/sweep', expect=202)
        api.wait_for_task(swept['task_id'], timeout=SWEEP_TIMEOUT)
        api.wait_idle(60)

        second_ids = provider_ids_for(db, item_id_of(db, library.pid('A03')), server_id)
        assert len(second_ids) == 1 and second_ids[0] in second_songs, second_ids
        similar = api.json('GET', f'/api/similar_tracks?item_id={second_ids[0]}&n=3&server=e2e-second')
        assert_no_fp_ids(similar)
        assert similar and all(r['item_id'] in second_songs for r in similar), similar
        playlists = api.json('GET', '/api/playlists')
        assert playlists.get('multi_server') is True
        assert len(playlists['servers']) <= 2

        promoted = api.json('POST', f'/api/servers/{server_id}/default')
        assert promoted.get('restart_acknowledged', True) is True, promoted
        assert promoted['default_id'] == server_id, promoted
        if promoted.get('sweep_task_id'):
            api.wait_for_task(promoted['sweep_task_id'], timeout=SWEEP_TIMEOUT)
        for worker in stack.workers.values():
            worker.wait_ready(180)
        assert _default_id(db) == server_id
        assert api.json('GET', f'/api/track?item_id={next(iter(second_songs))}')['title']
        restored = api.json('POST', f'/api/servers/{original_default}/default')
        assert restored.get('restart_acknowledged', True) is True, restored
        if restored.get('sweep_task_id'):
            api.wait_for_task(restored['sweep_task_id'], timeout=SWEEP_TIMEOUT)
        for worker in stack.workers.values():
            worker.wait_ready(180)
        assert _default_id(db) == original_default
        api.wait_idle(60)

        renamed = api.json('PUT', f'/api/servers/{server_id}', json={'name': 'e2e-second-b'})
        assert renamed['name'] == 'e2e-second-b'
        refused = api.delete(f'/api/servers/{original_default}')
        assert 400 <= refused.status_code < 500, refused.text
    finally:
        deleted = api.json('DELETE', f'/api/servers/{server_id}')
        assert deleted == {'deleted': server_id}
    for table in ('track_server_map', 'artist_server_map', 'chromaprint', 'analysis_exclusions', 'playlist'):
        assert scalar(db, f'SELECT count(*) FROM {table} WHERE server_id = %s', (server_id,)) == 0, table
    assert len(api.json('GET', '/api/servers')['servers']) == 1
    assert api.delete(f'/api/servers/{server_id}').status_code == 404
    assert _default_id(db) == original_default
    assert scalar(db, 'SELECT count(*) FROM track_server_map WHERE server_id = %s', (original_default,)) == stack.analyzable_files


def test_add_server_validation(stack, api, analyzed_library):
    assert api.post('/api/servers', json={'name': '', 'server_type': 'navidrome', 'creds': _creds(stack.navidrome2.base_url)}).status_code == 400
    assert api.post('/api/servers', json={'name': 'e2e-x', 'server_type': 'nope', 'creds': {}}).status_code == 400
    missing = api.post('/api/servers', json={'name': 'e2e-x', 'server_type': 'navidrome', 'creds': {'url': stack.navidrome2.base_url}})
    assert missing.status_code == 400, missing.text
    default_name = api.json('GET', '/api/servers')['servers'][0]['name']
    taken = api.post('/api/servers', json={'name': default_name, 'server_type': 'navidrome', 'creds': _creds(stack.navidrome2.base_url)})
    assert taken.status_code == 400, taken.text
