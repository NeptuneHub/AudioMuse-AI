# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Backup and restore: a real pg_dump of the analyzed catalogue and its round trip.

Creating a backup shells out to pg_dump against the live database and the
download must contain the whole schema and data (the dump is allowed to hold
the internal catalogue ids; the API is not). The restore is the real thing:
the endpoint stops the workers through the control listener, a detached
runner stops the web process through the control socket, feeds the dump to
psql, starts the web process and the workers again and writes its outcome
marker to its log. A row written after the backup must be gone afterwards and
every count must be back to the backup's.

Main Features:
* POST /api/backup/create answers with the file name and size, and replaces
  the previous backup
* the downloaded zip holds a pg_dump with every table and the catalogue rows
* invalid or unknown download names are 404, including path traversal
* restore refuses a missing confirmation and a missing file
* a restore of the just-taken backup completes, restarts the stack and drops
  the anchor created in between
"""

import io
import os
import time
import zipfile

import pytest

from test.e2e.e2e_helpers import item_id_of, scalar, unique_name
from test.e2e.stack import postgres

pytestmark = pytest.mark.e2e

CONFIRMATION = 'I want to restore the database from the backup. This action is not reversible'
TABLES = ('score', 'embedding', 'track_server_map', 'music_servers', 'app_config')
DATA_TABLES = TABLES[:-1]
RESULT_MARKER = 'RESTORE-RESULT:'
RESTORE_TIMEOUT = 600


def _create(api):
    body = api.json('POST', '/api/backup/create', timeout=600)
    assert body['success'] is True, body
    assert body['filename'].startswith('audiomuse_backup_') and body['filename'].endswith('.zip')
    assert body['size_bytes'] > 0
    return body['filename']


def _download(api, filename):
    response = api.get(f'/api/backup/download/{filename}', timeout=600)
    assert response.status_code == 200, response.text[:300]
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()
    assert len(names) == 1 and names[0].endswith('.sql'), names
    return archive.read(names[0]).decode('utf-8', 'replace')


def test_create_and_download(stack, api, db, library, analyzed_library):
    filename = _create(api)
    dump = _download(api, filename)
    for table in TABLES:
        assert f'CREATE TABLE public.{table}' in dump, table
    assert 'COPY public.track_server_map' in dump
    assert item_id_of(db, library.pid('A01')) in dump
    time.sleep(1.1)
    replacement = _create(api)
    assert replacement != filename
    assert api.get(f'/api/backup/download/{filename}').status_code == 404
    assert api.get(f'/api/backup/download/{replacement}').status_code == 200


def test_download_rejects_bad_names(stack, api, analyzed_library):
    assert api.get('/api/backup/download/audiomuse_backup_00000000_000000.zip').status_code == 404
    assert api.get('/api/backup/download/..%2F..%2Fetc%2Fpasswd').status_code == 404
    assert api.get('/api/backup/download/notabackup.zip').status_code == 404


def _wait_restore_result(log_path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.isfile(log_path):
            with open(log_path, encoding='utf-8', errors='replace') as handle:
                text = handle.read()
            for line in text.splitlines():
                if line.startswith(RESULT_MARKER):
                    return line[len(RESULT_MARKER):].split()[0], text
        time.sleep(2)
    raise AssertionError(f'no {RESULT_MARKER} in {log_path} after {timeout}s')


def test_restore_round_trip(stack, api, db, analyzed_library):
    api.wait_idle(180)
    before = {t: scalar(db, f'SELECT count(*) FROM {t}') for t in DATA_TABLES}
    anchors_before = scalar(db, 'SELECT count(*) FROM alchemy_anchors')
    filename = _create(api)
    payload = api.get(f'/api/backup/download/{filename}', timeout=600).content
    no_confirmation = api.post('/api/backup/restore', data={'confirmation': 'no'}, files={'file': (filename, payload)})
    assert no_confirmation.status_code == 400, no_confirmation.text
    no_file = api.post('/api/backup/restore', data={'confirmation': CONFIRMATION})
    assert no_file.status_code == 400, no_file.text
    centroid = [0.01] * 200
    anchor_id = api.json('POST', '/api/anchors', json={'name': unique_name('restore-anchor'), 'centroid': centroid})['anchor']['id']
    assert scalar(db, 'SELECT count(*) FROM alchemy_anchors') == anchors_before + 1

    started = api.json(
        'POST', '/api/backup/restore', data={'confirmation': CONFIRMATION},
        files={'file': (filename, payload)}, timeout=300,
    )
    assert started['success'] is True, started
    assert started['restore_log_name'].startswith('restore_'), started
    result, log_text = _wait_restore_result(started['restore_log'], RESTORE_TIMEOUT)
    assert result == 'completed', log_text[-2000:]
    assert 'Stopped local Flask service.' in log_text and 'Started local Flask service.' in log_text, log_text[-2000:]
    stack.flask.wait_ready(300)
    for worker in stack.workers.values():
        worker.wait_ready(180)
    api.wait_idle(300)

    fresh = postgres.connect(stack.dsn)
    try:
        after = {t: scalar(fresh, f'SELECT count(*) FROM {t}') for t in DATA_TABLES}
        assert after == before, (after, before)
        assert scalar(fresh, 'SELECT count(*) FROM alchemy_anchors') == anchors_before
    finally:
        fresh.close()
    assert anchor_id not in {a['id'] for a in api.json('GET', '/api/anchors')['anchors']}
    assert api.health()
    assert any(action == 'stop' and 'flask' in services for action, services in stack.control.requests), stack.control.requests
    assert api.json('GET', '/api/search_tracks?search_query=a'), 'the catalogue must be searchable after the restore'
