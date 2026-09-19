# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Self-test of the end-to-end harness: every process is up and wired to the others.

Runs first so a broken bring-up is reported as a harness problem, not as a
failure of the first feature test that happens to touch the stack.

Main Features:
* Flask answers /api/health and /api/config without the setup barrier
* the default navidrome server row was seeded from the environment
* Navidrome serves exactly the manifest, at the pinned version
* both queue workers are running and hold their LISTEN connection
* env.json was written with secrets masked
"""

import json
import os

import pytest

from test.e2e.e2e_helpers import rows, scalar
from test.e2e.stack.navidrome import NAVIDROME_VERSION

pytestmark = pytest.mark.e2e


def test_flask_answers_health_and_config(stack, api):
    assert api.health()
    config = api.json('GET', '/api/config')
    assert isinstance(config, dict)
    assert config


def test_default_server_seeded_from_env(stack, db):
    servers = rows(db, 'SELECT server_type, is_default, name FROM music_servers')
    assert len(servers) == 1, servers
    assert servers[0][0] == 'navidrome'
    assert servers[0][1] is True


def test_navidrome_serves_the_manifest(stack, navidrome, library):
    assert navidrome.server_version().startswith(NAVIDROME_VERSION), navidrome.server_version()
    assert navidrome.song_count() == stack.expected_files
    assert len(navidrome.albums()) == library.counts['album_folders'] + len(stack.seed.albums)
    if stack.subsonic2 is not None:
        assert stack.subsonic2.song_count() == stack.expected_files
    for key in library.tracks:
        assert library.pid(key)


def test_workers_running_and_listening(stack):
    for queue, worker in stack.workers.items():
        assert worker.running(), queue
        assert worker.listening(stack.dsn), queue
    assert stack.listener.running()


def test_env_json_masks_secrets(stack):
    with open(os.path.join(stack.run_dir, 'env.json'), encoding='utf-8') as handle:
        payload = json.load(handle)
    assert payload['flask']['NAVIDROME_PASSWORD'] == '***'
    assert payload['worker']['AUDIOMUSE_ROLE'] == 'worker'
    assert 'AUDIOMUSE_ROLE' not in payload['flask']


def test_schema_was_created_by_init_db(stack, db):
    assert scalar(db, "SELECT to_regclass('public.score')") == 'score'
    assert scalar(db, "SELECT to_regclass('public.task_status')") == 'task_status'
