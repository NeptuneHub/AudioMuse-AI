# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Behaviour of a freshly configured instance before any track was analyzed.

This is the only module that must run before the shared analysis: it asserts
the empty-library answers of every read endpoint, that the wizard is already
satisfied by the environment, that the model coverage bands are all empty, and
that starting a clustering on an empty catalogue reports the error on its first
attempt and can be cancelled. It deliberately never requests analyzed_library.

Main Features:
* setup_saved is true and every coverage band is 0 on the empty catalogue
* every search, similarity, map and stats endpoint answers empty or 404
* the dashboard sees two idle workers and no content
* clustering an empty library reports its error and a cancel ends the retries
"""

import time

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, scalar

pytestmark = pytest.mark.e2e

EMPTY_TABLES = ('score', 'embedding', 'track_server_map', 'playlist', 'clap_embedding', 'lyrics_embedding')


def test_catalogue_is_empty(stack, db):
    assert scalar(db, 'SELECT count(*) FROM score') == 0, 'this module must run before analyzed_library'
    for table in EMPTY_TABLES:
        assert scalar(db, f'SELECT count(*) FROM {table}') == 0, table


def test_health_and_setup_state(api):
    assert api.health()
    setup = api.json('GET', '/api/setup')
    assert setup['setup_saved'] is True
    coverage = setup['model_coverage']
    assert coverage, setup
    assert all(band == 0 for band in coverage.values()), coverage


def test_config_reflects_the_harness_knobs(api):
    config = api.json('GET', '/api/config')
    assert_no_fp_ids(config)
    assert str(config.get('ai_model_provider', '')).upper() == 'NONE'
    assert int(config['num_clusters_min']) == 2
    assert int(config['num_clusters_max']) == 4
    assert int(config['clustering_runs']) == 10


def test_dashboard_sees_idle_workers(api):
    summary = api.json('GET', '/api/dashboard/summary')
    assert_no_fp_ids(summary)
    workers = summary['workers']
    assert len(workers) == 2, workers
    queues = sorted(q for w in workers for q in w.get('queues', []))
    assert queues == ['default', 'high'], workers
    assert all(w.get('state') == 'idle' for w in workers), workers
    backlog = summary['queue_backlog']
    assert sorted(q['queue_name'] for q in backlog) == ['default', 'high'], backlog
    for entry in backlog:
        assert entry['pending_count'] == 0 and entry['running_count'] == 0 and entry['delayed_count'] == 0, backlog


def test_read_endpoints_answer_empty(api):
    assert api.json('GET', '/api/search_tracks?search_query=a') == []
    assert api.get('/api/similar_tracks?item_id=nope').status_code == 503
    map_body = api.json('GET', '/api/map')
    assert map_body['items'] == []
    for path in ('/api/clap/stats', '/api/lyrics/stats', '/api/sem_grove/stats'):
        stats = api.json('GET', path)
        assert not stats.get('loaded'), (path, stats)
        assert int(stats.get('song_count', 0)) == 0, (path, stats)
    assert api.post('/api/clap/search', json={'query': 'piano'}).status_code == 503
    assert api.json('GET', '/api/cron') == []
    playlists = api.json('GET', '/api/playlists')
    assert all(not server.get('playlists') for server in playlists.get('servers', [])), playlists
    sync = api.json('GET', '/api/sync?fields=index')
    assert sync['total_tracks'] == 0
    assert sync['provider_type'] == 'navidrome'


def test_no_task_history_yet(api):
    last = api.last_task()
    assert last['status'] == 'NO_PREVIOUS_MAIN_TASK', last
    assert api.active_task() == {}


def _reports_failure(status):
    if status is None:
        return False
    if status['state'] == 'FAIL':
        return True
    details = status.get('details') or {}
    return bool(details.get('error')) or 'clusterable' in str(status.get('status_message', '')).lower()


def test_clustering_on_empty_library_reports_the_error_and_cancels(api):
    api.wait_idle(60)
    task_id = api.start_task(
        '/api/clustering/start',
        {
            'clustering_method': 'kmeans',
            'enable_clustering_embeddings': True,
            'clustering_runs': 10,
            'num_clusters_min': 2,
            'num_clusters_max': 4,
            'auto_parameter_discovery': False,
            'ai_model_provider': 'NONE',
        },
    )
    deadline = time.monotonic() + 120
    status = api.status(task_id)
    while not _reports_failure(status):
        assert time.monotonic() < deadline, f'no failure reported within 120s: {status}'
        time.sleep(1)
        status = api.status(task_id)
    assert status['task_type_from_db'] == 'main_clustering'
    cancelled = api.json('POST', f'/api/cancel/{task_id}')
    assert cancelled['task_id'] == task_id
    final = api.wait_for_task(task_id, timeout=120, expect=None)
    assert final['state'] in ('REVOKED', 'FAIL'), final
    api.wait_idle(120)
    assert api.active_task() == {}
