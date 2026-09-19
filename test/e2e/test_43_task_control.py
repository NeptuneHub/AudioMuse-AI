# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Task control: one live main task at a time, and a cancel that stops everything.

A long clustering is started, a second main task is refused while it runs, and
POST /api/cancel/<id> revokes it: the row ends REVOKED, no child row survives
(the cancel is global by design), the queue reports idle, and the workers come
back so a normal clustering succeeds afterwards.

Main Features:
* a second start while a main task is live answers 409 with the live task id
* cancel answers the cancelled job count, the row ends REVOKED at 100 percent
* cancel_all with nothing live is 404, cancelling twice is harmless
* the queue recovers: a short clustering succeeds after the cancel
"""

import time

import pytest

from test.e2e.e2e_helpers import assert_no_fp_ids, scalar

pytestmark = pytest.mark.e2e

LONG_CLUSTERING = {
    'clustering_method': 'kmeans',
    'enable_clustering_embeddings': True,
    'clustering_runs': 2000,
    'num_clusters_min': 2,
    'num_clusters_max': 4,
    'pca_components_min': 0,
    'pca_components_max': 8,
    'auto_parameter_discovery': False,
    'top_n_clustering_playlist': 3,
    'max_songs_per_cluster': 0,
    'stratified_sampling_target_percentile': 100,
    'ai_model_provider': 'NONE',
}


def _wait_running(api, task_id, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = api.status(task_id)
        if status and status['state'] == 'RUNNING':
            return status
        time.sleep(1)
    raise AssertionError(f'task {task_id} never reached RUNNING')


def _wait_workers_idle(api, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summary = api.json('GET', '/api/dashboard/summary')
        workers = summary.get('workers') or []
        backlog = summary.get('queue_backlog') or []
        if workers and all(w.get('state') == 'idle' for w in workers) and all(
            q['running_count'] == 0 and q['pending_count'] == 0 for q in backlog
        ):
            return summary
        time.sleep(2)
    raise AssertionError('workers did not return to idle')


def test_cancel_a_running_clustering(stack, api, db, analyzed_library):
    api.wait_idle(120)
    task_id = api.start_task('/api/clustering/start', LONG_CLUSTERING)
    _wait_running(api, task_id)
    refused = api.post('/api/analysis/start', json={'num_recent_albums': 0})
    assert refused.status_code == 409, refused.text
    assert refused.json().get('task_id') == task_id, refused.text
    active = api.active_task()
    assert active.get('task_id') == task_id, active
    assert active.get('side_job') is False, active

    cancelled = api.json('POST', f'/api/cancel/{task_id}')
    assert_no_fp_ids(cancelled)
    assert cancelled['task_id'] == task_id
    assert cancelled.get('cancelled_jobs_count', 0) >= 1, cancelled

    final = api.wait_for_task(task_id, timeout=120, expect='REVOKED')
    assert final['progress'] == 100
    assert 'cancel' in final['status_message'].lower(), final
    api.wait_idle(120)
    last = api.last_task()
    assert last['task_id'] == task_id, last
    assert last['status'] == 'REVOKED', last
    assert scalar(db, 'SELECT count(*) FROM task_status WHERE parent_task_id = %s', (task_id,)) == 0
    _wait_workers_idle(api)
    for worker in stack.workers.values():
        worker.wait_ready(180)

    again = api.json('POST', f'/api/cancel/{task_id}')
    assert again['task_id'] == task_id
    assert api.post('/api/cancel_all/main_clustering').status_code == 404


def test_queue_recovers_after_cancel(stack, api, analyzed_library):
    api.wait_idle(120)
    short = dict(LONG_CLUSTERING, clustering_runs=10)
    task_id = api.start_task('/api/clustering/start', short)
    final = api.wait_for_task(task_id, timeout=600)
    assert final['state'] == 'SUCCESS'
    api.wait_idle(60)
