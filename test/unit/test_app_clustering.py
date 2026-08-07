# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The clustering start endpoint and the parameters it queues.

The endpoint is one INSERT now: the queue row is the claim, admission is a
partial unique index, and there is no failure handler because the worker writes
the terminal row itself and knows whether a restart is coming.

Main Features:
* A successful start returns 202 with the queued task id, at status NEW
* output_server_scope is forced to 'all' whatever the client posts
* Auto-calibration defaults on and can be turned off from the body
* top_n_clustering_playlist accepts the legacy payload spellings
* A live main task the gate can already see answers 409 and queues nothing
* A start that passed the gate and lost the INSERT answers 409, never 500, and
  names the task that actually won
"""

import pytest
from flask import Flask

import config
import app_clustering
import taskqueue
from app_clustering import clustering_bp


@pytest.fixture
def queued(monkeypatch):
    calls = []

    def _fake_enqueue(func, **kwargs):
        calls.append({'func': func, **kwargs})
        return kwargs['task_id']

    monkeypatch.setattr(app_clustering.taskqueue, 'enqueue', _fake_enqueue)
    monkeypatch.setattr(app_clustering, 'clean_up_previous_main_tasks', lambda: None)
    monkeypatch.setattr(app_clustering, 'get_active_main_task', lambda **_kw: None)
    return calls


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(clustering_bp)
    app.config['TESTING'] = True
    return app.test_client()


def _start(client, **body):
    return client.post('/api/clustering/start', json=body)


def _lose_the_admission_race(monkeypatch, winner):
    reads = []
    attempted = []

    def _active(**kwargs):
        reads.append(kwargs)
        return winner if len(reads) > 1 else None

    def _reject(func, **kwargs):
        attempted.append(func)
        raise taskqueue.TaskAlreadyRunning()

    monkeypatch.setattr(app_clustering, 'clean_up_previous_main_tasks', lambda: None)
    monkeypatch.setattr(app_clustering, 'get_active_main_task', _active)
    monkeypatch.setattr(app_clustering.taskqueue, 'enqueue', _reject)
    return attempted


class TestStartClustering:
    def test_a_successful_start_returns_the_queued_task_at_status_new(self, client, queued):
        response = _start(client)

        assert response.status_code == 202
        body = response.get_json()
        assert body['task_type'] == 'main_clustering'
        assert body['status'] == config.TASK_STATUS_NEW
        assert body['task_id'] == queued[0]['task_id']

    def test_the_queued_entry_names_the_clustering_task_on_the_high_queue(self, client, queued):
        _start(client)

        assert queued[0]['func'] == 'tasks.clustering.run_clustering_task'
        assert queued[0]['queue'] == taskqueue.QUEUE_HIGH

    def test_output_server_scope_is_forced_to_all_whatever_the_client_posts(
        self, client, queued
    ):
        _start(client, output_server_scope='srv-2')

        assert queued[0]['kwargs']['output_server_scope'] == 'all'

    def test_auto_calibration_defaults_on(self, client, queued):
        _start(client)

        assert queued[0]['kwargs']['auto_calibration_param'] is True

    def test_auto_calibration_can_be_turned_off_from_the_body(self, client, queued):
        _start(client, auto_parameter_discovery=False)

        assert queued[0]['kwargs']['auto_calibration_param'] is False

    def test_top_n_clustering_playlist_is_queued(self, client, queued):
        _start(client, top_n_clustering_playlist=10)

        assert queued[0]['kwargs']['top_n_playlists_param'] == 10

    def test_the_legacy_top_n_playlists_spelling_is_accepted(self, client, queued):
        _start(client, top_n_playlists=12)

        assert queued[0]['kwargs']['top_n_playlists_param'] == 12

    def test_a_live_main_task_the_gate_sees_answers_409_and_queues_nothing(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(app_clustering, 'clean_up_previous_main_tasks', lambda: None)
        monkeypatch.setattr(
            app_clustering, 'get_active_main_task',
            lambda **_kw: {'task_id': 'live-1', 'status': config.TASK_STATUS_RUNNING},
        )
        calls = []
        monkeypatch.setattr(
            app_clustering.taskqueue, 'enqueue', lambda func, **kw: calls.append(func)
        )

        response = _start(client)

        assert response.status_code == 409
        assert response.get_json()['task_id'] == 'live-1'
        assert calls == [], 'nothing may be queued once the gate has refused'

    def test_a_start_that_passed_the_gate_then_lost_the_insert_answers_409_not_500(
        self, client, monkeypatch
    ):
        attempted = _lose_the_admission_race(
            monkeypatch, {'task_id': 'winner-1', 'status': config.TASK_STATUS_RUNNING}
        )

        response = _start(client)

        assert attempted == ['tasks.clustering.run_clustering_task'], (
            'the gate has to let this start through, or the INSERT race is never run'
        )
        assert response.status_code == 409
        body = response.get_json()
        assert body['task_id'] == 'winner-1', (
            'the losing id names a row the savepoint rolled back, so the answer has '
            'to carry the task that actually holds the gate'
        )
        assert body['status'] == config.TASK_STATUS_RUNNING
        assert 'already running' in body['error'].lower()

    def test_a_queue_failure_answers_500_without_leaking_the_exception(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(app_clustering, 'clean_up_previous_main_tasks', lambda: None)
        monkeypatch.setattr(app_clustering, 'get_active_main_task', lambda **_kw: None)

        def _boom(func, **kwargs):
            raise RuntimeError('connection refused to 10.0.0.5')

        monkeypatch.setattr(app_clustering.taskqueue, 'enqueue', _boom)

        response = _start(client)

        assert response.status_code == 500
        assert '10.0.0.5' not in response.get_json()['error']


class TestNoFailureHandlerRemains:
    def test_the_rq_failure_handler_is_gone(self):
        assert not hasattr(app_clustering, 'clustering_task_failure_handler')
