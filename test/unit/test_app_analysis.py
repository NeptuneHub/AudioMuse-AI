# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The analysis and cleaning start endpoints.

Both endpoints are now one INSERT: the queue row IS the claim, so there is no
separate save-then-enqueue pair to keep consistent and no ambiguous enqueue
outcome to resolve. Admission is a partial unique index, so a second live main
task surfaces as TaskAlreadyRunning rather than as a check-then-act race.

Main Features:
* A successful start returns 202 with the task id it queued, at status NEW
* Request parameters and config defaults reach the queued kwargs unchanged
* A live main task the gate can already see answers 409 and queues nothing
* A start that passed the gate and lost the INSERT answers 409, never 500, and
  names the task that actually won, even when that task has since finished
* Cleaning also refuses while a sweep runs, which analysis deliberately does not
* A queue failure answers 500 rather than leaving a half-claimed row
"""

import pytest
from unittest.mock import patch
from flask import Flask

import config
import app_analysis
import taskqueue
from app_analysis import analysis_bp


@pytest.fixture
def queued(monkeypatch):
    calls = []

    def _fake_enqueue(func, **kwargs):
        calls.append({'func': func, **kwargs})
        return kwargs['task_id']

    monkeypatch.setattr(app_analysis.taskqueue, 'enqueue', _fake_enqueue)
    monkeypatch.setattr(app_analysis, 'clean_up_previous_main_tasks', lambda: None)
    monkeypatch.setattr(app_analysis, 'get_active_main_task', lambda **_kw: None)
    return calls


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(analysis_bp)
    app.config['TESTING'] = True
    return app.test_client()


def _lose_the_admission_race(monkeypatch, winner):
    reads = []
    attempted = []

    def _active(**kwargs):
        reads.append(kwargs)
        return winner if len(reads) > 1 else None

    def _reject(func, **kwargs):
        attempted.append(func)
        raise taskqueue.TaskAlreadyRunning()

    monkeypatch.setattr(app_analysis, 'clean_up_previous_main_tasks', lambda: None)
    monkeypatch.setattr(app_analysis, 'get_active_main_task', _active)
    monkeypatch.setattr(app_analysis.taskqueue, 'enqueue', _reject)
    return attempted


class TestStartAnalysis:
    def test_a_successful_start_returns_the_queued_task_at_status_new(self, client, queued):
        response = client.post('/api/analysis/start', json={})

        assert response.status_code == 202
        body = response.get_json()
        assert body['task_type'] == 'main_analysis'
        assert body['status'] == config.TASK_STATUS_NEW
        assert body['task_id'] == queued[0]['task_id']

    def test_the_queued_entry_names_the_analysis_task_on_the_high_queue(self, client, queued):
        client.post('/api/analysis/start', json={})

        assert queued[0]['func'] == 'tasks.analysis.run_analysis_task'
        assert queued[0]['queue'] == taskqueue.QUEUE_HIGH
        assert queued[0]['task_type'] == 'main_analysis'

    def test_posted_parameters_reach_the_queued_args(self, client, queued):
        client.post('/api/analysis/start', json={'num_recent_albums': 7, 'top_n_moods': 9})

        assert queued[0]['args'] == (7, 9)

    def test_config_defaults_are_used_when_the_body_omits_them(self, client, queued):
        client.post('/api/analysis/start', json={})

        assert queued[0]['args'] == (config.NUM_RECENT_ALBUMS, config.TOP_N_MOODS)

    def test_a_missing_json_body_still_starts_with_defaults(self, client, queued):
        response = client.post('/api/analysis/start')

        assert response.status_code == 202
        assert queued[0]['args'] == (config.NUM_RECENT_ALBUMS, config.TOP_N_MOODS)

    def test_previous_main_tasks_are_archived_before_the_claim(self, client, monkeypatch):
        order = []
        monkeypatch.setattr(
            app_analysis, 'clean_up_previous_main_tasks', lambda: order.append('cleanup')
        )
        monkeypatch.setattr(app_analysis, 'get_active_main_task', lambda **_kw: None)
        monkeypatch.setattr(
            app_analysis.taskqueue, 'enqueue',
            lambda func, **kwargs: order.append('enqueue') or kwargs['task_id'],
        )

        client.post('/api/analysis/start', json={})

        assert order == ['cleanup', 'enqueue']

    def test_a_live_main_task_the_gate_sees_answers_409_and_queues_nothing(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(app_analysis, 'clean_up_previous_main_tasks', lambda: None)
        monkeypatch.setattr(
            app_analysis, 'get_active_main_task',
            lambda **_kw: {'task_id': 'live-1', 'status': config.TASK_STATUS_RUNNING},
        )
        calls = []
        monkeypatch.setattr(
            app_analysis.taskqueue, 'enqueue', lambda func, **kw: calls.append(func)
        )

        response = client.post('/api/analysis/start', json={})

        assert response.status_code == 409
        body = response.get_json()
        assert body['task_id'] == 'live-1'
        assert body['status'] == config.TASK_STATUS_RUNNING
        assert 'already in progress' in body['error'].lower()
        assert calls == [], 'nothing may be queued once the gate has refused'

    def test_a_start_that_passed_the_gate_then_lost_the_insert_answers_409_not_500(
        self, client, monkeypatch
    ):
        attempted = _lose_the_admission_race(
            monkeypatch, {'task_id': 'winner-1', 'status': config.TASK_STATUS_RUNNING}
        )

        response = client.post('/api/analysis/start', json={})

        assert attempted == ['tasks.analysis.run_analysis_task'], (
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

    def test_losing_the_insert_to_a_task_that_has_since_finished_still_answers_409(
        self, client, monkeypatch
    ):
        attempted = _lose_the_admission_race(monkeypatch, None)

        response = client.post('/api/analysis/start', json={})

        assert attempted, 'the gate has to let this start through'
        assert response.status_code == 409
        assert response.get_json()['task_id'] is None

    def test_a_queue_failure_answers_500(self, client, monkeypatch):
        monkeypatch.setattr(app_analysis, 'clean_up_previous_main_tasks', lambda: None)
        monkeypatch.setattr(app_analysis, 'get_active_main_task', lambda **_kw: None)

        def _boom(func, **kwargs):
            raise RuntimeError('database is gone')

        monkeypatch.setattr(app_analysis.taskqueue, 'enqueue', _boom)

        response = client.post('/api/analysis/start', json={})

        assert response.status_code == 500
        assert 'database is gone' not in response.get_json()['error']


class TestStartCleaning:
    def test_a_successful_start_returns_the_queued_task_at_status_new(self, client, queued):
        response = client.post('/api/cleaning/start', json={})

        assert response.status_code == 202
        body = response.get_json()
        assert body['task_type'] == 'cleaning'
        assert body['status'] == config.TASK_STATUS_NEW

    def test_the_queued_entry_names_the_cleaning_task_on_the_high_queue(self, client, queued):
        client.post('/api/cleaning/start', json={})

        assert queued[0]['func'] == 'tasks.cleaning.identify_and_clean_orphaned_albums_task'
        assert queued[0]['queue'] == taskqueue.QUEUE_HIGH

    def test_the_clean_catalogue_flag_from_the_body_reaches_the_queued_args(
        self, client, queued
    ):
        client.post('/api/cleaning/start', json={'clean_catalogue': True})

        assert queued[0]['args'] == (True,)

    def test_cleaning_refuses_while_a_sweep_runs_which_analysis_does_not(
        self, client, monkeypatch
    ):
        seen = {}

        def _active(**kwargs):
            seen.update(kwargs)
            return {'task_id': 'sweep-1', 'status': config.TASK_STATUS_RUNNING}

        monkeypatch.setattr(app_analysis, 'get_active_main_task', _active)
        monkeypatch.setattr(app_analysis, 'clean_up_previous_main_tasks', lambda: None)

        response = client.post('/api/cleaning/start', json={})

        assert response.status_code == 409
        excluded = seen.get('exclude_task_types')
        assert 'server_sweep' not in excluded, (
            'a sweep writes the same mappings cleaning rewrites, so it has to keep '
            'blocking the start'
        )
        assert 'worker_control' in excluded, (
            'a restart handshake is machinery, not catalogue work; excluding '
            'NOTHING made an in-flight handshake answer 409 to a cleaning the user '
            'had just asked for'
        )
        assert response.get_json()['task_id'] == 'sweep-1'

    def test_a_cleaning_that_passed_the_gate_then_lost_the_insert_answers_409_not_500(
        self, client, monkeypatch
    ):
        attempted = _lose_the_admission_race(
            monkeypatch, {'task_id': 'winner-2', 'status': config.TASK_STATUS_RUNNING}
        )

        response = client.post('/api/cleaning/start', json={})

        assert attempted == ['tasks.cleaning.identify_and_clean_orphaned_albums_task'], (
            'the gate has to let this start through, or the INSERT race is never run'
        )
        assert response.status_code == 409
        body = response.get_json()
        assert body['task_id'] == 'winner-2'
        assert body['status'] == config.TASK_STATUS_RUNNING
        assert 'already running' in body['error'].lower()


class TestCleaningPage:
    def test_the_page_renders_with_the_catalogue_default(self, client):
        with patch('app_analysis.render_template', return_value='<html></html>') as render:
            response = client.get('/cleaning')

        assert response.status_code == 200
        render.assert_called_once_with(
            'cleaning.html',
            title='AudioMuse-AI - Database Cleaning',
            active='cleaning',
            cleaning_catalogue_default=config.CLEANING_CATALOGUE,
        )


class TestBlueprintWiring:
    def test_the_blueprint_is_registered(self, client):
        assert 'analysis_bp' in str(analysis_bp)

    @pytest.mark.parametrize('path', ['/api/analysis/start', '/api/cleaning/start'])
    def test_the_start_endpoints_accept_post_only(self, client, path):
        assert client.get(path).status_code == 405

    def test_the_cleaning_page_accepts_get_only(self, client):
        assert client.post('/cleaning').status_code == 405
