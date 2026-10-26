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
* The unanalyzable list is emptied only after the typed phrase, and the
  dashboard count drops to 0 in the same transaction
"""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

import config
import database
import app_analysis
import taskqueue
from app_analysis import analysis_bp
from app_helper import CLEAR_EXCLUSIONS_PHRASE


@pytest.fixture
def queued(monkeypatch):
    calls = []

    def _fake_enqueue(func, **kwargs):
        calls.append({'func': func, **kwargs})
        return kwargs['task_id']

    monkeypatch.setattr(app_analysis.taskqueue, 'enqueue', _fake_enqueue)
    monkeypatch.setattr(database, 'clean_up_previous_main_tasks', lambda: None)
    monkeypatch.setattr(database, 'get_queue_blocking_task', lambda **_kw: None)
    monkeypatch.setattr(database, 'get_active_main_task', lambda **_kw: None)
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
        if kwargs.get('task_type') == 'server_sweep':
            return None
        reads.append(kwargs)
        return winner

    def _reject(func, **kwargs):
        attempted.append(func)
        raise taskqueue.TaskAlreadyRunning()

    monkeypatch.setattr(database, 'clean_up_previous_main_tasks', lambda: None)
    monkeypatch.setattr(database, 'get_queue_blocking_task', lambda **_kw: None)
    monkeypatch.setattr(database, 'get_active_main_task', _active)
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

    @pytest.mark.parametrize('body', [
        {'num_recent_albums': 'not-a-number'},
        {'top_n_moods': 'lots'},
        {'num_recent_albums': None},
        {'num_recent_albums': []},
        {'top_n_moods': {'a': 1}},
    ])
    def test_a_count_that_is_not_a_number_is_refused_instead_of_crashing(
        self, client, queued, body
    ):
        response = client.post('/api/analysis/start', json=body)

        assert response.status_code == 400, (
            'an unguarded int() on the posted counts answered 500 with an '
            '"unexpected error", which is exactly what the error contract says '
            'a bad request must never get'
        )
        assert 'whole numbers' in response.get_json()['error']
        assert not queued, 'a refused start must never reach the queue'

    def test_previous_main_tasks_are_archived_before_the_claim(self, client, monkeypatch):
        order = []
        monkeypatch.setattr(
            database, 'clean_up_previous_main_tasks', lambda: order.append('cleanup')
        )
        monkeypatch.setattr(database, 'get_queue_blocking_task', lambda **_kw: None)
        monkeypatch.setattr(database, 'get_active_main_task', lambda **_kw: None)
        monkeypatch.setattr(
            app_analysis.taskqueue, 'enqueue',
            lambda func, **kwargs: order.append('enqueue') or kwargs['task_id'],
        )

        client.post('/api/analysis/start', json={})

        assert order == ['cleanup', 'enqueue']

    def test_a_live_main_task_the_gate_sees_answers_409_and_queues_nothing(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(database, 'clean_up_previous_main_tasks', lambda: None)
        monkeypatch.setattr(
            database, 'get_queue_blocking_task',
            lambda **_kw: {
                'task_id': 'live-1', 'task_type': 'main_analysis',
                'status': config.TASK_STATUS_RUNNING,
            },
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
        assert 'still running' in body['error'].lower()
        assert body['error_code'] == 1201
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
        monkeypatch.setattr(database, 'clean_up_previous_main_tasks', lambda: None)
        monkeypatch.setattr(database, 'get_queue_blocking_task', lambda **_kw: None)
        monkeypatch.setattr(database, 'get_active_main_task', lambda **_kw: None)

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
        monkeypatch.setattr(database, 'get_queue_blocking_task', lambda **_kw: None)
        monkeypatch.setattr(
            database, 'get_active_main_task',
            lambda **_kw: {
                'task_id': 'sweep-1', 'task_type': 'server_sweep',
                'status': config.TASK_STATUS_RUNNING,
            },
        )
        monkeypatch.setattr(database, 'clean_up_previous_main_tasks', lambda: None)

        response = client.post('/api/cleaning/start', json={})

        assert response.status_code == 409
        assert response.get_json()['task_id'] == 'sweep-1', (
            'a sweep writes the same mappings cleaning rewrites, so it has to keep '
            'blocking the start'
        )

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


class TestClearUnanalyzableList:
    PATH = '/api/analysis/exclusions/clear'

    @pytest.fixture
    def cleared(self, monkeypatch):
        calls = []

        def _clear():
            calls.append(True)
            return 7

        monkeypatch.setattr(database, 'clear_analysis_exclusions', _clear)
        return calls

    def test_the_exact_phrase_empties_the_list_and_reports_the_count(self, client, cleared):
        response = client.post(self.PATH, json={'confirmation': CLEAR_EXCLUSIONS_PHRASE})

        assert response.status_code == 200
        assert response.get_json() == {'success': True, 'deleted': 7}
        assert cleared == [True]

    def test_spaces_a_phone_keyboard_adds_around_the_phrase_are_ignored(self, client, cleared):
        response = client.post(
            self.PATH, json={'confirmation': ' ' + CLEAR_EXCLUSIONS_PHRASE + ' '}
        )

        assert response.status_code == 200
        assert cleared == [True]

    @pytest.mark.parametrize('body', [
        None,
        {},
        {'confirmation': ''},
        {'confirmation': 5},
        {'confirmation': CLEAR_EXCLUSIONS_PHRASE.lower()},
        {'confirmation': CLEAR_EXCLUSIONS_PHRASE[:-1]},
        CLEAR_EXCLUSIONS_PHRASE,
        [CLEAR_EXCLUSIONS_PHRASE],
    ])
    def test_anything_but_the_phrase_is_refused_with_400_and_deletes_nothing(
        self, client, cleared, body
    ):
        response = client.post(self.PATH, json=body) if body is not None else client.post(self.PATH)

        assert response.status_code == 400
        assert cleared == []

    def test_a_database_error_answers_503_without_its_detail(self, client, monkeypatch):
        def _boom():
            raise RuntimeError('relation is gone')

        monkeypatch.setattr(database, 'clear_analysis_exclusions', _boom)

        response = client.post(self.PATH, json={'confirmation': CLEAR_EXCLUSIONS_PHRASE})

        assert response.status_code == 503
        assert 'relation is gone' not in response.get_data(as_text=True)

    @staticmethod
    def _db(monkeypatch, cur):
        db = MagicMock()
        db.cursor.return_value.__enter__.return_value = cur
        monkeypatch.setattr(database, 'get_db', lambda: db)
        return db

    def test_the_delete_covers_every_server_and_zeroes_the_dashboard_count(self, monkeypatch):
        cur = MagicMock(rowcount=3)
        db = self._db(monkeypatch, cur)

        assert database.clear_analysis_exclusions() == 3
        statements = [c.args[0] for c in cur.execute.call_args_list]
        assert statements[0] == "DELETE FROM analysis_exclusions"
        assert 'dashboard_stats' in statements[1] and 'unanalyzable_tracks' in statements[1], (
            'the browse total and the dashboard card read the snapshot, so it has to '
            'drop to 0 with the rows instead of a minute later'
        )
        db.commit.assert_called_once_with()
        db.rollback.assert_not_called()

    def test_a_failed_delete_rolls_back_and_raises(self, monkeypatch):
        cur = MagicMock()
        cur.execute.side_effect = RuntimeError('lock timeout')
        db = self._db(monkeypatch, cur)

        with pytest.raises(RuntimeError, match='lock timeout'):
            database.clear_analysis_exclusions()
        db.rollback.assert_called_once_with()
        db.commit.assert_not_called()


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
    def test_the_endpoint_names_the_templates_call_url_for_on_map_to_their_paths(self, client):
        assert 'analysis_bp' in client.application.blueprints
        rules = {
            rule.endpoint: rule.rule for rule in client.application.url_map.iter_rules()
        }
        assert rules['analysis_bp.cleaning_page'] == '/cleaning'
        assert rules['analysis_bp.start_analysis_endpoint'] == '/api/analysis/start'
        assert rules['analysis_bp.start_cleaning_endpoint'] == '/api/cleaning/start'
        assert rules['analysis_bp.clear_analysis_exclusions_endpoint'] == (
            '/api/analysis/exclusions/clear'
        )

    @pytest.mark.parametrize(
        'path', ['/api/analysis/start', '/api/cleaning/start', '/api/analysis/exclusions/clear']
    )
    def test_the_start_endpoints_accept_post_only(self, client, path):
        assert client.get(path).status_code == 405

    def test_the_cleaning_page_accepts_get_only(self, client):
        assert client.post('/cleaning').status_code == 405
