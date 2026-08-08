# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Asking the workers to restart, exactly once, and only in this database.

Two protections here are invisible in review and easy to drop by accident: a
listener must refuse to run an action twice for the same request id, and
supervisorctl's output must be read tolerantly enough that a service already in
the requested state counts as success.

Main Features:
* A request this listener already answered re-records its ack, it does not re-run
* A first delivery does run the action, and records whatever it returned
* An unknown action never reaches the supervisor and never writes an ack
* A failed ack lookup executes rather than silently skipping the request
* Listener counting and the worker registry only see this database's sessions
* supervisorctl reporting "already started" / "not running" is not a failure
"""

import json
from unittest.mock import MagicMock

import pytest

import restart_manager
from taskqueue import control
from taskqueue import sql


class _Recorder:
    def __init__(self):
        self.statements = []
        self.params = []


class _Cursor:
    def __init__(self, recorder, rows):
        self._recorder = recorder
        self._rows = rows
        self._row = None

    def execute(self, statement, params=None):
        self._recorder.statements.append(' '.join(statement.split()))
        self._recorder.params.append(params)
        self._row = self._rows.pop(0) if self._rows else None

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _conn(recorder, rows=None):
    connection = MagicMock()
    connection.cursor.side_effect = lambda *a, **k: _Cursor(recorder, list(rows or []))
    return connection


class _Listener:
    def __init__(self, monkeypatch, rows=None, connect_error=None):
        self.listener = control.ControlListener()
        self.recorder = _Recorder()
        self.executed = []
        self.acks = []

        def connect():
            if connect_error is not None:
                raise connect_error
            return _conn(self.recorder, rows)

        def execute(action):
            self.executed.append(action)
            return True

        monkeypatch.setattr(self.listener, 'connect', connect)
        monkeypatch.setattr(self.listener, '_execute', execute)
        monkeypatch.setattr(
            self.listener,
            '_record_ack',
            lambda conn, request_id, action, ok: self.acks.append((request_id, action, ok)),
        )

    def deliver(self, action, request_id):
        self.listener.on_notify(
            sql.CHANNEL_CONTROL, json.dumps({'action': action, 'request_id': request_id})
        )


class TestARequestThisListenerAlreadyAnsweredIsNeverRunTwice:
    def test_the_recorded_acknowledgement_is_re_recorded_instead_of_restarting_again(
        self, monkeypatch
    ):
        harness = _Listener(monkeypatch, rows=[('SUCCESS',)])

        harness.deliver(control.ACTION_RESTART, 'control-already-done')

        assert harness.executed == [], (
            'the publisher re-publishes a request whose row is still RUNNING, so a '
            'listener that already acked must not bounce the whole fleet a second time'
        )
        assert harness.acks == [('control-already-done', control.ACTION_RESTART, True)]

    def test_a_previously_failed_acknowledgement_reruns_the_action(self, monkeypatch):
        harness = _Listener(monkeypatch, rows=[('FAIL',)])

        harness.deliver(control.ACTION_RESTART, 'control-failed-before')

        assert harness.executed == [control.ACTION_RESTART], (
            're-recording a stored FAIL as handled flipped it to SUCCESS via the ack '
            'upsert, and the migration handshake then believed a restart that never '
            'happened - a redelivery after failure must run the action again'
        )
        assert harness.acks == [('control-failed-before', control.ACTION_RESTART, True)]

    def test_a_first_delivery_runs_the_action_and_records_the_outcome(self, monkeypatch):
        harness = _Listener(monkeypatch, rows=[None])

        harness.deliver(control.ACTION_RESTART, 'control-fresh')

        assert harness.executed == [control.ACTION_RESTART]
        assert harness.acks == [('control-fresh', control.ACTION_RESTART, True)]

    def test_the_ack_row_is_looked_up_per_listener_not_per_request(self, monkeypatch):
        harness = _Listener(monkeypatch, rows=[None])

        harness.deliver(control.ACTION_STOP, 'control-scope')

        assert harness.recorder.params[0] == (f"control-scope:{control.listener_id()}",), (
            'two containers answer the same request id, so the dedup key must carry '
            'the listener identity or one listener would suppress the other'
        )


class TestAnUnknownActionNeverReachesTheSupervisor:
    @pytest.mark.parametrize('payload', [
        {'action': 'reboot', 'request_id': 'control-1'},
        {'action': control.ACTION_RESTART},
        {'request_id': 'control-2'},
    ])
    def test_it_is_refused_without_executing_or_acknowledging(self, monkeypatch, payload):
        harness = _Listener(monkeypatch, rows=[None])

        harness.listener.on_notify(sql.CHANNEL_CONTROL, json.dumps(payload))

        assert harness.executed == []
        assert harness.acks == []

    def test_an_unreadable_payload_is_ignored_rather_than_raising(self, monkeypatch):
        harness = _Listener(monkeypatch, rows=[None])

        harness.listener.on_notify(sql.CHANNEL_CONTROL, 'not json')

        assert harness.executed == []
        assert harness.acks == []


class TestAFailedAckLookupExecutesRatherThanSkipping:
    def test_the_action_still_runs_when_the_lookup_connection_dies(self, monkeypatch):
        harness = _Listener(monkeypatch, connect_error=RuntimeError('no database'))

        harness.deliver(control.ACTION_RESTART, 'control-probe-down')

        assert harness.executed == [control.ACTION_RESTART], (
            'a dedup probe that cannot answer must fall back to doing the work; '
            'skipping would drop a restart nobody ever performs'
        )
        assert harness.acks == [('control-probe-down', control.ACTION_RESTART, True)]


class TestOneClusterCanHostTwoDeploymentsWithoutCrossTalk:
    def test_listener_counting_only_sees_this_database(self):
        assert 'current_database()' in control._COUNT_LISTENERS, (
            'pg_stat_activity is cluster-wide but NOTIFY is database-local, so a '
            'foreign listener would inflate the expected ack count forever'
        )

    def test_the_worker_registry_only_sees_this_database(self):
        assert 'current_database()' in sql._WORKER_SNAPSHOT


class TestASupervisorActionThatWasAlreadySatisfied:
    @pytest.fixture(autouse=True)
    def _no_ipc(self, monkeypatch):
        monkeypatch.setattr(restart_manager, '_use_control_ipc', lambda: False)

    def _result(self, monkeypatch, returncode, stdout='', stderr=''):
        completed = MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)
        monkeypatch.setattr(restart_manager.subprocess, 'run', lambda *a, **k: completed)

    def test_starting_something_already_started_counts_as_success(self, monkeypatch):
        self._result(
            monkeypatch, 1, stdout='queue-worker-high: ERROR (already started)'
        )

        ok, _detail = restart_manager.run_supervisorctl_detail(['start', 'queue-worker-high'])

        assert ok is True

    def test_stopping_something_not_running_counts_as_success(self, monkeypatch):
        self._result(monkeypatch, 1, stdout='queue-worker-high: ERROR (not running)')

        ok, _detail = restart_manager.run_supervisorctl_detail(['stop', 'queue-worker-high'])

        assert ok is True

    def test_a_genuine_failure_is_still_a_failure(self, monkeypatch):
        self._result(monkeypatch, 1, stderr='queue-worker-high: ERROR (spawn error)')

        ok, detail = restart_manager.run_supervisorctl_detail(['start', 'queue-worker-high'])

        assert ok is False
        assert 'spawn error' in detail

    def test_a_restart_is_never_treated_as_already_satisfied(self, monkeypatch):
        self._result(monkeypatch, 1, stdout='queue-worker-high: started')

        ok, _detail = restart_manager.run_supervisorctl_detail(['restart', 'queue-worker-high'])

        assert ok is False, (
            'only start and stop have an idempotent reading; a restart that failed '
            'has not restarted anything'
        )

    def test_the_detail_carries_the_output_the_restore_log_needs(self, monkeypatch):
        self._result(monkeypatch, 0, stdout='queue-worker-high: stopped')

        ok, detail = restart_manager.run_supervisorctl_detail(['stop', 'queue-worker-high'])

        assert ok is True
        assert 'stopped' in detail


class TestTheControlResultRefusesAMismatchedAction:
    def test_a_request_recorded_for_another_action_answers_false(self, monkeypatch):
        monkeypatch.setattr(restart_manager, '_action_matches', lambda *_a: False)
        monkeypatch.setattr(
            'taskqueue.control.get_control_request_result',
            lambda *_a, **_k: True,
        )

        assert restart_manager.get_control_request_result('restart', 'control-9') is False

    def test_a_matching_action_returns_the_recorded_outcome(self, monkeypatch):
        monkeypatch.setattr(restart_manager, '_action_matches', lambda *_a: True)
        monkeypatch.setattr(
            'taskqueue.control.get_control_request_result',
            lambda *_a, **_k: True,
        )

        assert restart_manager.get_control_request_result('restart', 'control-9') is True


class TestADeliberateRestartNeverChargesAWorkerLossAttempt:
    def _harness(self, monkeypatch, execute_ok=True):
        harness = _Listener(monkeypatch)
        monkeypatch.setattr(harness.listener, '_execute', lambda action: execute_ok)
        harness.requeued = []
        monkeypatch.setattr(
            harness.listener, '_requeue_tasks_of_stopped_workers',
            lambda conn: harness.requeued.append(conn),
        )
        return harness

    def test_a_successful_restart_requeues_its_workers_tasks_itself(self, monkeypatch):
        harness = self._harness(monkeypatch)

        harness.deliver(control.ACTION_RESTART, 'req-1')

        assert len(harness.requeued) == 1, (
            'a wizard save is a deliberate restart, not a crash: leaving the tasks '
            'to the charged reclaim meant three saves during a long run failed it'
        )

    def test_a_stop_requeues_too_so_start_resumes_at_zero_cost(self, monkeypatch):
        harness = self._harness(monkeypatch)

        harness.deliver(control.ACTION_STOP, 'req-2')

        assert len(harness.requeued) == 1

    def test_a_failed_action_leaves_the_tasks_to_the_charged_reclaim(self, monkeypatch):
        harness = self._harness(monkeypatch, execute_ok=False)

        harness.deliver(control.ACTION_RESTART, 'req-3')

        assert harness.requeued == [], (
            'the workers may still be running; only the advisory-lock reclaim may '
            'decide their fate'
        )

    def test_a_plugin_sync_does_not_touch_the_queue(self, monkeypatch):
        harness = self._harness(monkeypatch)

        harness.deliver(control.ACTION_PLUGIN_SYNC, 'req-4')

        assert harness.requeued == []

    def test_the_requeue_itself_never_increments_attempts(self):
        statement = ' '.join(sql._REQUEUE_UNCHARGED.split())

        assert "SET status='NEW'" in statement
        assert 'attempts' not in statement, (
            'attempts counts real worker losses; a control-plane restart is not one'
        )
