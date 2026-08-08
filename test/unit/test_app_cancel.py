# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Global cancel: one transaction, one notification, no second system.

Cancel is one transaction against one system: queued rows stop being claimable
the moment they stop being NEW, and a NOTIFY riding on the same commit reaches
whichever worker holds the running one. There is no second store to scan, no
retry budget to zero and no queue to empty separately.

Main Features:
* The tombstone and the stop signal are one transaction, so neither can happen alone
* A failed transaction raises rather than reporting a false success
* Exactly one REVOKED recap row survives, for the task the caller named
* The global cancel epoch is incremented in that same transaction
* An un-acknowledged provider-migration handshake is spared, as is a live control request
* History keeps each row's real terminal status instead of restamping it
"""

import json
from unittest.mock import MagicMock

import pytest

import app_helper
import config
import database
import taskqueue


class _Cursor:
    def __init__(self, recorder):
        self._recorder = recorder
        self.rowcount = 3
        self._rows = []

    def execute(self, sql, params=None):
        self._recorder.append((' '.join(sql.split()), params))
        text = sql.upper()
        if 'FROM MIGRATION_SESSION' in text:
            self._rows = list(self._recorder.migration_rows)
        elif 'WHERE TASK_TYPE = %S AND STATUS IN' in text:
            self._rows = [(task_id,) for task_id in self._recorder.control_rows]
        elif 'APP_CONFIG' in text:
            self._rows = [('7',)]
        elif text.strip().startswith('SELECT'):
            self._rows = list(self._recorder.snapshot_rows)
        else:
            self._rows = []

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Recorder(list):
    def __init__(self):
        super().__init__()
        self.snapshot_rows = []
        self.migration_rows = []
        self.control_rows = []


def _row(task_id, task_type='main_analysis', status=config.TASK_STATUS_RUNNING):
    return {
        'task_id': task_id,
        'task_type': task_type,
        'status': status,
        'details': json.dumps({'message': 'x'}),
        'start_time': 100.0,
        'end_time': None,
    }


@pytest.fixture
def cancel_env(monkeypatch):
    recorder = _Recorder()
    db = MagicMock()
    db.cursor.side_effect = lambda *a, **k: _Cursor(recorder)
    monkeypatch.setattr(app_helper, 'get_db', lambda: db)
    history = []
    monkeypatch.setattr(
        app_helper.database, 'record_task_history',
        lambda task_id, task_type, status, **kw: history.append((task_id, task_type, status, kw)),
    )
    signalled = []
    monkeypatch.setattr(
        taskqueue, 'request_cancel_all', lambda conn=None: signalled.append(conn)
    )
    return recorder, db, history, signalled


def _statements(recorder):
    return [sql for sql, _params in recorder]


class TestTheTombstoneAndTheSignalAreOneTransaction:
    def test_the_stop_signal_is_published_before_the_commit(self, cancel_env):
        recorder, db, _history, signalled = cancel_env
        recorder.snapshot_rows = [_row('task-1')]

        app_helper.cancel_job_and_children_recursive('task-1')

        assert signalled == [db], 'the cancel must be signalled on the same connection'
        db.commit.assert_called_once()

    def test_a_failed_transaction_rolls_back_and_raises(self, cancel_env, monkeypatch):
        recorder, db, _history, _signalled = cancel_env
        recorder.snapshot_rows = [_row('task-1')]
        db.commit.side_effect = RuntimeError('disk full')

        with pytest.raises(RuntimeError):
            app_helper.cancel_job_and_children_recursive('task-1')

        db.rollback.assert_called_once()

    def test_every_task_row_is_deleted(self, cancel_env):
        recorder, _db, _history, _signalled = cancel_env
        recorder.snapshot_rows = [_row('task-1'), _row('task-2', 'main_clustering')]

        app_helper.cancel_job_and_children_recursive('task-1')

        assert any(
            sql == 'DELETE FROM task_status' for sql in _statements(recorder)
        ), 'the wipe is what stops queued work: a non-NEW row can never be claimed'

    def test_exactly_one_revoked_recap_row_survives(self, cancel_env):
        recorder, _db, _history, _signalled = cancel_env
        recorder.snapshot_rows = [_row('task-1')]

        app_helper.cancel_job_and_children_recursive('task-1')

        inserts = [
            params for sql, params in recorder if sql.startswith('INSERT INTO task_status')
        ]
        assert len(inserts) == 1
        assert inserts[0][0] == 'task-1'
        assert inserts[0][2] == config.TASK_STATUS_REVOKED


class TestTheCancelEpochIsActuallyBumped:
    def test_the_epoch_advances_inside_the_cancel_transaction(self, cancel_env):
        recorder, db, _history, _signalled = cancel_env
        recorder.snapshot_rows = [_row('task-1')]

        app_helper.cancel_job_and_children_recursive('task-1')

        bumps = [
            sql for sql, params in recorder
            if params and database.GLOBAL_CANCEL_EPOCH_KEY in params
        ]
        assert len(bumps) == 1, (
            'cancel must advance the epoch the migration endpoints read, and the key '
            'travels as a bind parameter, so only the parameters identify the statement'
        )
        assert bumps[0].startswith('INSERT INTO app_config'), (
            'reading the epoch back is not bumping it'
        )
        assert '+ 1' in bumps[0], 'the epoch has to advance, not be restamped'
        assert db.commit.call_count == 1


class TestProtectedRows:
    def test_an_unacknowledged_migration_handshake_is_spared(self, cancel_env):
        recorder, _db, _history, _signalled = cancel_env
        recorder.snapshot_rows = [_row('exec-1', 'provider_migration')]
        recorder.migration_rows = [('exec-1', 'align-1')]

        app_helper.cancel_job_and_children_recursive('other-1')

        deletes = [
            params for sql, params in recorder if sql.startswith('DELETE FROM task_status')
        ]
        assert deletes, 'a protected cancel still deletes everything unprotected'
        protected = set(deletes[0][0])
        assert {'exec-1', 'align-1'} <= protected

    def test_a_live_control_request_is_spared(self, cancel_env):
        recorder, _db, _history, _signalled = cancel_env
        recorder.snapshot_rows = [_row('task-1')]
        recorder.control_rows = ['control-abc']

        app_helper.cancel_job_and_children_recursive('task-1')

        deletes = [
            params for sql, params in recorder if sql.startswith('DELETE FROM task_status')
        ]
        assert 'control-abc' in set(deletes[0][0])

    def test_a_protected_task_id_gets_no_recap_row(self, cancel_env):
        recorder, _db, _history, _signalled = cancel_env
        recorder.snapshot_rows = [_row('exec-1', 'provider_migration')]
        recorder.migration_rows = [('exec-1', None)]

        app_helper.cancel_job_and_children_recursive('exec-1')

        inserts = [
            sql for sql in _statements(recorder) if sql.startswith('INSERT INTO task_status')
        ]
        assert not inserts, 'a spared row must not be overwritten by a REVOKED recap'


class TestCancelHistory:
    def test_an_already_finished_task_keeps_its_own_terminal_status(self, cancel_env):
        recorder, _db, history, _signalled = cancel_env
        recorder.snapshot_rows = [_row('done-1', status=config.TASK_STATUS_SUCCESS)]

        app_helper.cancel_job_and_children_recursive('done-1')

        assert history[0][2] == config.TASK_STATUS_SUCCESS, (
            'pressing Cancel must not rewrite a completed run as cancelled'
        )

    def test_a_running_task_is_recorded_as_revoked_with_the_reason(self, cancel_env):
        recorder, _db, history, _signalled = cancel_env
        recorder.snapshot_rows = [_row('live-1')]

        app_helper.cancel_job_and_children_recursive('live-1', reason='user pressed cancel')

        assert history[0][2] == config.TASK_STATUS_REVOKED
        assert history[0][3]['note'] == 'user pressed cancel'

    def test_history_failure_never_undoes_the_cancel(self, cancel_env, monkeypatch):
        recorder, db, _history, _signalled = cancel_env
        recorder.snapshot_rows = [_row('live-1')]
        monkeypatch.setattr(
            app_helper.database, 'record_task_history',
            MagicMock(side_effect=RuntimeError('history table is gone')),
        )

        app_helper.cancel_job_and_children_recursive('live-1')

        db.commit.assert_called_once()
