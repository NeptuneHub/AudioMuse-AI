# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""What happens when a worker's claim connection dies underneath a running job.

The advisory lock is this task's only liveness, and it is released the instant
the connection drops - even though the process is still running the job. Nothing
tells the worker: it uses that connection at claim time and again at finalize,
and in between an analysis can run for hours. Maintenance sees a RUNNING row
whose lock is free, requeues it, and a second worker starts the same task.

Main Features:
* The listener's idle tick stops a worker whose row is no longer its own
* A row still stamped with this worker is left running
* The terminal write refuses a row that was reclaimed and handed to someone else
* A dead connection at finalize reconnects rather than killing the worker
"""

from unittest.mock import MagicMock

import pytest

import config
from taskqueue import worker as worker_mod


class _Cursor:
    def __init__(self, rows):
        self._rows = rows
        self._row = None

    def execute(self, sql, params=None):
        self._row = self._rows.get(params[0] if params else None)

    def fetchone(self):
        return self._row

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _worker(monkeypatch):
    import threading

    instance = worker_mod.Worker.__new__(worker_mod.Worker)
    instance.identity = 'audiomuse-worker-default-hostA-11'
    instance.queue = 'default'
    instance._held_task_id = None
    instance._held_parent_id = None
    instance._conn = None
    instance._jobs_done = 0
    instance._claim_txn = threading.Lock()
    return instance


@pytest.fixture
def stopped(monkeypatch):
    stop = MagicMock()
    monkeypatch.setattr(worker_mod, 'stop_hard', stop)
    return stop
class TestAReclaimNoticeIsAddressedToOneGeneration:

    def _notice(self, task_id, worker_id, attempts):
        from taskqueue import sql

        return f"{task_id}{sql.RECLAIM_SEPARATOR}{worker_id}{sql.RECLAIM_SEPARATOR}{attempts}"

    def test_an_idle_worker_ignores_every_notice(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)

        instance.on_reclaimed(self._notice('task-1', instance.identity, 1))

        stopped.assert_not_called()

    def test_the_generation_that_lost_the_task_stops(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        instance._held_task_id = 'task-1'
        instance._held_attempts = 2

        instance.on_reclaimed(self._notice('task-1', instance.identity, 2))

        stopped.assert_called_once()

    def test_a_notice_for_a_different_task_is_ignored(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        instance._held_task_id = 'task-1'
        instance._held_attempts = 2

        instance.on_reclaimed(self._notice('task-2', instance.identity, 2))

        stopped.assert_not_called()

    def test_a_notice_for_a_different_worker_is_ignored(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        instance._held_task_id = 'task-1'
        instance._held_attempts = 2

        instance.on_reclaimed(self._notice('task-1', 'some-other-worker', 2))

        stopped.assert_not_called()

    def test_a_notice_for_an_earlier_attempt_cannot_stop_the_current_run(
        self, monkeypatch, stopped
    ):
        instance = _worker(monkeypatch)
        instance._held_task_id = 'task-1'
        instance._held_attempts = 3

        instance.on_reclaimed(self._notice('task-1', instance.identity, 2))

        stopped.assert_not_called()

    def test_a_malformed_notice_is_ignored(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        instance._held_task_id = 'task-1'
        instance._held_attempts = 1

        instance.on_reclaimed('not-a-notice')

        stopped.assert_not_called()


class TestReconnectingCoversTheNoticeItCouldNotHear:

    def test_a_reconnect_with_no_held_task_reads_nothing(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        conn = MagicMock()

        instance.on_listener_ready(conn)

        conn.cursor.assert_not_called()
        stopped.assert_not_called()

    def test_a_row_still_stamped_with_this_worker_keeps_running(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        instance._held_task_id = 'task-1'
        conn = MagicMock()
        conn.cursor.side_effect = lambda: _Cursor(
            {'task-1': (config.TASK_STATUS_RUNNING, 'main_analysis', None, instance.identity)}
        )

        instance.on_listener_ready(conn)

        stopped.assert_not_called()

    def test_a_row_handed_to_another_worker_stops_this_one(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        instance._held_task_id = 'task-1'
        conn = MagicMock()
        conn.cursor.side_effect = lambda: _Cursor(
            {
                'task-1': (
                    config.TASK_STATUS_RUNNING, 'main_analysis', None,
                    'audiomuse-worker-default-hostB-22',
                )
            }
        )

        instance.on_listener_ready(conn)

        stopped.assert_called_once()

    def test_a_row_requeued_back_to_new_stops_this_one(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        instance._held_task_id = 'task-1'
        conn = MagicMock()
        conn.cursor.side_effect = lambda: _Cursor(
            {'task-1': (config.TASK_STATUS_NEW, 'main_analysis', None, instance.identity)}
        )

        instance.on_listener_ready(conn)

        stopped.assert_called_once()

    def test_a_deleted_row_stops_this_one(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        instance._held_task_id = 'task-1'
        conn = MagicMock()
        conn.cursor.side_effect = lambda: _Cursor({})

        instance.on_listener_ready(conn)

        stopped.assert_called_once()


class TestFinalizeCannotKillTheWorker:
    def test_a_rollback_that_itself_raises_is_swallowed(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        instance._conn = MagicMock()
        instance._conn.closed = 0
        instance._conn.cursor.side_effect = RuntimeError('connection already closed')
        instance._conn.rollback.side_effect = RuntimeError('connection already closed')

        instance.finalize(
            {'task_id': 'task-1'}, config.TASK_STATUS_SUCCESS, None
        )

    def test_a_dead_connection_reconnects_rather_than_giving_up(
        self, monkeypatch, stopped
    ):
        instance = _worker(monkeypatch)
        dead = MagicMock()
        dead.closed = 1
        instance._conn = dead
        live = MagicMock()
        live.closed = 0
        live.cursor.side_effect = lambda: _Cursor({})

        def _connect():
            instance._conn = live
            return live

        instance.connect = _connect

        instance.finalize({'task_id': 'task-1'}, config.TASK_STATUS_SUCCESS, None)

        assert instance._conn is live


class TestTheTerminalWriteIsOwnershipChecked:
    def test_finish_task_binds_the_worker_identity(self):
        from taskqueue import sql

        cur = MagicMock()
        cur.fetchone.return_value = (config.TASK_STATUS_SUCCESS,)

        sql.finish_task(
            cur, 'task-1', config.TASK_STATUS_SUCCESS, {'message': 'done'}, 1.0,
            worker_id='audiomuse-worker-default-hostA-11',
        )

        statement, params = cur.execute.call_args.args
        assert 'worker_id' in statement
        assert params[-1] == 'audiomuse-worker-default-hostA-11'

    def test_a_reclaimed_row_is_not_overwritten(self, monkeypatch, stopped):
        instance = _worker(monkeypatch)
        instance._conn = MagicMock()
        instance._conn.closed = 0
        cur = MagicMock()
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: False
        cur.fetchone.return_value = (
            config.TASK_STATUS_RUNNING, 'main_analysis', None, 'other-worker'
        )
        instance._conn.cursor.return_value = cur
        monkeypatch.setattr(worker_mod.sql, 'finish_task', lambda *a, **k: None)

        instance.finalize({'task_id': 'task-1'}, config.TASK_STATUS_SUCCESS, None)

        instance._conn.commit.assert_called_once()
