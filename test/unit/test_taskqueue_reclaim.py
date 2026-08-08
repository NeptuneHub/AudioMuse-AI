# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Restarting a task whose worker died, and refusing to restart anything else.

A task is orphaned when its row says RUNNING and nobody holds its advisory lock,
which Postgres answers exactly because the lock died with the worker's
connection. That is the only reason a task ever restarts, and it may do so
``QUEUE_MAX_ATTEMPTS`` times.

Main Features:
* A task a live worker still holds is left completely alone
* Losing the hold probe costs no pg_advisory_unlock and no server warning
* A failed reclaim rolls back before releasing, so one bad row does not abandon the pass
* Requeued tasks wake both queues exactly once for the whole pass
"""

from unittest.mock import MagicMock

import pytest

import config
from taskqueue import maintenance


class _Cursor:
    def __init__(self, recorder):
        self._recorder = recorder

    def execute(self, sql, params=None):
        self._recorder.statements.append(' '.join(sql.split()))

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Recorder:
    def __init__(self):
        self.statements = []


@pytest.fixture
def conn():
    recorder = _Recorder()
    connection = MagicMock()
    connection.cursor.side_effect = lambda *a, **k: _Cursor(recorder)
    connection.statements = recorder.statements
    return connection


@pytest.fixture
def running(monkeypatch):
    rows = []
    monkeypatch.setattr(
        maintenance.sql, 'running_tasks', lambda _cur, grace_seconds=None: rows
    )
    return rows


class TestALiveTaskIsNeverTouched:
    def test_a_task_whose_lock_is_held_is_not_requeued(self, conn, running, monkeypatch):
        running.append({'task_id': 'live-1', 'attempts': 1, 'max_attempts': 3})
        monkeypatch.setattr(maintenance.sql, 'try_hold', lambda _cur, _tid: False)
        requeued = []
        monkeypatch.setattr(
            maintenance.sql, 'requeue_or_fail',
            lambda *a, **k: requeued.append(a) or config.TASK_STATUS_NEW,
        )

        assert maintenance.reclaim_orphans(conn) == []
        assert requeued == []

    def test_losing_the_hold_probe_releases_nothing(self, conn, running, monkeypatch):
        running.append({'task_id': 'live-1', 'attempts': 1, 'max_attempts': 3})
        monkeypatch.setattr(maintenance.sql, 'try_hold', lambda _cur, _tid: False)
        released = []
        monkeypatch.setattr(
            maintenance.sql, 'release', lambda _cur, tid: released.append(tid)
        )

        maintenance.reclaim_orphans(conn)

        assert released == []


class TestAnOrphanIsRestartedThenGivenUpOn:
    def test_an_unheld_task_is_requeued_and_the_probe_released(
        self, conn, running, monkeypatch
    ):
        running.append({'task_id': 'dead-1', 'attempts': 1, 'max_attempts': 3})
        monkeypatch.setattr(maintenance.sql, 'try_hold', lambda _cur, _tid: True)
        monkeypatch.setattr(
            maintenance.sql, 'requeue_or_fail',
            lambda *a, **k: config.TASK_STATUS_NEW,
        )
        released = []
        monkeypatch.setattr(
            maintenance.sql, 'release', lambda _cur, tid: released.append(tid)
        )
        notified = []
        monkeypatch.setattr(
            maintenance.sql, 'notify_job', lambda _cur, queue: notified.append(queue)
        )

        assert maintenance.reclaim_orphans(conn) == [('dead-1', config.TASK_STATUS_NEW)]
        assert released == ['dead-1']
        assert notified == [maintenance.sql.QUEUE_HIGH, maintenance.sql.QUEUE_DEFAULT]

    def test_the_last_attempt_fails_the_task_for_good(self, conn, running, monkeypatch):
        running.append({'task_id': 'dead-1', 'attempts': 3, 'max_attempts': 3})
        monkeypatch.setattr(maintenance.sql, 'try_hold', lambda _cur, _tid: True)
        monkeypatch.setattr(
            maintenance.sql, 'requeue_or_fail',
            lambda *a, **k: config.TASK_STATUS_FAIL,
        )
        monkeypatch.setattr(maintenance.sql, 'release', lambda *a: None)
        monkeypatch.setattr(maintenance.sql, 'notify_job', lambda *a: None)

        assert maintenance.reclaim_orphans(conn) == [('dead-1', config.TASK_STATUS_FAIL)]

    def test_one_failing_row_does_not_abandon_the_rest_of_the_pass(
        self, conn, running, monkeypatch
    ):
        running.extend([
            {'task_id': 'boom-1', 'attempts': 1, 'max_attempts': 3},
            {'task_id': 'dead-2', 'attempts': 1, 'max_attempts': 3},
        ])
        monkeypatch.setattr(maintenance.sql, 'try_hold', lambda _cur, _tid: True)

        def _requeue(_cur, task_id, *_a, **_k):
            if task_id == 'boom-1':
                raise RuntimeError('deadlock detected')
            return config.TASK_STATUS_NEW

        monkeypatch.setattr(maintenance.sql, 'requeue_or_fail', _requeue)
        monkeypatch.setattr(maintenance.sql, 'release', lambda *a: None)
        monkeypatch.setattr(maintenance.sql, 'notify_job', lambda *a: None)

        assert maintenance.reclaim_orphans(conn) == [('dead-2', config.TASK_STATUS_NEW)]

    def test_a_failing_row_rolls_back_before_it_releases(
        self, conn, running, monkeypatch
    ):
        running.append({'task_id': 'boom-1', 'attempts': 1, 'max_attempts': 3})
        order = []
        monkeypatch.setattr(maintenance.sql, 'try_hold', lambda _cur, _tid: True)
        monkeypatch.setattr(
            maintenance.sql, 'requeue_or_fail',
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')),
        )
        monkeypatch.setattr(
            maintenance.sql, 'release', lambda *_a: order.append('release')
        )
        conn.rollback.side_effect = lambda: order.append('rollback')

        maintenance.reclaim_orphans(conn)

        assert order == ['rollback', 'release']


class TestNothingIsWokenWhenNothingMoved:
    def test_a_pass_that_reclaims_nothing_sends_no_notification(
        self, conn, running, monkeypatch
    ):
        monkeypatch.setattr(maintenance.sql, 'try_hold', lambda _cur, _tid: True)
        notified = []
        monkeypatch.setattr(
            maintenance.sql, 'notify_job', lambda _cur, queue: notified.append(queue)
        )

        assert maintenance.reclaim_orphans(conn) == []
        assert notified == []
