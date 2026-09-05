# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The worker that walks away from a RUNNING row is the one that puts it back.

A task that loses its database connection re-raises on purpose and writes no
terminal row, so the worker keeps the row RUNNING rather than failing a job over
a two-second Postgres blip. Nobody else can recover that row: every reclaim path
reads liveness from ``pg_stat_activity`` and this worker process is still alive
under the same identity, so its own row is excluded from reclaim for as long as
the container lives.

The free retry is narrow and it is bounded, and both halves are pinned here.
``psycopg2.OperationalError`` is NOT a synonym for "the connection died": its
hierarchy is flat, so the same base class covers ``QueryCanceled`` (which is what
the app's own ``statement_timeout=600000`` produces), ``DeadlockDetected``,
``SerializationFailure``, ``DiskFull`` and ``OutOfMemory``. Handing those the
uncharged requeue meant a ten-minute statement timeout, or a deadlock that
reproduces in milliseconds, requeued its row for ever with no attempt ever
charged and no sleep anywhere on the path. Even a genuinely lost connection now
gets only ``UNCHARGED_REQUEUE_LIMIT`` free passes per row before the requeue
starts costing a worker-loss attempt, and a repeat pass waits first.

Main Features:
* Only a lost connection remembers the row: the database refusing the work fails it
* SQLSTATE class 08, the 57Pxx shutdown codes and 53300 are the lost-connection codes
* A libpq error with no SQLSTATE at all, InterfaceError and ConnectionLostError count too
* The abandoned row is requeued with no worker-loss attempt charged, bound to this worker
* After UNCHARGED_REQUEUE_LIMIT free passes the same row's requeue charges an attempt
* A charged requeue is refused unless the row is still this worker's RUNNING row
* A repeat pass sleeps first, doubling and capped, so the loop cannot spin hot
* run_forever calls requeue_abandoned at the top of EVERY loop, which is asserted
  by driving the loop rather than by trusting the method to be reachable
* A row that is no longer this worker's RUNNING row is dropped, never resurrected
* A requeue that raises is retried on the next loop instead of being forgotten
* Both queues are woken once per pass that actually put something back
* The shared payload is read under the same lock the listener swaps the connection under
* RUNNING is the ONLY status either statement's guard names: the status literals
  are collected out of the WHERE clause and compared as a whole set, so widening
  the guard to admit a terminal row is caught rather than merely un-asserted
"""

import re
import threading
from unittest.mock import MagicMock

import psycopg2
import pytest

import config
import database
from taskqueue import sql
from taskqueue import worker as worker_mod


def _normalise(statement):
    return ' '.join(statement.split())


def _where_clause(statement):
    normalised = _normalise(statement)
    assert ' WHERE ' in normalised, 'the statement has no WHERE clause at all'
    return normalised.split(' WHERE ', 1)[1]


def _status_literals(fragment):
    found = set(re.findall(r"status\s*=\s*'([A-Z_]+)'", fragment))
    for group in re.findall(r"status\s+IN\s*\(([^)]*)\)", fragment):
        found.update(re.findall(r"'([A-Z_]+)'", group))
    return found


def _worker():
    instance = worker_mod.Worker.__new__(worker_mod.Worker)
    instance.identity = 'audiomuse-worker-default-hostA-11'
    instance.queue = 'default'
    instance.max_jobs = 0
    instance._held_task_id = None
    instance._held_parent_id = None
    instance._held_attempts = None
    instance._conn = MagicMock()
    instance._conn.closed = 0
    instance._jobs_done = 0
    instance._shared_cache = {}
    instance._abandoned = []
    instance._uncharged = {}
    instance._wake = threading.Event()
    instance._claim_txn = threading.Lock()
    instance._fork_jobs = False
    return instance


def _coded(sqlstate):
    coded = type(
        'CodedOperationalError', (psycopg2.OperationalError,), {'pgcode': sqlstate}
    )
    return coded('the server answered with a SQLSTATE')


def _running_row(worker_id):
    return {
        'status': config.TASK_STATUS_RUNNING,
        'task_type': 'main_analysis',
        'parent_task_id': None,
        'worker_id': worker_id,
    }


class _StopTheLoop(BaseException):
    pass


def _job(task_id):
    return {
        'task_id': task_id,
        'task_type': 'main_analysis',
        'parent_task_id': None,
        'func': 'tasks.analysis.run_analysis_task',
        'args': (),
        'kwargs': {},
        'attempts': 0,
        'max_attempts': 3,
    }


def _raising(exc):
    def _call(*_args, **_kwargs):
        raise exc

    return _call


@pytest.fixture
def ran(monkeypatch):
    import taskqueue

    holder = {}

    def _resolve(_dotted):
        return holder['func']

    monkeypatch.setattr(taskqueue, 'resolve_func', _resolve)
    return holder


class TestOnlyALostConnectionHandsTheRowBack:
    def test_a_connectivity_error_remembers_the_row_instead_of_failing_it(
        self, monkeypatch, ran
    ):
        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        finalized = []
        monkeypatch.setattr(instance, 'finalize', lambda *a, **k: finalized.append(a))
        ran['func'] = _raising(psycopg2.OperationalError('server closed the connection'))

        instance.run_job(_job('task-1'))

        assert instance._abandoned == ['task-1']
        assert finalized == []

    @staticmethod
    def _charged_requeue(monkeypatch, instance):
        charged = []
        monkeypatch.setattr(
            worker_mod.sql, 'current_row',
            lambda _cur, _task_id: _running_row(instance.identity),
        )
        monkeypatch.setattr(
            worker_mod.sql, 'requeue_or_fail',
            lambda _cur, task_id, _now, _details, delay_seconds=None: (
                charged.append((task_id, delay_seconds)) or config.TASK_STATUS_NEW
            ),
        )
        return charged

    def test_a_task_that_failed_on_its_own_merits_is_retried_with_an_attempt_charged(
        self, monkeypatch, ran
    ):
        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        finalized = []
        monkeypatch.setattr(
            instance, 'finalize', lambda job, status, *a, **k: finalized.append(status)
        )
        charged = self._charged_requeue(monkeypatch, instance)
        ran['func'] = _raising(ValueError('the album has no tracks'))

        instance.run_job(_job('task-1'))

        assert instance._abandoned == [], (
            'it is not a lost connection, so it gets none of the free passes'
        )
        assert finalized == [], (
            'attempt 1 of 3 is not the last one, so the queue requeues instead of '
            'writing the terminal row'
        )
        assert [task_id for task_id, _delay in charged] == ['task-1']
        assert charged[0][1] > 0, 'a retry waits; a deterministic failure must not spin'

    def test_the_last_allowed_attempt_fails_the_row_for_good(self, monkeypatch, ran):
        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        finalized = []
        monkeypatch.setattr(
            instance, 'finalize', lambda job, status, *a, **k: finalized.append(status)
        )
        charged = self._charged_requeue(monkeypatch, instance)
        ran['func'] = _raising(ValueError('still no tracks'))
        job = _job('task-1')
        job['attempts'] = job['max_attempts']

        instance.run_job(job)

        assert charged == []
        assert finalized == [config.TASK_STATUS_FAIL]

    def test_a_permanent_failure_is_never_retried(self, monkeypatch, ran):
        from taskqueue import TaskFailed

        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        finalized = []
        monkeypatch.setattr(
            instance, 'finalize', lambda job, status, *a, **k: finalized.append(status)
        )
        charged = self._charged_requeue(monkeypatch, instance)
        ran['func'] = _raising(TaskFailed('this server type is not supported'))

        instance.run_job(_job('task-1'))

        assert charged == [], (
            'TaskFailed is the one thing a task may say that means "no retry can fix '
            'this"; retrying it would only spend the budget a real failure needs'
        )
        assert finalized == [config.TASK_STATUS_FAIL]

    def test_a_cooperative_cancel_revokes_the_row_and_charges_nothing(
        self, monkeypatch, ran
    ):
        from taskqueue import TaskCancelled

        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        finalized = []
        monkeypatch.setattr(
            instance, 'finalize', lambda job, status, *a, **k: finalized.append(status)
        )
        charged = self._charged_requeue(monkeypatch, instance)
        ran['func'] = _raising(TaskCancelled('task-1 was revoked'))

        instance.run_job(_job('task-1'))

        assert charged == []
        assert finalized == [config.TASK_STATUS_REVOKED]

    def test_a_zero_attempt_budget_fails_on_the_first_error(self, monkeypatch, ran):
        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        finalized = []
        monkeypatch.setattr(
            instance, 'finalize', lambda job, status, *a, **k: finalized.append(status)
        )
        charged = self._charged_requeue(monkeypatch, instance)
        ran['func'] = _raising(ValueError('planner blew up'))
        job = _job('task-1')
        job['max_attempts'] = 0

        instance.run_job(job)

        assert charged == [], (
            'the migration planner is enqueued with max_attempts=0 on purpose: a dry '
            'run that silently re-ran would hold the session claimed an extra attempt'
        )
        assert finalized == [config.TASK_STATUS_FAIL]

    def test_the_retry_is_requeued_only_after_the_hold_is_dropped(
        self, monkeypatch, ran
    ):
        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        monkeypatch.setattr(instance, 'finalize', lambda *a, **k: None)
        held_at_requeue = []
        monkeypatch.setattr(
            worker_mod.sql, 'current_row',
            lambda _cur, _task_id: _running_row(instance.identity),
        )

        def requeue(_cur, task_id, _now, _details, delay_seconds=None):
            held_at_requeue.append((instance._held_task_id, instance._held_attempts))
            return config.TASK_STATUS_NEW

        monkeypatch.setattr(worker_mod.sql, 'requeue_or_fail', requeue)
        ran['func'] = _raising(ValueError('boom'))
        instance._held_task_id = 'task-1'
        instance._held_attempts = 0

        instance.run_job(_job('task-1'))

        assert held_at_requeue == [(None, None)], (
            'requeue_or_fail publishes the same reclaim notice a maintenance reclaim '
            'does, and on_reclaimed ends a worker whose held task and attempt match '
            'it; requeueing while still holding would make the worker end itself '
            'on every retry'
        )

    def test_a_task_that_succeeded_is_never_remembered(self, monkeypatch, ran):
        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        monkeypatch.setattr(instance, 'finalize', lambda *a, **k: None)
        ran['func'] = lambda *a, **k: {'done': True}

        instance.run_job(_job('task-1'))

        assert instance._abandoned == []

    def test_the_same_row_is_remembered_once_however_often_it_is_abandoned(
        self, monkeypatch, ran
    ):
        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        monkeypatch.setattr(instance, 'finalize', lambda *a, **k: None)
        ran['func'] = _raising(psycopg2.InterfaceError('connection already closed'))

        instance.run_job(_job('task-1'))
        instance.run_job(_job('task-1'))

        assert instance._abandoned == ['task-1']

    def test_a_statement_timeout_is_charged_and_bounded_never_free(
        self, monkeypatch, ran
    ):
        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        monkeypatch.setattr(instance, 'finalize', lambda *a, **k: None)
        charged = self._charged_requeue(monkeypatch, instance)
        ran['func'] = _raising(
            psycopg2.errors.QueryCanceled(
                'canceling statement due to statement timeout'
            )
        )

        instance.run_job(_job('task-1'))

        assert instance._abandoned == [], (
            'psycopg2 puts QueryCanceled under OperationalError, and treating it as a '
            'lost connection once requeued it forever with no attempt charged and no '
            'sleep. It is not a lost connection: it takes the charged, bounded, '
            'backed-off path like any other failure'
        )
        assert [task_id for task_id, _delay in charged] == ['task-1']

    def test_a_deadlock_is_charged_and_bounded_never_free(self, monkeypatch, ran):
        instance = _worker()
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        monkeypatch.setattr(instance, 'finalize', lambda *a, **k: None)
        charged = self._charged_requeue(monkeypatch, instance)
        ran['func'] = _raising(psycopg2.errors.DeadlockDetected('deadlock detected'))

        instance.run_job(_job('task-1'))

        assert instance._abandoned == []
        assert [task_id for task_id, _delay in charged] == ['task-1']

    def test_a_terminal_outcome_gives_the_row_its_free_retries_back(
        self, monkeypatch, ran
    ):
        instance = _worker()
        instance._uncharged = {'task-1': worker_mod.UNCHARGED_REQUEUE_LIMIT}
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        monkeypatch.setattr(instance, 'finalize', lambda *a, **k: None)
        ran['func'] = lambda *a, **k: {'done': True}

        instance.run_job(_job('task-1'))

        assert instance._uncharged == {}

    def test_a_lost_connection_leaves_the_free_retry_count_for_the_requeue_to_read(
        self, monkeypatch, ran
    ):
        instance = _worker()
        instance._uncharged = {'task-1': 1}
        monkeypatch.setattr(instance, 'hydrate_config', lambda: None)
        monkeypatch.setattr(instance, 'finalize', lambda *a, **k: None)
        ran['func'] = _raising(psycopg2.InterfaceError('connection already closed'))

        instance.run_job(_job('task-1'))

        assert instance._uncharged == {'task-1': 1}


_WORK_REFUSED_ERRORS = (
    'QueryCanceled',
    'DeadlockDetected',
    'SerializationFailure',
    'DiskFull',
    'OutOfMemory',
    'LockNotAvailable',
    'IoError',
)

_LOST_CONNECTION_ERRORS = (
    'ConnectionException',
    'ConnectionDoesNotExist',
    'ConnectionFailure',
    'SqlclientUnableToEstablishSqlconnection',
    'SqlserverRejectedEstablishmentOfSqlconnection',
    'TransactionResolutionUnknown',
    'ProtocolViolation',
    'AdminShutdown',
    'CrashShutdown',
    'CannotConnectNow',
    'DatabaseDropped',
    'IdleSessionTimeout',
    'TooManyConnections',
)

_LOST_CONNECTION_SQLSTATES = (
    '08000', '08001', '08003', '08004', '08006', '08007', '08P01',
    '53300', '57P01', '57P02', '57P03', '57P04', '57P05',
)

_WORK_REFUSED_SQLSTATES = (
    '57014', '40001', '40P01', '53100', '53200', '55P03', '58030', '25006',
)


class TestOperationalErrorIsNotASynonymForALostConnection:
    @pytest.mark.parametrize('name', _WORK_REFUSED_ERRORS)
    def test_the_database_refusing_the_work_is_not_a_lost_connection(self, name):
        exc = getattr(psycopg2.errors, name)('the database refused this work')

        assert isinstance(exc, psycopg2.OperationalError)
        assert worker_mod._is_connectivity_error(exc) is False

    @pytest.mark.parametrize('name', _LOST_CONNECTION_ERRORS)
    def test_a_dropped_or_unreachable_connection_is_a_lost_connection(self, name):
        exc = getattr(psycopg2.errors, name)('the connection is gone')

        assert worker_mod._is_connectivity_error(exc) is True

    @pytest.mark.parametrize('sqlstate', _LOST_CONNECTION_SQLSTATES)
    def test_a_server_error_in_the_lost_connection_sqlstates_counts(self, sqlstate):
        assert worker_mod._is_connectivity_error(_coded(sqlstate)) is True

    @pytest.mark.parametrize('sqlstate', _WORK_REFUSED_SQLSTATES)
    def test_a_server_error_outside_those_sqlstates_does_not_count(self, sqlstate):
        assert worker_mod._is_connectivity_error(_coded(sqlstate)) is False

    def test_a_libpq_failure_with_no_sqlstate_at_all_is_a_lost_connection(self):
        exc = psycopg2.OperationalError('server closed the connection unexpectedly')

        assert exc.pgcode is None
        assert worker_mod._is_connectivity_error(exc) is True

    def test_a_broken_connection_object_is_a_lost_connection(self):
        assert worker_mod._is_connectivity_error(
            psycopg2.InterfaceError('connection already closed')
        ) is True

    def test_the_apps_own_connection_lost_error_is_a_lost_connection(self):
        assert worker_mod._is_connectivity_error(
            database.ConnectionLostError('database connection lost, retry the operation')
        ) is True

    def test_an_error_that_is_not_a_database_error_at_all_is_not(self):
        assert worker_mod._is_connectivity_error(ValueError('the album has no tracks')) is False


class TestEveryLoopStartsByPuttingTheAbandonedRowsBack:
    def _order_of(self, monkeypatch, instance, claims):
        order = []
        remaining = list(claims)

        def _claim():
            order.append('claim')
            if not remaining:
                raise _StopTheLoop()
            return remaining.pop(0)

        instance._wake = MagicMock()
        monkeypatch.setattr(
            instance, 'requeue_abandoned', lambda: order.append('requeue')
        )
        monkeypatch.setattr(instance, 'claim', _claim)
        monkeypatch.setattr(instance, 'run_job', lambda _job: order.append('run'))
        with pytest.raises(_StopTheLoop):
            instance.run_forever()
        return order

    def test_the_requeue_runs_before_the_very_first_claim(self, monkeypatch):
        assert self._order_of(monkeypatch, _worker(), []) == ['requeue', 'claim']

    def test_the_requeue_runs_again_on_a_loop_that_found_no_work(self, monkeypatch):
        assert self._order_of(monkeypatch, _worker(), [None, None]) == [
            'requeue', 'claim', 'requeue', 'claim', 'requeue', 'claim',
        ]

    def test_the_requeue_runs_again_on_the_loop_after_a_job_ran(self, monkeypatch):
        assert self._order_of(monkeypatch, _worker(), [_job('task-1')]) == [
            'requeue', 'claim', 'run', 'requeue', 'claim',
        ]


class TestTheAbandonedRowIsPutBackWithoutChargingAnAttempt:
    def test_an_abandoned_row_is_requeued_for_this_worker_only(self, monkeypatch):
        instance = _worker()
        instance._abandoned = ['task-1']
        calls = []
        monkeypatch.setattr(
            worker_mod.sql, 'requeue_uncharged',
            lambda _cur, task_id, worker_id=None: calls.append((task_id, worker_id)) or True,
        )
        monkeypatch.setattr(worker_mod.sql, 'notify_job', lambda *a: None)
        charged = []
        monkeypatch.setattr(
            worker_mod.sql, 'requeue_or_fail', lambda *a, **k: charged.append(a)
        )

        instance.requeue_abandoned()

        assert calls == [('task-1', instance.identity)]
        assert charged == []
        assert instance._abandoned == []

    def test_both_queues_are_woken_once_for_the_whole_pass(self, monkeypatch):
        instance = _worker()
        instance._abandoned = ['task-1', 'task-2']
        monkeypatch.setattr(
            worker_mod.sql, 'requeue_uncharged', lambda *a, **k: True
        )
        notified = []
        monkeypatch.setattr(
            worker_mod.sql, 'notify_job', lambda _cur, queue: notified.append(queue)
        )

        instance.requeue_abandoned()

        assert notified == [sql.QUEUE_HIGH, sql.QUEUE_DEFAULT]

    def test_a_row_that_is_no_longer_this_workers_is_dropped_not_retried(
        self, monkeypatch
    ):
        instance = _worker()
        instance._abandoned = ['task-1']
        monkeypatch.setattr(
            worker_mod.sql, 'requeue_uncharged', lambda *a, **k: False
        )
        notified = []
        monkeypatch.setattr(
            worker_mod.sql, 'notify_job', lambda _cur, queue: notified.append(queue)
        )

        instance.requeue_abandoned()

        assert instance._abandoned == []
        assert notified == []

    def test_a_requeue_that_raises_is_retried_on_the_next_loop(self, monkeypatch):
        instance = _worker()
        instance._abandoned = ['task-1']
        monkeypatch.setattr(
            worker_mod.sql, 'requeue_uncharged',
            _raising(RuntimeError('the connection is still down')),
        )
        notified = []
        monkeypatch.setattr(
            worker_mod.sql, 'notify_job', lambda _cur, queue: notified.append(queue)
        )

        instance.requeue_abandoned()

        assert instance._abandoned == ['task-1']
        assert notified == []

    def test_one_row_that_cannot_be_put_back_does_not_hold_up_the_others(
        self, monkeypatch
    ):
        instance = _worker()
        instance._abandoned = ['boom-1', 'task-2']

        def _requeue(_cur, task_id, worker_id=None):
            if task_id == 'boom-1':
                raise RuntimeError('deadlock detected')
            return True

        monkeypatch.setattr(worker_mod.sql, 'requeue_uncharged', _requeue)
        notified = []
        monkeypatch.setattr(
            worker_mod.sql, 'notify_job', lambda _cur, queue: notified.append(queue)
        )

        instance.requeue_abandoned()

        assert instance._abandoned == ['boom-1']
        assert notified == [sql.QUEUE_HIGH, sql.QUEUE_DEFAULT]

    def test_a_worker_with_nothing_abandoned_never_touches_the_database(self, monkeypatch):
        instance = _worker()
        slept = []
        monkeypatch.setattr(worker_mod.time, 'sleep', lambda seconds: slept.append(seconds))
        notified = []
        monkeypatch.setattr(
            worker_mod.sql, 'notify_job', lambda _cur, queue: notified.append(queue)
        )

        instance.requeue_abandoned()

        instance._conn.cursor.assert_not_called()
        instance._conn.commit.assert_not_called()
        instance._conn.rollback.assert_not_called()
        assert slept == []
        assert notified == []


class TestTheFreeRetryIsBoundedPerRow:
    def _wired(self, monkeypatch, charged_status=config.TASK_STATUS_NEW, owner=None):
        instance = _worker()
        monkeypatch.setattr(worker_mod.time, 'sleep', lambda _seconds: None)
        monkeypatch.setattr(worker_mod.sql, 'notify_job', lambda *a: None)
        seen = {'uncharged': [], 'charged': []}
        monkeypatch.setattr(
            worker_mod.sql, 'requeue_uncharged',
            lambda _cur, task_id, worker_id=None: (
                seen['uncharged'].append(task_id) or True
            ),
        )
        monkeypatch.setattr(
            worker_mod.sql, 'current_row',
            lambda _cur, _task_id: _running_row(
                instance.identity if owner is None else owner
            ),
        )
        monkeypatch.setattr(
            worker_mod.sql, 'requeue_or_fail',
            lambda _cur, task_id, _now, _details: (
                seen['charged'].append(task_id) or charged_status
            ),
        )
        return instance, seen

    def _spin(self, instance, passes, task_id='task-1'):
        for _ in range(passes):
            instance._abandoned = [task_id]
            instance.requeue_abandoned()

    def test_the_first_free_passes_charge_nothing(self, monkeypatch):
        instance, seen = self._wired(monkeypatch)

        self._spin(instance, worker_mod.UNCHARGED_REQUEUE_LIMIT)

        assert seen['uncharged'] == ['task-1'] * worker_mod.UNCHARGED_REQUEUE_LIMIT
        assert seen['charged'] == []

    def test_the_next_requeue_of_the_same_row_costs_an_attempt(self, monkeypatch):
        instance, seen = self._wired(monkeypatch)

        self._spin(instance, worker_mod.UNCHARGED_REQUEUE_LIMIT + 1)

        assert seen['uncharged'] == ['task-1'] * worker_mod.UNCHARGED_REQUEUE_LIMIT
        assert seen['charged'] == ['task-1']

    def test_every_requeue_past_the_limit_keeps_costing_an_attempt(self, monkeypatch):
        instance, seen = self._wired(monkeypatch)

        self._spin(instance, worker_mod.UNCHARGED_REQUEUE_LIMIT + 3)

        assert seen['uncharged'] == ['task-1'] * worker_mod.UNCHARGED_REQUEUE_LIMIT
        assert seen['charged'] == ['task-1'] * 3

    def test_a_different_row_keeps_its_own_free_passes(self, monkeypatch):
        instance, seen = self._wired(monkeypatch)

        self._spin(instance, worker_mod.UNCHARGED_REQUEUE_LIMIT + 1)
        self._spin(instance, 1, task_id='task-2')

        assert seen['uncharged'][-1] == 'task-2'
        assert seen['charged'] == ['task-1']

    def test_a_charged_requeue_that_lands_on_new_wakes_the_queues(self, monkeypatch):
        instance, _seen = self._wired(monkeypatch)
        notified = []
        monkeypatch.setattr(
            worker_mod.sql, 'notify_job', lambda _cur, queue: notified.append(queue)
        )

        self._spin(instance, worker_mod.UNCHARGED_REQUEUE_LIMIT + 1)

        assert notified[-2:] == [sql.QUEUE_HIGH, sql.QUEUE_DEFAULT]

    def test_a_charged_requeue_that_failed_the_row_neither_wakes_nor_remembers(
        self, monkeypatch
    ):
        instance, seen = self._wired(monkeypatch, charged_status=config.TASK_STATUS_FAIL)
        notified = []
        monkeypatch.setattr(
            worker_mod.sql, 'notify_job', lambda _cur, queue: notified.append(queue)
        )

        self._spin(instance, worker_mod.UNCHARGED_REQUEUE_LIMIT)
        notified.clear()

        self._spin(instance, 1)

        assert seen['charged'] == ['task-1']
        assert notified == []
        assert instance._abandoned == []
        assert instance._uncharged == {}

    def test_a_charged_requeue_is_refused_when_the_row_belongs_to_another_worker(
        self, monkeypatch
    ):
        instance, seen = self._wired(monkeypatch, owner='audiomuse-worker-default-hostB-99')

        self._spin(instance, worker_mod.UNCHARGED_REQUEUE_LIMIT + 1)

        assert seen['charged'] == []
        assert instance._abandoned == []
        assert instance._uncharged == {}

    @pytest.mark.parametrize('terminal', config.TASK_STATUS_TERMINAL)
    def test_a_charged_requeue_is_refused_when_the_row_is_already_terminal(
        self, monkeypatch, terminal
    ):
        instance, seen = self._wired(monkeypatch)
        monkeypatch.setattr(
            worker_mod.sql, 'current_row',
            lambda _cur, _task_id: dict(_running_row(instance.identity), status=terminal),
        )

        self._spin(instance, worker_mod.UNCHARGED_REQUEUE_LIMIT + 1)

        assert seen['charged'] == []
        assert instance._abandoned == []

    def test_a_charged_requeue_is_refused_when_the_row_has_been_deleted(
        self, monkeypatch
    ):
        instance, seen = self._wired(monkeypatch)
        monkeypatch.setattr(worker_mod.sql, 'current_row', lambda _cur, _task_id: None)

        self._spin(instance, worker_mod.UNCHARGED_REQUEUE_LIMIT + 1)

        assert seen['charged'] == []
        assert instance._abandoned == []

    def test_a_row_dropped_because_it_is_no_longer_ours_forgets_its_free_passes(
        self, monkeypatch
    ):
        instance, _seen = self._wired(monkeypatch)
        instance._uncharged = {'task-1': 1}
        monkeypatch.setattr(worker_mod.sql, 'requeue_uncharged', lambda *a, **k: False)

        self._spin(instance, 1)

        assert instance._uncharged == {}


class TestARepeatedLossCannotSpinTheLoopWithoutADelay:
    def _wired(self, monkeypatch):
        instance = _worker()
        slept = []
        monkeypatch.setattr(worker_mod.time, 'sleep', lambda seconds: slept.append(seconds))
        monkeypatch.setattr(worker_mod.sql, 'notify_job', lambda *a: None)
        monkeypatch.setattr(worker_mod.sql, 'requeue_uncharged', lambda *a, **k: True)
        monkeypatch.setattr(
            worker_mod.sql, 'current_row',
            lambda _cur, _task_id: _running_row(instance.identity),
        )
        monkeypatch.setattr(
            worker_mod.sql, 'requeue_or_fail',
            lambda *a, **k: config.TASK_STATUS_NEW,
        )
        return instance, slept

    def _spin(self, instance, passes):
        for _ in range(passes):
            instance._abandoned = ['task-1']
            instance.requeue_abandoned()

    def test_the_first_loss_of_a_row_is_put_back_without_waiting(self, monkeypatch):
        instance, slept = self._wired(monkeypatch)

        self._spin(instance, 1)

        assert slept == []

    def test_a_repeat_loss_waits_the_reconnect_delay_first(self, monkeypatch):
        instance, slept = self._wired(monkeypatch)

        self._spin(instance, 2)

        assert slept == [config.QUEUE_RECONNECT_DELAY_SECONDS]

    def test_the_wait_doubles_for_each_further_loss_of_the_same_row(self, monkeypatch):
        instance, slept = self._wired(monkeypatch)

        self._spin(instance, 3)

        assert slept == [
            config.QUEUE_RECONNECT_DELAY_SECONDS,
            config.QUEUE_RECONNECT_DELAY_SECONDS * 2,
        ]

    def test_the_wait_never_grows_past_the_idle_poll_interval(self, monkeypatch):
        instance, slept = self._wired(monkeypatch)
        monkeypatch.setattr(config, 'QUEUE_POLL_INTERVAL_SECONDS', 3.0)

        self._spin(instance, worker_mod.UNCHARGED_REQUEUE_LIMIT + 3)

        assert max(slept) == 3.0

    def test_the_wait_happens_before_the_database_is_touched_again(self, monkeypatch):
        instance = _worker()
        order = []
        monkeypatch.setattr(worker_mod.time, 'sleep', lambda _s: order.append('wait'))
        monkeypatch.setattr(worker_mod.sql, 'notify_job', lambda *a: None)
        monkeypatch.setattr(
            worker_mod.sql, 'requeue_uncharged',
            lambda *a, **k: order.append('requeue') or True,
        )

        self._spin(instance, 2)

        assert order == ['requeue', 'wait', 'requeue']

    def test_a_row_on_its_first_loss_never_waits_for_an_unrelated_repeat(
        self, monkeypatch
    ):
        instance, slept = self._wired(monkeypatch)
        instance._uncharged = {'task-9': 3}
        instance._abandoned = ['task-1']

        instance.requeue_abandoned()

        assert slept == []


class TestTheUnchargedRequeueCannotResurrectATerminalRow:
    def _executed(self, worker_id=None):
        cur = MagicMock()
        cur.fetchone.return_value = None

        sql.requeue_uncharged(cur, 'task-1', worker_id=worker_id)

        statement, params = cur.execute.call_args.args
        return _normalise(statement), params

    def test_the_statement_only_matches_a_running_row(self):
        statement, _params = self._executed()

        assert _status_literals(_where_clause(statement)) == {config.TASK_STATUS_RUNNING}

    @pytest.mark.parametrize('terminal', config.TASK_STATUS_TERMINAL)
    def test_no_terminal_status_is_named_by_the_guard(self, terminal):
        statement, _params = self._executed()

        assert terminal not in _where_clause(statement)

    def test_naming_a_worker_binds_the_requeue_to_that_worker(self):
        statement, params = self._executed(worker_id='audiomuse-worker-default-hostA-11')
        where = _where_clause(statement)

        assert 'worker_id = %s' in where
        assert 'worker_id IS NULL' in where
        assert _status_literals(where) == {config.TASK_STATUS_RUNNING}
        assert params[-1] == 'audiomuse-worker-default-hostA-11'
        assert params.count('audiomuse-worker-default-hostA-11') == 2

    def test_the_control_plane_still_requeues_without_naming_a_worker(self):
        cur = MagicMock()
        cur.fetchone.return_value = ('task-1',)

        assert sql.requeue_uncharged(cur, 'task-1') is True

        statement, params = cur.execute.call_args.args
        assert params[1] is None
        assert params[2] is None
        assert _status_literals(_where_clause(statement)) == {config.TASK_STATUS_RUNNING}


class TestAdoptingAStagedRowKeepsTheParentTheCallerAsked:
    def _conflict_branch(self):
        cur = MagicMock()
        cur.fetchone.return_value = ('align-1',)
        sql.insert_job(
            cur,
            'align-1',
            'server_sweep',
            'tasks.multiserver_sync.sweep_server',
            parent_task_id='root-1',
            sub_type_identifier='srv-1',
            details={'message': 'go'},
        )
        statement = ' '.join(cur.execute.call_args.args[0].split())
        return statement.split('ON CONFLICT', 1)[1]

    def test_the_adopted_row_gets_the_parent_and_sub_type_it_was_enqueued_with(self):
        conflict = self._conflict_branch()

        assert (
            'parent_task_id = COALESCE(EXCLUDED.parent_task_id, '
            'task_status.parent_task_id)'
        ) in conflict
        assert (
            'sub_type_identifier = COALESCE(EXCLUDED.sub_type_identifier, '
            'task_status.sub_type_identifier)'
        ) in conflict

    def test_the_staged_parent_and_details_survive_an_enqueue_that_carries_neither(self):
        conflict = self._conflict_branch()

        assert 'details = COALESCE(EXCLUDED.details, task_status.details)' in conflict
        assert 'COALESCE(EXCLUDED.parent_task_id, task_status.parent_task_id)' in conflict

    def test_only_a_never_queued_live_row_can_be_adopted(self):
        where = _where_clause(self._conflict_branch())

        assert 'task_status.func IS NULL' in where
        assert _status_literals(where) == set(config.TASK_STATUS_LIVE)

    @pytest.mark.parametrize('terminal', config.TASK_STATUS_TERMINAL)
    def test_a_terminal_row_is_never_adopted_by_a_late_enqueue(self, terminal):
        assert terminal not in _where_clause(self._conflict_branch())


class TestHydratingASharedPayloadCannotRaceTheListener:
    def test_the_shared_payload_is_read_under_the_claim_lock(self, monkeypatch):
        instance = _worker()
        locked = []
        monkeypatch.setattr(
            worker_mod.sql, 'get_shared',
            lambda _cur, _owner, _token: locked.append(instance._claim_txn.locked()) or 'body',
        )

        assert instance.shared_body('owner-1', 'token-1') == 'body'
        assert locked == [True]

    def test_a_cached_payload_needs_neither_the_lock_nor_the_connection(self):
        instance = _worker()
        instance._shared_cache = {'token-1': 'body'}

        assert instance.shared_body('owner-1', 'token-1') == 'body'
        instance._conn.cursor.assert_not_called()
