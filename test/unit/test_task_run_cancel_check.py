# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The one cancel check stops a task whose own row is already terminal.

The check used to stop a task only when its row was gone or REVOKED. A parent
that gives up on a child writes that child's row as FAIL, so the child kept
running until it noticed its parent was over, holding the one default worker
its siblings needed, and when it finally returned the queue logged it as a
task that had written its own terminal row. A terminal own row means the
verdict is already in, whoever wrote it, and there is nothing left to report.

A second thing lived beside the check for a while: a pre-check, hand-written
in six places, that read the row before any work to refuse a run whose row
was gone or terminal. It answered the same question with a second reader, so it
is gone, and the first forced check is where every task learns its row is
dead, before its first report.

Main Features:
* Any terminal own row stops the task, not only REVOKED
* A missing own row still stops it, which is what the global cancel leaves
* A supervised child stops when its parent is terminal or gone
* A live task under a live parent goes on, and a root watches only itself
* The first forced check stops a task whose row is gone or terminal before it
  has written anything, and a read that fails lets it go on
* The prologue reads no row; the check is the only reader
* The shared per-server loop lets a closed connection through, OperationalError
  and InterfaceError alike, instead of counting it as one server's failure
"""

from contextlib import nullcontext

import pytest
from psycopg2 import InterfaceError, OperationalError

from config import TASK_STATUS_RUNNING, TASK_STATUS_TERMINAL
from taskqueue import TaskCancelled, TaskFailed
from tasks import task_run


class TestTheCancelCheckReadsTheOwnRowAndTheParent:
    @pytest.mark.parametrize('status', TASK_STATUS_TERMINAL)
    def test_a_terminal_own_row_stops_the_task(self, status):
        with pytest.raises(TaskCancelled):
            task_run._raise_if_cancelled('t1', None, {'t1': status})

    def test_a_missing_own_row_stops_the_task(self):
        with pytest.raises(TaskCancelled):
            task_run._raise_if_cancelled('t1', None, {})

    @pytest.mark.parametrize('status', TASK_STATUS_TERMINAL)
    def test_a_terminal_parent_stops_a_supervised_child(self, status):
        with pytest.raises(TaskCancelled):
            task_run._raise_if_cancelled(
                'c1', 'p1', {'c1': TASK_STATUS_RUNNING, 'p1': status}
            )

    def test_a_missing_parent_stops_a_supervised_child(self):
        with pytest.raises(TaskCancelled):
            task_run._raise_if_cancelled('c1', 'p1', {'c1': TASK_STATUS_RUNNING})

    def test_a_live_task_under_a_live_parent_goes_on(self):
        task_run._raise_if_cancelled(
            'c1', 'p1', {'c1': TASK_STATUS_RUNNING, 'p1': TASK_STATUS_RUNNING}
        )

    def test_a_root_watches_only_itself(self):
        task_run._raise_if_cancelled('r1', None, {'r1': TASK_STATUS_RUNNING})


def _check_reading(monkeypatch, statuses):
    monkeypatch.setattr(task_run, '_open_check_connection', lambda: object())
    reads = []

    def _read(_conn, ids):
        reads.append(list(ids))
        if isinstance(statuses, Exception):
            raise statuses
        return statuses

    monkeypatch.setattr(task_run, '_read_task_statuses', _read)
    return reads


class TestTheFirstForcedCheckRunsBeforeAnyWork:
    def test_a_task_whose_row_is_gone_stops_before_its_first_report(self, monkeypatch):
        reads = _check_reading(monkeypatch, {})

        with task_run.cancel_guard('t1') as cancel:
            with pytest.raises(TaskCancelled):
                cancel(force=True)

        assert reads == [['t1']]

    @pytest.mark.parametrize('status', TASK_STATUS_TERMINAL)
    def test_a_task_whose_row_is_terminal_stops_before_its_first_report(
        self, monkeypatch, status
    ):
        _check_reading(monkeypatch, {'t1': status})

        with task_run.cancel_guard('t1') as cancel:
            with pytest.raises(TaskCancelled):
                cancel(force=True)

    def test_a_read_that_fails_lets_the_task_go_on(self, monkeypatch):
        _check_reading(monkeypatch, RuntimeError('db blip'))

        with task_run.cancel_guard('t1') as cancel:
            cancel(force=True)

    def test_a_task_with_no_claim_has_nothing_to_check(self, monkeypatch):
        reads = _check_reading(monkeypatch, {})

        with task_run.cancel_guard(None) as cancel:
            cancel(force=True)

        assert reads == []


class TestThePrologueReadsNoRow:
    def test_it_resolves_the_ids_and_nothing_else(self, monkeypatch):
        import taskqueue

        monkeypatch.setattr(taskqueue, 'current_task_id', lambda: 'claimed-1')

        assert task_run.task_run_prologue() == ('claimed-1', 'claimed-1')
        assert task_run.task_run_prologue('given-1') == ('claimed-1', 'given-1')
        assert not hasattr(task_run, 'get_task_info_from_db'), (
            'the row is read by the cancel check and by nothing else; a second '
            'reader is how six copies of the same pre-check came to exist'
        )

    def test_an_unclaimed_run_gets_a_fresh_id(self, monkeypatch):
        import taskqueue

        monkeypatch.setattr(taskqueue, 'current_task_id', lambda: None)

        claimed, task_id = task_run.task_run_prologue()

        assert claimed is None
        assert task_id


class TestTheSharedServerLoopLetsALostConnectionThrough:
    def _loop(self, monkeypatch, error):
        from tasks.mediaserver import registry

        monkeypatch.setattr(registry, 'servers_for_scope', lambda scope, conn=None: [None, None])
        monkeypatch.setattr(registry, 'bind', lambda server, conn=None: nullcontext())
        seen = []

        def step(_server, name):
            seen.append(name)
            raise error

        return seen, step

    @pytest.mark.parametrize('error', [
        InterfaceError('connection already closed'),
        OperationalError('server closed the connection unexpectedly'),
    ])
    def test_a_closed_connection_is_not_a_per_server_failure(self, monkeypatch, error):
        seen, step = self._loop(monkeypatch, error)

        with pytest.raises(type(error)):
            task_run.for_each_server_in_scope('all', step)

        assert len(seen) == 1, (
            'psycopg2 reports a connection closed under the caller as InterfaceError; '
            'swallowed as one server\'s failure the loop marched over every remaining '
            'server on a dead connection and the worker charged an attempt where it '
            'has an uncharged path for exactly this'
        )

    @pytest.mark.parametrize('error', [TaskFailed('no retry'), TaskCancelled('stop')])
    def test_the_tasks_own_verdict_passes_through(self, monkeypatch, error):
        seen, step = self._loop(monkeypatch, error)

        with pytest.raises(type(error)):
            task_run.for_each_server_in_scope('all', step)

        assert len(seen) == 1

    def test_any_other_error_is_one_servers_failure(self, monkeypatch):
        seen, step = self._loop(monkeypatch, RuntimeError('502 from the media server'))

        servers, results, failed = task_run.for_each_server_in_scope('all', step)

        assert len(seen) == 2
        assert results == []
        assert failed == ['default server', 'default server']
