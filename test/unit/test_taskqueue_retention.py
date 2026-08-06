# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""task_status holds one run: the one happening now, or the recap of the last.

Three statements enforce that and nothing else does. A run that starts empties
the table before inserting itself, a Cancel empties it and leaves one REVOKED
recap, and a run that finishes empties it apart from its own recap. There is no
prune, no cap, no age, no ranking and no knob anywhere in the queue - so the
tests here exist to keep any of those from coming back.

The one thing a wipe cannot do is drop the clustering genre map from a row it
keeps: the recap of a finished run is exactly the row that owned the shared
payload, and that payload can exceed a gigabyte.

Main Features:
* A starting run, a cancel and a finishing run each empty the table
* A refused start leaves the table untouched, wipe included
* The shared slot is cleared on terminal rows and only on terminal rows
* No orphan sweep, superseded-root prune or retention knob exists
"""

import os
import re
from unittest.mock import MagicMock

import pytest

from taskqueue import maintenance, sql

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


class _Cursor:
    def __init__(self, recorder):
        self._recorder = recorder
        self.rowcount = 0

    def execute(self, statement, params=None):
        self._recorder.statements.append(' '.join(statement.split()))

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


class TestATerminalRootDropsItsSharedPayload:
    def test_the_sweep_clears_shared_columns_only_on_terminal_rows(self, conn):
        maintenance.clear_terminal_shared_payloads(conn)

        sweep = next(
            (s for s in conn.statements if 'shared_payload = NULL' in s), None
        )
        assert sweep is not None
        assert "status IN ('SUCCESS','FAIL','REVOKED')" in sweep
        assert 'NEW' not in sweep
        assert 'RUNNING' not in sweep

    def test_it_is_the_only_thing_the_slow_half_does_to_finished_rows(self, conn):
        maintenance.clear_terminal_shared_payloads(conn)

        assert len(conn.statements) == 1, (
            'the three wipes already leave one row, so there is nothing here to '
            'rank, age or delete'
        )


class TestAHandshakeCleansUpAfterItself:
    def test_a_new_control_request_deletes_the_previous_one_and_its_acks(self):
        from taskqueue import control

        statement = ' '.join(control._CLEAR_PREVIOUS_CONTROL_ROWS.split())

        assert statement.startswith('DELETE FROM task_status')
        assert 'task_type = %s' in statement
        assert 'task_id <> %s' in statement
        assert 'parent_task_id <> %s' in statement, (
            'a handshake leaves one request row and one ack row per listener, and '
            'nothing else ever removed them: repeated wizard saves accumulated a '
            'set per save until some unrelated task happened to start or finish'
        )


class TestAHandshakeIsNotTheRunItInterrupts:
    def test_a_finished_control_request_keeps_the_running_task_recap(self):
        import config
        import database

        recorder = _Recorder()
        db = MagicMock()
        db.cursor.side_effect = lambda *a, **k: _Cursor(recorder)

        database._collapse_finished_task(
            db, 'ctl-1', 'worker_control', None, config.TASK_STATUS_SUCCESS
        )

        assert recorder.statements == [], (
            'a restart handshake finishes DURING a run - the provider migration '
            'publishes one and keeps reporting - so collapsing here made the '
            'handshake the surviving recap and deleted the migration the wizard '
            'was still polling, which then answered 404 with no countdown'
        )


class TestTaskStatusHoldsOneRunAtATime:
    def test_a_finished_task_leaves_exactly_its_own_recap(self):
        import config
        import database

        recorder = _Recorder()
        db = MagicMock()
        db.cursor.side_effect = lambda *a, **k: _Cursor(recorder)

        database._collapse_finished_task(
            db, 'done-1', 'main_analysis', None, config.TASK_STATUS_SUCCESS
        )

        assert len(recorder.statements) == 1
        statement = recorder.statements[0]
        assert statement.startswith('DELETE FROM task_status WHERE task_id <> %s')
        assert "status IN ('SUCCESS','FAIL','REVOKED')" in statement, (
            'a finishing task drops the OTHER finished rows and only those; taking '
            'live ones killed work still going - the migration queues its alignment '
            'and only then reports terminal, and a radio or plugin task can run '
            'alongside'
        )

    def test_a_new_run_drops_the_finished_rows_and_spares_the_live_ones(self):
        recorder = _Recorder()
        cur = _Cursor(recorder)

        sql.clear_task_status(cur)

        assert len(recorder.statements) == 1
        statement = recorder.statements[0]
        assert statement.startswith('DELETE FROM task_status WHERE')
        assert "status IN ('SUCCESS','FAIL','REVOKED')" in statement, (
            'the whole retention policy: no cap, no age, no ranking - and scoped to '
            'finished rows, because an unconditional wipe deleted work that was '
            'still running and a missing row IS the cancellation signal'
        )
        assert "live.status IN ('NEW','RUNNING')" in statement, (
            "and it must spare a finished child whose parent is still draining: a "
            "fan-out leaves them terminal until the next reap, so wiping them made "
            "the parent wait for children that no longer existed"
        )

    def test_a_child_task_never_wipes_the_run_it_belongs_to(self):
        import config
        import database

        recorder = _Recorder()
        db = MagicMock()
        db.cursor.side_effect = lambda *a, **k: _Cursor(recorder)

        database._collapse_finished_task(
            db, 'child-1', 'clustering_batch', 'parent-1', config.TASK_STATUS_SUCCESS
        )

        assert recorder.statements == []

    def test_retention_needs_no_tuning_knob(self):
        import config

        assert not hasattr(config, 'QUEUE_MAX_TERMINAL_ROOTS')
        assert not hasattr(config, 'QUEUE_MAX_SELF_MANAGED_ROOTS')
        assert not hasattr(config, 'QUEUE_MAX_ROOTS_PER_TYPE')


class TestARefusedStartLeavesTheTableAlone:
    def test_the_wipe_is_inside_the_savepoint_the_insert_rolls_back_to(self):
        with open(
            os.path.join(_REPO_ROOT, 'taskqueue', '__init__.py'), encoding='utf-8'
        ) as handle:
            source = handle.read()

        savepoint = source.index('SAVEPOINT audiomuse_enqueue')
        wipe = source.index('sql.clear_task_status(cur)')
        rollback = source.index('ROLLBACK TO SAVEPOINT audiomuse_enqueue')
        release = source.index('RELEASE SAVEPOINT audiomuse_enqueue')

        assert savepoint < wipe < rollback, (
            'a start refused by the unique index must undo its own wipe; with the '
            'wipe outside the savepoint the DELETE stayed pending on a borrowed '
            'connection and the next caller committed it'
        )
        assert source.index('raise TaskNotQueued') < release, (
            'the other refusal has to undo the wipe too, so the savepoint may only '
            'be released once the row is provably in'
        )


class TestNothingElsePrunesTheTable:
    def test_no_orphan_child_sweep_survives_anywhere(self):
        import database
        import taskqueue

        assert not hasattr(sql, 'PRUNE_ORPHAN_CHILDREN')
        assert not hasattr(sql, 'prune_orphan_children')
        assert not hasattr(taskqueue, 'prune_orphan_children')
        assert not hasattr(maintenance, 'prune_terminal_roots')
        assert not hasattr(database, 'prune_task_status_history')

    def test_no_layer_reintroduces_a_prune_of_its_own(self):
        orphan_shape = re.compile(
            r'child\.parent_task_id IS NOT NULL.*?NOT EXISTS.*?parent\.status IN',
            re.DOTALL,
        )
        superseded_shape = re.compile(r'newer\.id > old\.id')
        for relative in (
            'database.py',
            os.path.join('taskqueue', 'sql.py'),
            os.path.join('taskqueue', 'maintenance.py'),
            os.path.join('taskqueue', '__init__.py'),
        ):
            with open(os.path.join(_REPO_ROOT, relative), encoding='utf-8') as handle:
                source = handle.read()
            assert not orphan_shape.search(source), relative
            assert not superseded_shape.search(source), relative


class TestTheWorkerSafetyNetDropsTheSharedSlot:
    def test_finish_task_nulls_shared_columns_with_func_and_payload(self):
        recorder = _Recorder()
        cur = _Cursor(recorder)

        sql.finish_task(cur, 'task-1', 'SUCCESS', {}, 123.0, worker_id='w-1')

        statement = recorder.statements[0]
        assert 'func = NULL' in statement
        assert 'payload = NULL' in statement
        assert 'shared_token = NULL' in statement
        assert 'shared_payload = NULL' in statement
