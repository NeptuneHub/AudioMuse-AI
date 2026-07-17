# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The Stop path: cancel_job_and_children_recursive and the cancel endpoints.

The cancel WIPES task_status (so the table cannot grow without bound) and leaves a
single REVOKED recap row for the id the user actually cancelled. The wipe is
therefore the cancellation signal itself: every long task polls its own row, and a
task that can no longer FIND its row has been cancelled. Reading a missing row as
"not revoked, carry on" is the original bug - it let a cancelled analysis keep
enqueuing albums onto the queue the cancel had just emptied.

Main Features:
* The global cancel deletes every task_status row and leaves one REVOKED recap
* task_history is snapshotted BEFORE task_status is wiped, so history survives
* Every cooperative check treats a missing row as revoked (analysis, sweep, clustering)
* A failed status QUERY is not an empty answer, and leaves the task running
"""

from unittest.mock import MagicMock, patch

import pytest


class _FakeCursor:
    """Records executed SQL and answers the snapshot SELECT."""

    def __init__(self, rows):
        self._rows = rows
        self.executed = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))
        if sql.strip().upper().startswith("SELECT"):
            self._pending = list(self._rows)
        else:
            self.rowcount = len(self._rows)

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _rows():
    return [
        {
            'task_id': 'analysis-1', 'task_type': 'main_analysis', 'status': 'PROGRESS',
            'details': None, 'start_time': 1.0, 'end_time': None,
        },
        {
            'task_id': 'sweep-1', 'task_type': 'server_sweep', 'status': 'PROGRESS',
            'details': None, 'start_time': 2.0, 'end_time': None,
        },
    ]


@pytest.fixture
def cancel_env():
    cur = _FakeCursor(_rows())
    db = MagicMock()
    db.cursor.return_value = cur
    with (
        patch('app_helper.get_db', return_value=db),
        patch('app_helper.redis_conn') as redis,
        patch('app_helper.rq_queue_high'),
        patch('app_helper.rq_queue_default'),
        patch('app_helper.save_task_status') as save,
        patch('app_helper.record_task_history') as hist,
    ):
        redis.keys.return_value = []
        yield cur, save, hist


def test_global_cancel_wipes_task_status_so_it_cannot_grow_without_bound(cancel_env):
    from app_helper import cancel_job_and_children_recursive

    cur, _save, _hist = cancel_env
    cancel_job_and_children_recursive('analysis-1')

    statements = [sql for sql, _ in cur.executed]
    assert any(s.startswith("DELETE FROM task_status") for s in statements)


def test_global_cancel_leaves_exactly_one_revoked_recap_row(cancel_env):
    """The only row to survive is the id the user actually cancelled, so the UI has
    one canonical cancelled task to show."""
    from app_helper import cancel_job_and_children_recursive

    _cur, save, _hist = cancel_env
    cancel_job_and_children_recursive('analysis-1')

    save.assert_called_once()
    assert save.call_args[0][0] == 'analysis-1'
    assert save.call_args[0][2] == 'REVOKED'


def test_history_is_snapshotted_before_task_status_is_touched(cancel_env):
    """The dashboard's history must still show what was running when Stop was hit."""
    from app_helper import cancel_job_and_children_recursive

    cur, _save, hist = cancel_env
    cancel_job_and_children_recursive('analysis-1')

    assert hist.call_count == 2
    recorded = {c[0][0]: c[0][2] for c in hist.call_args_list}
    assert recorded == {'analysis-1': 'REVOKED', 'sweep-1': 'REVOKED'}

    wipe_idx = next(
        i for i, (sql, _) in enumerate(cur.executed) if sql.startswith("DELETE FROM task_status")
    )
    select_idx = next(
        i for i, (sql, _) in enumerate(cur.executed) if sql.startswith("SELECT task_id")
    )
    assert select_idx < wipe_idx, "history must be snapshotted before the wipe"


def test_a_sweep_whose_row_was_wiped_treats_that_as_cancelled():
    """The wipe IS the signal. A sweep that can no longer find its own row has been
    cancelled; reading absence as 'carry on' let it run to completion against a queue
    the cancel had already emptied."""
    from tasks.multiserver_sync import make_cancel_check, SweepCancelled

    conn = MagicMock()
    cur = conn.cursor.return_value
    cur.fetchone.return_value = None  # the cancel deleted this sweep's row

    with patch('tasks.multiserver_sync.connect_raw', return_value=conn):
        check, close = make_cancel_check('sweep-1')
        with pytest.raises(SweepCancelled):
            check()
        close()


def test_a_failed_status_query_is_not_an_empty_answer_and_leaves_the_sweep_running():
    """Absence means cancelled; an unreachable DB does not."""
    from tasks.multiserver_sync import make_cancel_check

    conn = MagicMock()
    conn.cursor.side_effect = RuntimeError("database is unreachable")

    with patch('tasks.multiserver_sync.connect_raw', return_value=conn):
        check, close = make_cancel_check('sweep-1')
        check()  # must not raise
        close()
