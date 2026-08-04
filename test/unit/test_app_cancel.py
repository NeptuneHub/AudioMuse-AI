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

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def stub_the_start_lock(monkeypatch):
    import app_helper

    monkeypatch.setattr(app_helper, 'main_task_start_lock', nullcontext)
    monkeypatch.setattr(
        app_helper.database,
        'janitor_cycle_lock',
        lambda conn=None, blocking=False, timeout_seconds=None: nullcontext(True),
    )


class _FakeCursor:
    """Records executed SQL and answers the snapshot SELECT."""

    def __init__(self, rows, protected=()):
        self._rows = rows
        self._protected = list(protected)
        self._pending = []
        self._returning = None
        self.executed = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))
        self._returning = None
        if sql.strip().upper().startswith("SELECT"):
            self._pending = (
                list(self._protected) if 'FROM migration_session' in sql
                else list(self._rows)
            )
        else:
            self.rowcount = len(self._rows)
            if 'app_config' in sql and 'RETURNING value' in sql:
                self._returning = ('1',)

    def fetchall(self):
        return list(self._pending)

    def fetchone(self):
        if self._returning is None:
            raise AssertionError(
                "fetchone() on a statement this fake does not answer: "
                + (self.executed[-1][0] if self.executed else '<none>')
            )
        return self._returning

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
def protected_cancel_env():
    cur = _FakeCursor(_rows(), protected=[('exec-1', 'align-1')])
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

    cur, save, _hist = cancel_env
    cancel_job_and_children_recursive('analysis-1')

    save.assert_not_called()
    recap = [
        params for sql, params in cur.executed
        if sql.startswith("INSERT INTO task_status")
    ]
    assert len(recap) == 1
    assert recap[0][:3] == ('analysis-1', 'main_analysis', 'REVOKED')


def test_history_is_recorded_after_the_atomic_cancel_tombstone(cancel_env):
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
    recap_idx = next(
        i for i, (sql, _) in enumerate(cur.executed) if sql.startswith("INSERT INTO task_status")
    )
    assert wipe_idx < recap_idx


def test_first_cancel_commits_tombstone_then_disables_retries_and_stops_all_jobs(
    monkeypatch,
):
    import app_helper

    events = []
    cur = _FakeCursor(_rows())
    db = MagicMock()
    db.cursor.return_value = cur
    db.commit.side_effect = lambda: events.append(('commit', None))
    monkeypatch.setattr(app_helper, 'get_db', lambda: db)
    monkeypatch.setattr(app_helper, 'record_task_history', lambda *a, **k: None)
    monkeypatch.setattr(app_helper.redis_conn, 'scan_iter', lambda **k: [])
    high_queue = MagicMock()
    default_queue = MagicMock()
    monkeypatch.setattr(app_helper, 'rq_queue_high', high_queue)
    monkeypatch.setattr(app_helper, 'rq_queue_default', default_queue)

    running = MagicMock(retries_left=3)
    running.get_status.return_value = 'started'
    queued = MagicMock(retries_left=3)
    queued.get_status.return_value = 'queued'
    jobs = {'cluster-parent': running, 'cluster-batch': queued}
    monkeypatch.setattr(
        app_helper.Job,
        'fetch',
        lambda task_id, connection=None: jobs[task_id],
    )
    monkeypatch.setattr(
        app_helper.rq_job_state,
        'forbid_retries',
        lambda task_id, conn: events.append(('retry-zero', task_id)) or True,
    )
    monkeypatch.setattr(
        app_helper,
        'send_stop_job_command',
        lambda conn, task_id: events.append(('stop', task_id)),
    )
    queued.cancel.side_effect = lambda: events.append(('cancel', 'cluster-batch'))
    high_queue.job_ids = ['cluster-parent', 'cluster-batch']
    default_queue.job_ids = []
    high_queue.empty.side_effect = lambda: events.append(('empty', 'high'))
    default_queue.empty.side_effect = lambda: events.append(('empty', 'default'))

    app_helper.cancel_job_and_children_recursive('cluster-parent')

    commit_idx = events.index(('commit', None))
    assert commit_idx < events.index(('retry-zero', 'cluster-parent'))
    assert events.index(('retry-zero', 'cluster-parent')) < events.index(
        ('stop', 'cluster-parent')
    )
    assert events.index(('retry-zero', 'cluster-batch')) < events.index(
        ('cancel', 'cluster-batch')
    )
    assert ('empty', 'high') in events and ('empty', 'default') in events


def test_cancel_reports_incomplete_when_redis_queue_cannot_be_emptied(cancel_env):
    import app_helper

    cur, _save, _hist = cancel_env
    app_helper.rq_queue_high.empty.side_effect = RuntimeError('redis unavailable')

    with pytest.raises(app_helper.CancellationIncompleteError):
        app_helper.cancel_job_and_children_recursive('analysis-1')

    assert any(sql.startswith('INSERT INTO task_status') for sql, _ in cur.executed)


def test_cancel_takes_blocking_janitor_lock_before_main_start_lock(monkeypatch):
    import app_helper

    events = []

    class _Cycle:
        def __enter__(self):
            events.append('cycle-enter')
            return True

        def __exit__(self, *_args):
            events.append('cycle-exit')

    class _Main:
        def __enter__(self):
            events.append('main-enter')

        def __exit__(self, *_args):
            events.append('main-exit')

    monkeypatch.setattr(app_helper, 'get_db', lambda: object())
    monkeypatch.setattr(
        app_helper.database,
        'janitor_cycle_lock',
        lambda conn=None, blocking=False, timeout_seconds=None: _Cycle(),
    )
    monkeypatch.setattr(app_helper, 'main_task_start_lock', _Main)
    monkeypatch.setattr(
        app_helper, '_cancel_job_and_children_locked',
        lambda task_id, reason: events.append('cancel') or 0,
    )

    assert app_helper.cancel_job_and_children_recursive('cluster-parent') == 0
    assert events == [
        'cycle-enter', 'main-enter', 'cancel', 'main-exit', 'cycle-exit'
    ]


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


def test_cancelling_an_in_process_task_revokes_its_row_and_spares_the_queues(cancel_env):
    """The alchemy radio runs inside the web process with no RQ job, so the global
    cancel would stop nothing while still emptying both queues and deleting every
    task_status row - destroying an unrelated queued analysis to cancel something it
    cannot reach."""
    from app_helper import revoke_inline_task_row

    cur, save, _hist = cancel_env
    with patch(
        'app_helper.get_task_info_from_db',
        return_value={'task_id': 'radio-1', 'task_type': 'alchemy_radio'},
    ):
        message = revoke_inline_task_row('radio-1')

    assert message
    assert save.call_args[0][:3] == ('radio-1', 'alchemy_radio', 'REVOKED')
    statements = [sql for sql, _ in cur.executed]
    assert not any(s.startswith("DELETE FROM task_status") for s in statements)


def test_cancelling_a_queued_task_is_not_diverted_to_the_in_process_path(cancel_env):
    from app_helper import revoke_inline_task_row

    _cur, save, _hist = cancel_env
    with patch(
        'app_helper.get_task_info_from_db',
        return_value={'task_id': 'analysis-1', 'task_type': 'main_analysis'},
    ):
        assert revoke_inline_task_row('analysis-1') is None

    save.assert_not_called()


def test_a_failed_status_query_is_not_an_empty_answer_and_leaves_the_sweep_running():
    """Absence means cancelled; an unreachable DB does not."""
    from tasks.multiserver_sync import make_cancel_check

    conn = MagicMock()
    conn.cursor.side_effect = RuntimeError("database is unreachable")

    with patch('tasks.multiserver_sync.connect_raw', return_value=conn):
        check, close = make_cancel_check('sweep-1')
        check()  # must not raise
        close()


def test_protected_delete_spares_the_handshake_rows_without_sparing_every_root(
    protected_cancel_env,
):
    from app_helper import cancel_job_and_children_recursive

    cur, _save, _hist = protected_cancel_env
    cancel_job_and_children_recursive('analysis-1')

    deletes = [
        (sql, params) for sql, params in cur.executed if sql.startswith('DELETE')
    ]
    assert len(deletes) == 1
    sql, params = deletes[0]
    # Every root row has parent_task_id IS NULL, and NOT (NULL = ANY(...)) is NULL
    # rather than TRUE, so the unguarded predicate matched NO root at all and the
    # recap INSERT then aborted the whole cancellation on the UNIQUE task_id.
    assert 'parent_task_id IS NULL OR NOT (parent_task_id = ANY(%s))' in sql
    assert sorted(params[0]) == ['align-1', 'exec-1']
    assert params[0] == params[1]


def test_protected_ids_are_not_cancelled_in_rq(protected_cancel_env):
    import app_helper

    cur, _save, _hist = protected_cancel_env
    app_helper.rq_queue_high.job_ids = ['exec-1', 'align-1', 'analysis-1']
    app_helper.rq_queue_default.job_ids = []
    with patch.object(app_helper, 'Job') as job_cls:
        job_cls.fetch.return_value.get_status.return_value = 'finished'
        app_helper.cancel_job_and_children_recursive('analysis-1')

    fetched = [call[0][0] for call in job_cls.fetch.call_args_list]
    # The loop really ran - the unprotected id went through it.
    assert 'analysis-1' in fetched
    assert 'exec-1' not in fetched
    assert 'align-1' not in fetched


def test_a_protected_job_id_gets_no_revoked_recap_row(protected_cancel_env):
    from app_helper import cancel_job_and_children_recursive

    cur, _save, _hist = protected_cancel_env
    cancel_job_and_children_recursive('exec-1')

    inserts = [sql for sql, _ in cur.executed if sql.startswith('INSERT')]
    assert not any('INTO task_status' in sql for sql in inserts)


def test_a_protected_cancel_still_drops_every_unprotected_queued_job(
    protected_cancel_env,
):
    import app_helper

    _cur, _save, _hist = protected_cancel_env
    app_helper.rq_queue_high.job_ids = ['exec-1', 'align-1', 'analysis-1']
    app_helper.rq_queue_default.job_ids = ['other-1']
    with patch.object(app_helper, 'Job') as job_cls:
        job_cls.fetch.return_value.get_status.return_value = 'finished'
        app_helper.cancel_job_and_children_recursive('analysis-1')

    removed_high = [c[0][0] for c in app_helper.rq_queue_high.remove.call_args_list]
    removed_default = [c[0][0] for c in app_helper.rq_queue_default.remove.call_args_list]
    # Skipping the whole block when anything was protected left every queued job in
    # place, so work still STARTED after the user pressed Cancel.
    assert removed_high == ['analysis-1']
    assert removed_default == ['other-1']
    app_helper.rq_queue_high.empty.assert_not_called()
    app_helper.rq_queue_default.empty.assert_not_called()


def test_a_protected_cancel_is_not_reported_as_incomplete(protected_cancel_env):
    import app_helper

    _cur, _save, _hist = protected_cancel_env
    app_helper.rq_queue_high.job_ids = []
    app_helper.rq_queue_default.job_ids = []
    with patch.object(app_helper, 'Job') as job_cls:
        job_cls.fetch.return_value.get_status.return_value = 'finished'
        # Preserving the handshake is the DESIGNED outcome. Recording it as an RQ
        # error made every successful cancel raise and answer the user a 503.
        cancelled = app_helper.cancel_job_and_children_recursive('analysis-1')

    assert isinstance(cancelled, int)


def test_cancel_proceeds_when_the_janitor_lock_times_out(monkeypatch):
    import app_helper
    from contextlib import nullcontext

    events = []
    monkeypatch.setattr(app_helper, 'get_db', lambda: MagicMock())
    monkeypatch.setattr(app_helper, 'main_task_start_lock', nullcontext)
    monkeypatch.setattr(
        app_helper.database,
        'janitor_cycle_lock',
        lambda conn=None, blocking=False, timeout_seconds=None: nullcontext(False),
    )
    monkeypatch.setattr(
        app_helper, '_cancel_job_and_children_locked',
        lambda task_id, reason: events.append('cancel') or 4,
    )

    # A 503 with nothing cancelled is strictly worse than an unserialized cancel:
    # it is exactly the "click Cancel several times" symptom users reported.
    assert app_helper.cancel_job_and_children_recursive('cluster-parent') == 4
    assert events == ['cancel']


def test_cancel_passes_a_bounded_wait_to_the_janitor_lock(monkeypatch):
    import app_helper
    from contextlib import nullcontext

    seen = {}

    def fake_lock(conn=None, blocking=False, timeout_seconds=None):
        seen['blocking'] = blocking
        seen['timeout'] = timeout_seconds
        return nullcontext(True)

    monkeypatch.setattr(app_helper, 'get_db', lambda: MagicMock())
    monkeypatch.setattr(app_helper, 'main_task_start_lock', nullcontext)
    monkeypatch.setattr(app_helper.database, 'janitor_cycle_lock', fake_lock)
    monkeypatch.setattr(
        app_helper, '_cancel_job_and_children_locked', lambda task_id, reason: 0,
    )

    app_helper.cancel_job_and_children_recursive('cluster-parent')

    assert seen['blocking'] is True
    assert seen['timeout'] == app_helper.CANCEL_JANITOR_LOCK_WAIT_SECONDS
    assert 0 < seen['timeout'] <= 60
