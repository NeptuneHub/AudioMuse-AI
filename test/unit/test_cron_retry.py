# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Cron retry orchestration: recording blocked runs and re-attempting them.

Covers the retry list lifecycle: a blocked scheduled run is recorded, retried
on later ticks, cleared once it enqueues, and recorded as a visible skip when
it expires without ever running. Also pins the centralized queue guard query
and the speaking 409 body the manual endpoints return.

Main Features:
* get_queue_blocking_task only admits the queue-guard task types
* A blocked cron run records a retry entry with a bounded deadline
* retry_due_cron_jobs enqueues when the guard clears and drops the entry
* A still-blocked retry bumps its attempt count instead of re-enqueueing
* An expired retry is cleared and recorded as a visible skip
* A retry whose task already completed since it was recorded is dropped
* A retry that succeeds moves the cron row's last_run forward
* Plugin cron rows get retry coverage too; alchemy_radio does not
* A retry whose plugin handler is gone is cleared, not bumped
* cron_retry_task_already_done self-heals only on a SUCCESS that started after the retry was recorded
* record_cron_retry stores first_blocked_at as a UTC epoch, so no wall-clock conversion is ever involved
* _touch_cron_last_run never aborts the retry tick on a DB failure
* An expired retry's visible-skip row uses the queue type, not the cron name
* queue_busy_error_body carries the centralized error code and message
"""

from unittest.mock import MagicMock, patch

import config
import database


def _pending_entry(task_type='clustering', retry_until=10**12, attempts=0):
    return {
        'task_type': task_type,
        'retry_until': retry_until,
        'attempts': attempts,
        'first_blocked_at': None,
        'blocker_task_id': None,
        'blocker_task_type': 'main_analysis',
    }


@patch('app_cron.get_db')
def test_retry_runs_the_blocked_cron_job_when_the_guard_clears(mock_get_db):
    from app_cron import retry_due_cron_jobs

    db = MagicMock()
    mock_get_db.return_value = db
    with (
        patch('app_cron.list_pending_cron_retries', return_value=[_pending_entry()]),
        patch(
            'app_cron._cron_row_for_retry',
            return_value={'id': 7, 'task_type': 'clustering'},
        ),
        patch('app_cron.cron_retry_task_already_done', return_value=False),
        patch('app_cron._dispatch_cron_row', return_value='enqueued') as dispatch,
        patch('app_cron.clear_cron_retry') as clear,
        patch('app_cron.bump_cron_retry') as bump,
        patch('app_cron._record_retry_expired') as expired,
        patch('app_cron._touch_cron_last_run') as touch,
    ):
        assert retry_due_cron_jobs() == 1

    dispatch.assert_called_once()
    clear.assert_called_once_with('clustering', conn=db)
    bump.assert_not_called()
    expired.assert_not_called()
    touch.assert_called_once_with(db, 7)


@patch('app_cron.get_db')
def test_retry_keeps_a_still_blocked_entry_and_bumps_its_attempts(mock_get_db):
    from app_cron import retry_due_cron_jobs

    db = MagicMock()
    mock_get_db.return_value = db
    blocker = {'task_id': 'live-1', 'task_type': 'main_analysis', 'status': 'RUNNING'}
    with (
        patch('app_cron.list_pending_cron_retries', return_value=[_pending_entry()]),
        patch('app_cron._cron_row_for_retry', return_value={'task_type': 'clustering'}),
        patch('app_cron.cron_retry_task_already_done', return_value=False),
        patch('app_cron._dispatch_cron_row', return_value='blocked'),
        patch('app_cron.get_queue_blocking_task', return_value=blocker),
        patch('app_cron.clear_cron_retry') as clear,
        patch('app_cron.bump_cron_retry') as bump,
    ):
        retry_due_cron_jobs()

    clear.assert_not_called()
    bump.assert_called_once_with(
        'clustering', blocker_task_id='live-1', blocker_task_type='main_analysis',
        conn=db,
    )


@patch('app_cron.get_db')
def test_an_expired_retry_is_cleared_and_recorded_as_a_visible_skip(mock_get_db):
    from app_cron import retry_due_cron_jobs

    db = MagicMock()
    mock_get_db.return_value = db
    entry = _pending_entry(retry_until=1000)
    with (
        patch('app_cron.list_pending_cron_retries', return_value=[entry]),
        patch('app_cron.clear_cron_retry') as clear,
        patch('app_cron._record_retry_expired') as expired,
        patch('app_cron._dispatch_cron_row') as dispatch,
    ):
        assert retry_due_cron_jobs() == 1

    clear.assert_called_once_with('clustering', conn=db)
    expired.assert_called_once_with('clustering', entry)
    dispatch.assert_not_called()


@patch('app_cron.get_db')
def test_a_retry_whose_task_completed_since_it_was_recorded_is_dropped(mock_get_db):
    from app_cron import retry_due_cron_jobs

    db = MagicMock()
    mock_get_db.return_value = db
    with (
        patch('app_cron.list_pending_cron_retries', return_value=[_pending_entry()]),
        patch('app_cron._cron_row_for_retry', return_value={'task_type': 'clustering'}),
        patch('app_cron.cron_retry_task_already_done', return_value=True),
        patch('app_cron.clear_cron_retry') as clear,
        patch('app_cron._dispatch_cron_row') as dispatch,
    ):
        retry_due_cron_jobs()

    clear.assert_called_once_with('clustering', conn=db)
    dispatch.assert_not_called()


@patch('app_cron.get_db')
def test_a_retry_whose_plugin_handler_is_gone_is_cleared(mock_get_db):
    from app_cron import retry_due_cron_jobs

    db = MagicMock()
    mock_get_db.return_value = db
    with (
        patch(
            'app_cron.list_pending_cron_retries',
            return_value=[_pending_entry(task_type='plugin.demo.sync')],
        ),
        patch(
            'app_cron._cron_row_for_retry',
            return_value={'id': 3, 'task_type': 'plugin.demo.sync'},
        ),
        patch('app_cron.cron_retry_task_already_done', return_value=False),
        patch('app_cron._dispatch_cron_row', return_value='no_handler'),
        patch('app_cron.clear_cron_retry') as clear,
        patch('app_cron.bump_cron_retry') as bump,
    ):
        retry_due_cron_jobs()

    clear.assert_called_once_with('plugin.demo.sync', conn=db)
    bump.assert_not_called()


@patch('app_cron.get_db')
def test_a_retry_with_no_cron_row_left_is_cleared(mock_get_db):
    from app_cron import retry_due_cron_jobs

    db = MagicMock()
    mock_get_db.return_value = db
    with (
        patch('app_cron.list_pending_cron_retries', return_value=[_pending_entry()]),
        patch('app_cron._cron_row_for_retry', return_value=None),
        patch('app_cron.clear_cron_retry') as clear,
        patch('app_cron._dispatch_cron_row') as dispatch,
    ):
        retry_due_cron_jobs()

    clear.assert_called_once_with('clustering', conn=db)
    dispatch.assert_not_called()


@patch('app_cron.get_db')
def test_record_cron_retry_skips_types_outside_the_retry_set(mock_get_db):
    from app_cron import _record_cron_retry

    db = MagicMock()
    mock_get_db.return_value = db
    with (
        patch('app_cron.record_cron_retry') as record,
        patch('app_cron.get_queue_blocking_task', return_value=None),
    ):
        # alchemy_radio runs inline and never goes through the blocking=True
        # gate, so it can never legitimately produce a 'blocked' dispatch -
        # it stays outside the retry set.
        _record_cron_retry(db, 'alchemy_radio')

    record.assert_not_called()


@patch('app_cron.get_db')
def test_record_cron_retry_covers_plugin_cron_rows_too(mock_get_db):
    from app_cron import _record_cron_retry

    db = MagicMock()
    mock_get_db.return_value = db
    with (
        patch('app_cron.record_cron_retry') as record,
        patch('app_cron.get_queue_blocking_task', return_value=None),
    ):
        _record_cron_retry(db, 'plugin.demo.sync')

    record.assert_called_once()
    assert record.call_args[0][0] == 'plugin.demo.sync'


def test_queue_type_for_cron_task_type_passes_plugin_types_through():
    from app_cron import _queue_type_for_cron_task_type

    assert _queue_type_for_cron_task_type('analysis') == 'main_analysis'
    assert _queue_type_for_cron_task_type('clustering') == 'main_clustering'
    assert _queue_type_for_cron_task_type('sonic_fingerprint') == 'sonic_fingerprint'
    assert _queue_type_for_cron_task_type('plugin.demo.sync') == 'plugin.demo.sync'


def test_record_retry_expired_saves_under_the_queue_type_not_the_cron_name():
    from app_cron import _record_retry_expired

    entry = _pending_entry(task_type='analysis')
    with patch('app_cron.save_task_status') as save:
        _record_retry_expired('analysis', entry)

    assert save.call_args[0][1] == 'main_analysis'


def test_get_queue_blocking_task_queries_only_the_guard_task_types():
    cur = MagicMock()
    cur.fetchone.return_value = None
    db = MagicMock()
    db.cursor.return_value = cur

    with patch('database.get_db', return_value=db):
        assert database.get_queue_blocking_task() is None

    sql, params = cur.execute.call_args[0]
    assert 'task_type = ANY(%s)' in sql
    assert set(params[1]) == set(config.QUEUE_BLOCKING_TASK_TYPES)
    assert 'task_type LIKE %s' in sql
    assert params[2] == 'plugin.%'


def test_get_queue_blocking_task_admits_a_live_plugin_task():
    cur = MagicMock()
    cur.fetchone.return_value = {
        'task_id': 'plugin-run-1', 'task_type': 'plugin.demo.sync',
        'status': 'RUNNING', 'details': None,
    }
    db = MagicMock()
    db.cursor.return_value = cur

    with patch('database.get_db', return_value=db):
        blocking = database.get_queue_blocking_task()

    assert blocking['task_type'] == 'plugin.demo.sync'


def test_cron_retry_task_already_done_requires_a_success_that_started_later():
    cur = MagicMock()
    cur.fetchone.return_value = [True]
    db = MagicMock()
    db.cursor.return_value = cur

    database.cron_retry_task_already_done('analysis', None, conn=db)

    sql, params = cur.execute.call_args[0]
    assert 'status = %s' in sql
    assert 'start_time >= %s' in sql
    assert 'EXTRACT' not in sql, (
        'first_blocked_at is a UTC epoch like retry_until and cron.last_run; '
        'the comparison must stay pure epoch, never a wall-clock conversion'
    )
    assert 'ANY' not in sql, (
        'a FAIL or REVOKED completion of some other run must not clear a '
        'still-pending scheduled retry - only a genuine SUCCESS does'
    )
    assert params[1] == config.TASK_STATUS_SUCCESS


def test_cron_retry_task_already_done_resolves_a_plugin_type_to_itself():
    cur = MagicMock()
    cur.fetchone.return_value = [False]
    db = MagicMock()
    db.cursor.return_value = cur

    database.cron_retry_task_already_done('plugin.demo.sync', None, conn=db)

    sql, params = cur.execute.call_args[0]
    assert params[0] == 'plugin.demo.sync'


def test_touch_cron_last_run_writes_the_current_time():
    from app_cron import _touch_cron_last_run

    cur = MagicMock()
    db = MagicMock()
    db.cursor.return_value = cur

    _touch_cron_last_run(db, 7)

    sql, params = cur.execute.call_args[0]
    assert 'UPDATE cron SET last_run' in sql
    assert params[1] == 7
    db.commit.assert_called_once()


def test_touch_cron_last_run_swallows_a_db_failure():
    from app_cron import _touch_cron_last_run

    db = MagicMock()
    db.cursor.side_effect = RuntimeError('database is down')

    _touch_cron_last_run(db, 7)


def test_record_cron_retry_never_extends_the_deadline_of_an_existing_entry():
    cur = MagicMock()
    cur.__enter__.return_value = cur
    db = MagicMock()
    db.cursor.return_value = cur

    with patch('database.get_db', return_value=db):
        database.record_cron_retry('clustering', 12345.0, 1000.0)

    sql = cur.execute.call_args[0][0]
    assert 'ON CONFLICT (task_type) DO UPDATE SET' in sql
    update_clause = sql.split('DO UPDATE SET', 1)[1]
    assert 'retry_until' not in update_clause, (
        'an existing retry keeps its original deadline; a re-block must never '
        'extend the window'
    )


def test_record_cron_retry_stores_first_blocked_at_as_a_plain_epoch():
    cur = MagicMock()
    cur.__enter__.return_value = cur
    db = MagicMock()
    db.cursor.return_value = cur

    with patch('database.get_db', return_value=db):
        database.record_cron_retry('clustering', 12345.0, 1000.0)

    sql, params = cur.execute.call_args[0]
    insert_clause = sql.split('ON CONFLICT', 1)[0]
    assert 'first_blocked_at' in insert_clause
    assert params[2] == 1000.0, (
        'first_blocked_at carries the caller-supplied epoch verbatim'
    )
    assert 'AT TIME ZONE' not in sql, (
        'the retry window anchor is a UTC epoch like retry_until and '
        'cron.last_run; no wall-clock timestamp belongs in the comparison path'
    )


def test_queue_busy_error_body_uses_the_centralized_error_code():
    from app_helper import queue_busy_error_body
    from error.error_dictionary import ERR_TASK_IN_PROGRESS

    body = queue_busy_error_body(
        {'task_id': 'live-1', 'task_type': 'main_analysis', 'status': 'RUNNING'},
        'clustering',
    )

    assert body['error_code'] == ERR_TASK_IN_PROGRESS
    assert body['task_id'] == 'live-1'
    assert body['status'] == 'RUNNING'
    assert 'main_analysis' in body['error']
    assert 'clustering' in body['error']


def test_queue_race_error_body_also_carries_the_centralized_error_code():
    from app_helper import queue_race_error_body
    from error.error_dictionary import ERR_TASK_IN_PROGRESS

    body = queue_race_error_body('another task won the race', None)
    assert body['error_code'] == ERR_TASK_IN_PROGRESS
    assert body['task_id'] is None
    assert body['status'] is None
    assert 'won the race' in body['error']

    body = queue_race_error_body(
        'another task won the race', {'task_id': 'w-1', 'status': 'RUNNING'}
    )
    assert body['task_id'] == 'w-1'
    assert body['status'] == 'RUNNING'


def test_cron_retry_interval_is_used_unchanged_when_below_the_window():
    from app_cron import cron_retry_interval_seconds

    with (
        patch('app_cron.CRON_RETRY_MAX_MINUTES', 240),
        patch('app_cron.CRON_RETRY_INTERVAL_MINUTES', 10),
    ):
        assert cron_retry_interval_seconds() == 600


def test_cron_retry_interval_is_clamped_when_not_below_the_window():
    from app_cron import cron_retry_interval_seconds

    with (
        patch('app_cron.CRON_RETRY_MAX_MINUTES', 5),
        patch('app_cron.CRON_RETRY_INTERVAL_MINUTES', 60),
    ):
        assert cron_retry_interval_seconds() == 4 * 60
