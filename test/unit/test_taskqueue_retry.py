# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The queue's retry decision, its backoff, and the terminal row it writes.

The worker used to retry exactly one thing, a worker death, and failed a task
that raised on the spot. These tests pin the replacement: one decision function
that every outcome passes through, a backoff that keeps a deterministic failure
from spending its whole budget in milliseconds, and a terminal row that carries
the message and log the task built up, because the dashboard recap reads
details.status_message first and the task no longer writes that row itself.

Main Features:
* A retryable failure gets another attempt until the budget is spent; a
  permanent failure, a cancel and a success never do
* max_attempts of zero means never retry, which is the migration planner's opt-out
* No sentinel ever reaches the database: row_status maps all three back
* The backoff doubles from the base, is capped, jitters within its band, and a
  base of zero disables it
* The terminal row keeps the previous details, promotes the message the task
  returned, collapses the log on SUCCESS and appends on FAIL, capped
* The terminal row is committed BEFORE the history line and the collapse are
  attempted, so their rollback on failure can no longer undo the verdict
* The probe that gates the column migration names the column added last
"""

import re
from unittest.mock import MagicMock

import pytest

import config
from taskqueue import retry, sql
from taskqueue import worker as worker_mod


def _job(attempts=0, max_attempts=3):
    return {'task_id': 't', 'attempts': attempts, 'max_attempts': max_attempts}


class TestDecide:
    def test_a_retryable_failure_with_attempts_left_is_retried(self):
        assert retry.decide(_job(0, 3), retry.FAIL_RETRYABLE) == retry.RETRY
        assert retry.decide(_job(1, 3), retry.FAIL_RETRYABLE) == retry.RETRY

    def test_max_attempts_counts_restarts_so_the_fourth_ending_is_final(self):
        assert retry.decide(_job(2, 3), retry.FAIL_RETRYABLE) == retry.RETRY, (
            'three restarts are allowed, exactly as the worker-death reclaim has '
            'always read the same number; the third bad ending still earns one'
        )
        assert retry.decide(_job(3, 3), retry.FAIL_RETRYABLE) == retry.FINISH

    @pytest.mark.parametrize('outcome', [
        retry.FAIL_PERMANENT, retry.REVOKED_BY_TASK, config.TASK_STATUS_SUCCESS, None,
    ])
    def test_nothing_but_a_retryable_failure_is_ever_retried(self, outcome):
        assert retry.decide(_job(0, 3), outcome) == retry.FINISH

    def test_a_zero_budget_never_retries(self):
        assert retry.decide(_job(0, 0), retry.FAIL_RETRYABLE) == retry.FINISH

    def test_worker_deaths_and_errors_spend_one_shared_budget(self):
        assert retry.decide(_job(3, 3), retry.FAIL_RETRYABLE) == retry.FINISH, (
            'attempts counts worker deaths AND application errors on ONE counter, '
            'as every comparable queue does; the third bad ending of any kind is '
            'the last, and nobody may split the budget back in two by accident'
        )


class TestRowStatus:
    def test_both_failure_sentinels_become_fail(self):
        assert retry.row_status(retry.FAIL_RETRYABLE) == config.TASK_STATUS_FAIL
        assert retry.row_status(retry.FAIL_PERMANENT) == config.TASK_STATUS_FAIL

    def test_the_cancel_sentinel_becomes_revoked(self):
        assert retry.row_status(retry.REVOKED_BY_TASK) == config.TASK_STATUS_REVOKED

    def test_success_passes_through(self):
        assert retry.row_status(config.TASK_STATUS_SUCCESS) == config.TASK_STATUS_SUCCESS

    def test_no_sentinel_is_a_status_the_database_knows(self):
        for sentinel in (retry.FAIL_RETRYABLE, retry.FAIL_PERMANENT, retry.REVOKED_BY_TASK):
            assert sentinel not in config.TASK_STATUS_TERMINAL
            assert sentinel not in config.TASK_STATUS_LIVE


class _Rng:
    def __init__(self, value):
        self.value = value

    def uniform(self, _low, _high):
        return self.value


class TestBackoff:
    def test_it_doubles_from_the_base_and_is_capped(self, monkeypatch):
        monkeypatch.setattr(config, 'QUEUE_RETRY_BASE_SECONDS', 30.0)
        monkeypatch.setattr(config, 'QUEUE_RETRY_MAX_SECONDS', 100.0)
        flat = _Rng(0.0)

        waits = [retry.backoff_seconds(attempt, rng=flat) for attempt in (1, 2, 3, 4)]

        assert waits == [30.0, 60.0, 100.0, 100.0]

    def test_the_jitter_stays_inside_its_band(self, monkeypatch):
        monkeypatch.setattr(config, 'QUEUE_RETRY_BASE_SECONDS', 30.0)
        monkeypatch.setattr(config, 'QUEUE_RETRY_MAX_SECONDS', 600.0)

        low = retry.backoff_seconds(1, rng=_Rng(-retry.JITTER_FRACTION))
        high = retry.backoff_seconds(1, rng=_Rng(retry.JITTER_FRACTION))

        assert low == pytest.approx(27.0)
        assert high == pytest.approx(33.0)

    def test_a_zero_base_disables_the_wait(self, monkeypatch):
        monkeypatch.setattr(config, 'QUEUE_RETRY_BASE_SECONDS', 0.0)

        assert retry.backoff_seconds(3) == 0.0

    def test_the_schedule_is_monotonic_under_any_jitter(self, monkeypatch):
        monkeypatch.setattr(config, 'QUEUE_RETRY_BASE_SECONDS', 30.0)
        monkeypatch.setattr(config, 'QUEUE_RETRY_MAX_SECONDS', 600.0)
        worst_early = retry.backoff_seconds(1, rng=_Rng(retry.JITTER_FRACTION))
        best_late = retry.backoff_seconds(2, rng=_Rng(-retry.JITTER_FRACTION))

        assert best_late > worst_early


class TestTheTerminalRowTheQueueWrites:
    def test_the_message_the_task_returned_is_what_the_dashboard_reads(self):
        details = worker_mod._terminal_details(
            config.TASK_STATUS_SUCCESS, None,
            {'message': 'Alignment complete: 12/40 pending tracks matched on Plex.'},
        )

        assert details['status_message'] == (
            'Alignment complete: 12/40 pending tracks matched on Plex.'
        ), (
            'app.py reads details.status_message before anything else, and the task '
            'no longer writes that row itself, so a generic message here turns every '
            'recap on the dashboard into "Task completed successfully."'
        )
        assert details['message'] == details['status_message']

    def test_a_generic_message_only_when_the_task_supplied_none(self):
        details = worker_mod._terminal_details(config.TASK_STATUS_SUCCESS, None, {'n': 1})

        assert details['message'] == 'Task completed successfully.'

    def test_the_previous_details_survive_and_the_summary_wins(self):
        details = worker_mod._terminal_details(
            config.TASK_STATUS_SUCCESS, None, {'tracks_analyzed': 7, 'status': 'SUCCESS'},
            previous={'album_name': 'Kind of Blue', 'tracks_analyzed': 0},
        )

        assert details['album_name'] == 'Kind of Blue'
        assert details['tracks_analyzed'] == 7
        assert 'status' not in details, 'the status column is the truth, not a details key'
        assert details['final_summary_details'] == {'tracks_analyzed': 7, 'status': 'SUCCESS'}

    def test_success_collapses_the_log_to_one_line(self):
        details = worker_mod._terminal_details(
            config.TASK_STATUS_SUCCESS, None, {'message': 'All done.'},
            previous={'log': ['[t] one', '[t] two']},
        )

        assert details['log'] == ['Task completed successfully. Final status: All done.']

    def test_a_failure_appends_to_the_log_instead_of_collapsing(self):
        details = worker_mod._terminal_details(
            config.TASK_STATUS_FAIL, 'jellyfin answered 502', None,
            previous={'log': ['[t] one', '[t] two']},
        )

        assert len(details['log']) == 3
        assert details['log'][:2] == ['[t] one', '[t] two']
        assert details['log'][-1].endswith('jellyfin answered 502')
        assert details['error'] == 'jellyfin answered 502'

    def test_the_log_never_exceeds_the_shared_cap(self):
        from database import MAX_LOG_ENTRIES_STORED

        details = worker_mod._terminal_details(
            config.TASK_STATUS_FAIL, 'boom', None,
            previous={'log': [f'[t] {i}' for i in range(MAX_LOG_ENTRIES_STORED + 5)]},
        )

        assert len(details['log']) == MAX_LOG_ENTRIES_STORED
        assert details['log'][-1].endswith('boom')

    def test_a_structured_error_record_the_task_wrote_survives_the_summary(self):
        recorded = {'error_code': 2005, 'error_message': 'every server failed'}
        details = worker_mod._terminal_details(
            config.TASK_STATUS_FAIL, 'RuntimeError: every server failed', None,
            previous={'error': recorded},
        )

        assert details['error'] == recorded, (
            'the dashboard reads details.error.error_code; a task records that '
            'structured error on its last progress write and then raises, and the '
            'queue must not flatten it into the exception text'
        )

    def test_a_success_drops_the_error_a_failed_attempt_left_on_the_row(self):
        details = worker_mod._terminal_details(
            config.TASK_STATUS_SUCCESS, None, {'message': 'Done on the retry.'},
            previous={
                'error': {'error_code': 2005, 'error_message': 'attempt 1 failed'},
                'album_name': 'Kind of Blue',
            },
        )

        assert 'error' not in details, (
            'requeue_or_fail keeps the row details across a retry; a retry that '
            'returns without a progress write of its own would otherwise show a '
            'SUCCESS recap still carrying the previous attempt\'s error record'
        )
        assert details['album_name'] == 'Kind of Blue'

    def test_a_summary_value_json_cannot_encode_still_finishes_the_row(self):
        import datetime
        from taskqueue import sql

        cur = MagicMock()
        cur.fetchone.return_value = ('t1',)

        assert sql.finish_task(
            cur, 't1', config.TASK_STATUS_SUCCESS,
            {'when': datetime.datetime(2026, 9, 5, 12, 0)}, 0.0, worker_id='w1',
        ) == 't1'
        assert '2026-09-05' in cur.execute.call_args.args[1][1], (
            'a terminal row that cannot be written leaves the task RUNNING until '
            'reclaim; a value JSON cannot encode is stored as its text instead'
        )

    def test_a_revoked_row_says_so(self):
        details = worker_mod._terminal_details(
            config.TASK_STATUS_REVOKED, 'task t was revoked', None,
        )

        assert details['message'] == 'task t was revoked'


class TestTheColumnMigrationProbe:
    def test_the_probe_names_the_column_added_last(self):
        added = re.findall(r'ADD COLUMN IF NOT EXISTS\s+(\w+)', sql._ADD_COLUMNS)

        assert added, 'the ALTER block lists no columns'
        assert added[-1] == sql.NEWEST_COLUMN, (
            'ensure_schema runs the whole ALTER block only when the probed column is '
            'absent. Every existing install already has the earlier columns, so a '
            'column appended here without moving the probe never arrives, and the '
            'first claim after the upgrade raises UndefinedColumn'
        )
        assert f"column_name = '{sql.NEWEST_COLUMN}'" in sql._PROBE_NEWEST_COLUMN

    def test_next_run_at_is_the_newest_column(self):
        assert sql.NEWEST_COLUMN == 'next_run_at'


class _RowCursor:
    def __init__(self, rows):
        self._rows = rows
        self._row = None

    def execute(self, statement, params=None):
        self._row = self._rows.get(params[0] if params else None)

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestTheQueueRecordsHistoryAndCollapsesOnATerminalWrite:
    def _worker(self):
        import threading

        instance = worker_mod.Worker.__new__(worker_mod.Worker)
        instance.identity = 'audiomuse-worker-high-hostA-11'
        instance._claim_txn = threading.Lock()
        return instance

    def test_a_root_task_gets_a_history_line_and_the_table_collapses(self, monkeypatch):
        import database

        instance = self._worker()
        conn = MagicMock()
        conn.cursor.side_effect = lambda: _RowCursor({'task-1': (10.0, 25.0)})
        instance._conn = conn
        recorded = {}
        collapsed = {}
        monkeypatch.setattr(
            database, 'record_task_history',
            lambda *a, **k: recorded.update({'args': a, 'kwargs': k}),
        )
        monkeypatch.setattr(
            database, 'collapse_finished_task',
            lambda *a: collapsed.update({'args': a}),
        )

        instance._record_and_collapse(
            'task-1',
            {'task_type': 'main_analysis', 'parent_task_id': None},
            config.TASK_STATUS_SUCCESS, {'message': 'done'},
        )

        assert recorded['args'][:3] == ('task-1', 'main_analysis', config.TASK_STATUS_SUCCESS)
        assert recorded['kwargs']['duration_seconds'] == 15.0
        assert recorded['kwargs']['conn'] is conn, (
            'tasks used to write their terminal row through save_task_status, which '
            'recorded the history line and collapsed the table for them; now that the '
            'queue writes every terminal row it has to do both, on its own connection'
        )
        assert collapsed['args'][:2] == (conn, 'task-1')

    def test_a_child_gets_neither(self, monkeypatch):
        import database

        instance = self._worker()
        instance._conn = MagicMock()
        monkeypatch.setattr(
            database, 'record_task_history',
            lambda *a, **k: pytest.fail('children are reaped by their parent'),
        )
        monkeypatch.setattr(
            database, 'collapse_finished_task',
            lambda *a: pytest.fail('collapsing under a live parent hangs the fan-out'),
        )

        instance._record_and_collapse(
            'child-1', {'task_type': 'album_analysis', 'parent_task_id': 'root-1'},
            config.TASK_STATUS_SUCCESS, {},
        )

    def test_the_terminal_row_is_committed_before_the_history_line_is_attempted(
        self, monkeypatch
    ):
        import database

        instance = self._worker()
        order = []
        conn = MagicMock()
        conn.closed = 0
        conn.commit.side_effect = lambda: order.append('commit')
        conn.rollback.side_effect = lambda: order.append('rollback')
        instance._conn = conn
        monkeypatch.setattr(worker_mod.sql, 'current_row', lambda cur, task_id: {
            'status': config.TASK_STATUS_RUNNING, 'task_type': 'cleaning',
            'parent_task_id': None, 'worker_id': instance.identity,
        })
        monkeypatch.setattr(worker_mod.sql, 'current_details', lambda cur, task_id: {})
        monkeypatch.setattr(
            worker_mod.sql, 'finish_task',
            lambda *a, **k: order.append('finish') or config.TASK_STATUS_FAIL,
        )

        def history_that_rolls_back(*args, **kwargs):
            order.append('history')
            kwargs['conn'].rollback()

        monkeypatch.setattr(database, 'record_task_history', history_that_rolls_back)
        monkeypatch.setattr(database, 'collapse_finished_task', lambda *a: 0)

        instance.finalize({'task_id': 'task-1'}, config.TASK_STATUS_FAIL, 'boom')

        assert order[:2] == ['finish', 'commit'], (
            'record_task_history rolls back on failure; while it shared the terminal '
            "write's transaction that rollback undid the verdict, the row stayed "
            'RUNNING under a live idle worker, and every start answered 409'
        )
        assert order.index('history') > order.index('commit')

    def test_a_failure_inside_the_bookkeeping_is_rolled_back_after_the_recap_is_durable(
        self, monkeypatch
    ):
        import database

        instance = self._worker()
        instance._conn = MagicMock()
        rolled_back = []
        monkeypatch.setattr(instance, '_safe_rollback', lambda: rolled_back.append(True))

        def history_is_gone(*args, **kwargs):
            raise RuntimeError('history table is gone')

        monkeypatch.setattr(database, 'record_task_history', history_is_gone)
        monkeypatch.setattr(database, 'collapse_finished_task', lambda *a: 0)

        instance._record_and_collapse(
            'task-1', {'task_type': 'cleaning', 'parent_task_id': None},
            config.TASK_STATUS_FAIL, {},
        )

        assert rolled_back == [True]


class TestASharedPayloadThatCannotComeBackIsPermanent:
    def test_a_missing_or_mismatched_body_fails_without_a_retry(self, monkeypatch):
        import threading

        instance = worker_mod.Worker.__new__(worker_mod.Worker)
        instance.identity = 'audiomuse-worker-default-hostA-11'
        instance._claim_txn = threading.Lock()
        instance._fork_jobs = False

        def gone(kwargs):
            raise sql.SharedPayloadUnavailable('the shared payload for p is gone')

        monkeypatch.setattr(instance, 'hydrate_shared', gone)

        outcome, summary, result = instance._execute(
            {'task_id': 'kid-1', 'kwargs': {'__audiomuse_shared__': {}}}
        )

        assert outcome == retry.FAIL_PERMANENT, (
            'a body whose token no longer matches its owner cannot come back, so '
            'spending the restart budget on it only delays the FAIL the parent is '
            'waiting for'
        )
        assert 'gone' in summary
        assert result is None


class TestNoProgressReportRegressesATerminalRow:
    def test_the_status_write_guard_names_every_terminal_status(self):
        import inspect
        import database

        source = inspect.getsource(database.save_task_status)

        assert "<> ALL(%s)" in source
        assert "IS DISTINCT FROM" not in source, (
            'the guard used to spare only REVOKED, so a progress write from a stale '
            'duplicate worker could turn a queue-written SUCCESS back into RUNNING; '
            "a terminal row is the queue's verdict and nothing may reopen it"
        )


class TestTheDashboardReadsTheSameFieldsFromAQueueWrittenRow:
    def _sanitized(self, status, error, result, previous, task_type='cleaning'):
        from app_helper import sanitize_task_details

        row = worker_mod._terminal_details(status, error, result, previous=previous)
        return sanitize_task_details(dict(row), status, task_type)

    def test_a_success_recap_shows_the_message_and_the_summary_the_task_returned(self):
        details = self._sanitized(
            config.TASK_STATUS_SUCCESS, None,
            {'status': 'SUCCESS', 'message': 'Cleanup complete: 3 stale mappings unbound.',
             'deleted_count': 0, 'orphaned_albums': []},
            previous={'log': ['[t] Fetching the track list from One...']},
        )

        assert details['status_message'] == 'Cleanup complete: 3 stale mappings unbound.'
        assert details['message'] == details['status_message']
        assert details['final_summary_details']['deleted_count'] == 0, (
            'templates/cleaning.html reads final_summary_details for its results table'
        )
        assert details['deleted_count'] == 0
        assert details['log'] == [
            'Task completed successfully. Final status: Cleanup complete: 3 stale mappings unbound.'
        ]

    def test_a_failure_keeps_the_structured_error_the_task_recorded(self):
        recorded = {
            'error_code': 6002, 'error_class': 'Cleaning Error',
            'error_message': 'Database cleaning failed.',
        }
        details = self._sanitized(
            config.TASK_STATUS_FAIL, 'RuntimeError: server One could not be read', None,
            previous={'error': recorded, 'log': ['[t] one'],
                      'final_summary_details': {'failed_servers': ['One']}},
        )

        assert details['error'] == recorded, (
            'static/error_display.js shows the error block only for a structured '
            'record with an error_code; the queue must not flatten it'
        )
        assert details['final_summary_details'] == {'failed_servers': ['One']}
        assert details['status_message'] == 'RuntimeError: server One could not be read'
        assert details['log'][-1].endswith('RuntimeError: server One could not be read')

    def test_a_failure_with_no_recorded_error_still_names_the_reason(self):
        details = self._sanitized(
            config.TASK_STATUS_FAIL, 'jellyfin answered 502', None, previous={},
        )

        assert details['status_message'] == 'jellyfin answered 502'
        assert details['error']['error_code'], (
            'sanitize_task_details builds a generic record when the task recorded '
            'none, exactly as it did for the sweep and plugin tasks before'
        )
