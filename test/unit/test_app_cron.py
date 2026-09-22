# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Cron scheduler dispatch and the online tasks it runs inline in Flask.

Exercises run_due_cron_jobs and the task behind the sonic-fingerprint row.
Batch rows enqueue so a slow media server cannot swallow a scheduling window; the
alchemy radio, the sonic fingerprint and the album of the week run inline in
Flask, the only process holding the similarity index.

Main Features:
* The sonic-fingerprint row runs its task inline rather than enqueueing it
* The alchemy-radio row runs inline in Flask, never on a worker, and records
  STARTED then SUCCESS (or FAILURE, without leaving the row STARTED forever)
* The terminal row MERGES into the details the run reported, a failure carries
  the classified error record the worker path writes, and an executed inline
  run answers 'ran', which clears a retry and is never retried
* A tick evaluates every minute since the previous tick: an inline run that
  spans a minute cannot swallow a batch schedule due in it, and an every-minute
  row fires once per tick, not once per missed minute; a gap past the catch-up
  is a clock jump that evaluates only the current minute
* Online rows due in one tick start 10 seconds apart, after every batch row of
  that tick was dispatched; a slow run lets the next start at once
* Empty fingerprint results skip both playlist upsert and the legacy fallback
* Non-empty results upsert under the constant cron playlist name via item_ids
* NotImplementedError from the backend falls back to a timestamped legacy playlist
* A live main task blocks a cron analysis/clustering start, as the manual endpoints do
* A failed enqueue leaves no row at all, never a PENDING row that would 409 every later start
* Every task type the Scheduled Tasks page can save reaches a real dispatch
  branch, so a row that fires for ever and runs nothing cannot ship
* The scheduled playlist tasks own their track ids and nothing else: the loop,
  the reporter, the heartbeat and the fallbacks are one shared scaffold, and a
  task that grows its own copy of any of them fails here
* The Scheduled Tasks page warns when two enabled rows (built-in or plugin) can
  start in the same minute, and the warning never stops or delays a save
"""

import inspect
import json
import logging
import pathlib
import re
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import psycopg2
import pytest
from psycopg2.extensions import TRANSACTION_STATUS_INERROR

import config
import task_types
import taskqueue
from error import error_manager
from error.error_dictionary import ERR_SEARCH_FAILED
from taskqueue import TaskCancelled


@pytest.fixture(autouse=True)
def _fresh_cron_clock_and_no_row_read():
    import app_cron

    app_cron._cron_clock.update(last_minute=None, last_monotonic=None)
    app_cron._cron_saved_at.clear()
    with patch('app_cron.get_task_info_from_db', return_value=None):
        yield
    app_cron._cron_clock.update(last_minute=None, last_monotonic=None)
    app_cron._cron_saved_at.clear()


def _open_db():
    db = MagicMock()
    db.closed = 0
    return db


def _make_cron_row(task_type='sonic_fingerprint'):
    return {
        'id': 1,
        'name': 'Sonic Fingerprint',
        'task_type': task_type,
        'cron_expr': '* * * * *',
        'enabled': True,
        'last_run': 0,
    }


def _setup_db_mock(task_type='sonic_fingerprint'):
    cur = MagicMock()
    cur.fetchall.return_value = [_make_cron_row(task_type)]
    cur.fetchone.return_value = None
    cur.__enter__.return_value = cur
    cur.rowcount = 1
    db = MagicMock()
    db.cursor.return_value = cur
    return db, cur


def _run_fingerprint_task():
    from tasks.sonic_fingerprint_manager import run_sonic_fingerprint_task

    with patch('tasks.mediaserver.registry.servers_for_scope', return_value=[None]):
        return run_sonic_fingerprint_task(server_scope='all')


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_sonic_fingerprint_row_runs_inline_instead_of_enqueueing(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs

    db, _cur = _setup_db_mock()
    mock_get_db.return_value = db

    with (
        patch('app_cron.save_task_status'),
        patch('app_cron.get_queue_blocking_task', return_value=None),
        patch('app_cron.clean_up_previous_main_tasks'),
        patch('app_cron.taskqueue.enqueue') as enqueue,
        patch(
            'tasks.sonic_fingerprint_manager.run_sonic_fingerprint_task',
            return_value={'message': 'done'},
        ) as run,
    ):
        run_due_cron_jobs()

    enqueue.assert_not_called()
    run.assert_called_once()
    assert run.call_args[1]['server_scope'] == 'all'
    assert run.call_args[1]['inline_task_id']


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_a_minute_already_claimed_by_another_web_process_fires_nothing(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs

    cur = MagicMock()
    cur.fetchall.return_value = [
        _make_cron_row('sonic_fingerprint'),
        _make_cron_row('alchemy_radio'),
    ]
    cur.fetchone.return_value = None
    cur.__enter__.return_value = cur
    cur.rowcount = 0
    db = MagicMock()
    db.cursor.return_value = cur
    mock_get_db.return_value = db

    with (
        patch('app_cron.save_task_status') as save,
        patch('app_cron.taskqueue.enqueue') as enqueue,
        patch('tasks.radio_manager.run_radio_playlists') as run,
    ):
        run_due_cron_jobs()

    enqueue.assert_not_called()
    run.assert_not_called()
    save.assert_not_called()


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_the_minute_claim_writes_last_run_only_when_it_is_older_than_this_minute(
    mock_get_db, _matches
):
    from app_cron import run_due_cron_jobs

    db, cur = _setup_db_mock()
    mock_get_db.return_value = db

    with (
        patch('app_cron.save_task_status'),
        patch('app_cron.get_queue_blocking_task', return_value=None),
        patch('app_cron.clean_up_previous_main_tasks'),
        patch('app_cron.taskqueue.enqueue'),
    ):
        run_due_cron_jobs()

    updates = [c for c in cur.execute.call_args_list if c[0][0].startswith('UPDATE cron')]
    assert len(updates) == 1
    sql, params = updates[0][0]
    assert 'last_run IS NULL OR last_run < %s' in sql
    assert 'enabled = true AND cron_expr = %s' in sql, (
        'the SELECT is from the tick start: a row disabled or edited on the page '
        'while an earlier online run held the tick must not be claimed'
    )
    minute_start, row_id, cron_expr, guard = params
    assert row_id == 1
    assert cron_expr == '* * * * *'
    assert guard == minute_start
    assert minute_start % 60 == 0


def test_sonic_fingerprint_task_skips_on_empty_results():
    with (
        patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=[]) as gen,
        patch('tasks.mediaserver.create_or_replace_playlist') as upsert,
        patch('tasks.ivf_manager.create_playlist_from_ids') as legacy,
    ):
        summary = _run_fingerprint_task()

    gen.assert_called_once()
    upsert.assert_not_called()
    legacy.assert_not_called()
    assert summary['playlists_created'] == 0


def test_dequeued_sonic_task_with_wiped_claim_does_no_work():
    from tasks.sonic_fingerprint_manager import run_sonic_fingerprint_task

    with (
        patch.object(taskqueue, 'current_task_id', return_value='sonic-cancelled'),
        patch('tasks.task_run._read_task_statuses', return_value={}),
        patch('tasks.task_run.save_task_status') as save,
        patch('tasks.mediaserver.registry.servers_for_scope') as servers,
    ):
        with pytest.raises(TaskCancelled):
            run_sonic_fingerprint_task(server_scope='all')

    save.assert_not_called()
    servers.assert_not_called()


def test_sonic_fingerprint_task_calls_upsert_with_constant_name():
    from config import SONIC_FINGERPRINT_CRON_PLAYLIST_NAME

    fp = [{'item_id': 'a'}, {'item_id': 'b'}, {'item_id': 'c'}]

    with (
        patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=fp),
        patch(
            'tasks.mediaserver.create_or_replace_playlist', return_value={'Id': 'pl-x'}
        ) as upsert,
        patch('tasks.ivf_manager.create_playlist_from_ids') as legacy,
    ):
        summary = _run_fingerprint_task()

    upsert.assert_called_once_with(SONIC_FINGERPRINT_CRON_PLAYLIST_NAME, ['a', 'b', 'c'])
    legacy.assert_not_called()
    assert summary['playlists_created'] == 1


def test_sonic_fingerprint_task_falls_back_for_unsupported_backend():
    fp = [{'item_id': 'a'}]

    with (
        patch('tasks.sonic_fingerprint_manager.generate_sonic_fingerprint', return_value=fp),
        patch('tasks.mediaserver.create_or_replace_playlist', side_effect=NotImplementedError),
        patch('tasks.ivf_manager.create_playlist_from_ids', return_value='legacy-id') as legacy,
    ):
        _run_fingerprint_task()

    legacy.assert_called_once()
    legacy_name = legacy.call_args[0][0]
    assert legacy_name.startswith('Sonic Fingerprint (Cron ')
    assert legacy.call_args[0][1] == ['a']


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_alchemy_radio_row_runs_inline_in_flask_never_on_a_worker(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs
    from config import TASK_STATUS_STARTED, TASK_STATUS_SUCCESS

    db, _cur = _setup_db_mock(task_type='alchemy_radio')
    mock_get_db.return_value = db

    summary = {'playlists_created': 2, 'failed': []}
    with (
        patch('app_cron.save_task_status') as save,
        patch('app_cron.taskqueue.enqueue') as enqueue,
        patch('tasks.radio_manager.run_radio_playlists', return_value=summary) as run,
    ):
        run_due_cron_jobs()

    enqueue.assert_not_called()
    run.assert_called_once()
    assert run.call_args.kwargs['server_scope'] == 'all'
    assert callable(run.call_args.kwargs['report'])
    statuses = [c[0][2] for c in save.call_args_list]
    assert statuses == [TASK_STATUS_STARTED, TASK_STATUS_SUCCESS]
    final = save.call_args_list[-1][1]['details']
    assert final['playlists_created'] == 2
    assert final['final_summary_details'] == summary


@pytest.mark.parametrize('failure', [
    RuntimeError('radio backend refused the request: ' + 'x' * 2000),
    RuntimeError(),
    RuntimeError('   '),
], ids=['long-message', 'no-message', 'blank-message'])
@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_failed_inline_radio_run_records_the_same_error_summary_a_worker_failure_does(
    mock_get_db, _matches, failure
):
    from app_cron import run_due_cron_jobs

    db, _cur = _setup_db_mock(task_type='alchemy_radio')
    mock_get_db.return_value = db

    with (
        patch('app_cron.save_task_status') as save,
        patch('tasks.radio_manager.run_radio_playlists', side_effect=failure),
    ):
        run_due_cron_jobs()

    last_call = save.call_args_list[-1]
    assert last_call[0][2] == config.TASK_STATUS_FAILURE
    error = last_call[1]['details']['error']
    assert error['error_code'] == ERR_SEARCH_FAILED, (
        'the failure carries the same classified record a worker failure does'
    )
    assert error == error_manager.build(ERR_SEARCH_FAILED, taskqueue.error_summary(failure)), (
        'the inline record carries the same cut summary the worker records'
    )
    assert 'x' * 600 not in error['error_message']
    if not str(failure).strip():
        assert error['error_message'].endswith('RuntimeError'), (
            'the worker summary names the exception class when it has no text; '
            'recording str(exc) instead would leave the reason empty'
        )
    assert 'container logs' in last_call[1]['details']['status_message']
    db.rollback.assert_called_once()


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_the_inline_radio_row_heartbeats_progress_into_its_own_task_row(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs
    from config import TASK_STATUS_RUNNING

    db, _cur = _setup_db_mock(task_type='alchemy_radio')
    mock_get_db.return_value = db

    def _fake_run(server_scope='all', report=None):
        report('Radio 1 of 2', 50.0)
        return {'playlists_created': 1, 'failed': []}

    with (
        patch('app_cron.save_task_status') as save,
        patch('tasks.radio_manager.run_radio_playlists', side_effect=_fake_run),
    ):
        run_due_cron_jobs()

    running_calls = [c for c in save.call_args_list if c[0][2] == TASK_STATUS_RUNNING]
    assert running_calls
    heartbeat = running_calls[-1]
    assert heartbeat[1]['progress'] == 50
    assert heartbeat[1]['details']['status_message'] == 'Radio 1 of 2'


@patch('app_cron.get_db')
def test_an_inline_run_interrupted_by_a_restart_is_failed_when_the_cron_thread_starts(
    mock_get_db,
):
    from app_cron import reap_interrupted_inline_runs
    from config import TASK_STATUS_FAILURE, TASK_STATUS_SUCCESS, TASK_STATUS_REVOKED

    cur = MagicMock()
    cur.fetchall.return_value = [{'task_id': 'radio-1', 'task_type': 'alchemy_radio'}]
    db = MagicMock()
    db.cursor.return_value = cur
    mock_get_db.return_value = db

    with patch('app_cron.save_task_status') as save:
        assert reap_interrupted_inline_runs() == 1

    select_sql, select_params = cur.execute.call_args[0]
    assert 'func IS NULL' in select_sql, (
        'a queued row a worker owns (an old install mid-upgrade) must never be '
        'failed by the Flask startup reap'
    )
    assert set(select_params[0]) == set(task_types.INLINE_FLASK_TASK_TYPES)
    assert 'alchemy_radio' in select_params[0]
    assert {'sonic_fingerprint', 'album_of_the_week'} <= set(select_params[0]), (
        'both playlist crons run inline now, so a restart must fail their rows too'
    )
    assert set(select_params[1:]) == {
        TASK_STATUS_SUCCESS, TASK_STATUS_FAILURE, TASK_STATUS_REVOKED,
    }
    assert save.call_args[0][:3] == ('radio-1', 'alchemy_radio', TASK_STATUS_FAILURE)
    assert 'restart' in save.call_args[1]['details']['error']


@patch('app_cron.get_db')
def test_startup_reap_writes_nothing_when_no_inline_run_was_interrupted(mock_get_db):
    from app_cron import reap_interrupted_inline_runs

    cur = MagicMock()
    cur.fetchall.return_value = []
    db = MagicMock()
    db.cursor.return_value = cur
    mock_get_db.return_value = db

    with patch('app_cron.save_task_status') as save:
        assert reap_interrupted_inline_runs() == 0

    save.assert_not_called()


def test_a_radio_row_can_never_gate_a_start_because_only_flask_can_finish_it():
    import database

    cur = MagicMock()
    cur.fetchone.return_value = None
    db = MagicMock()
    db.cursor.return_value = cur

    with patch('database.get_db', return_value=db):
        assert database.get_active_main_task() is None

    params = cur.execute.call_args[0][1]
    excluded = next(param for param in params if isinstance(param, list))
    assert 'alchemy_radio' in excluded
    assert 'alchemy_radio' in database.SELF_MANAGED_TASK_TYPES


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_cron_analysis_does_not_start_a_second_run_while_one_is_live(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs

    db, _cur = _setup_db_mock(task_type='analysis')
    mock_get_db.return_value = db

    active = {'task_id': 'live-1', 'task_type': 'main_analysis', 'status': 'RUNNING'}
    with (
        patch('app_cron.get_queue_blocking_task', return_value=active),
        patch('app_cron.record_cron_retry') as retry,
        patch('app_cron.save_task_status') as save,
        patch('app_cron.taskqueue.enqueue') as enqueue,
    ):
        run_due_cron_jobs()

    enqueue.assert_not_called()
    save.assert_not_called()
    retry.assert_called_once()
    assert retry.call_args[0][0] == 'analysis'


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_a_failed_queue_write_leaves_no_row_behind(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs

    db, _cur = _setup_db_mock(task_type='analysis')
    mock_get_db.return_value = db

    with (
        patch('app_cron.get_queue_blocking_task', return_value=None),
        patch('app_cron.save_task_status') as save,
        patch(
            'app_cron.taskqueue.enqueue', side_effect=RuntimeError("database is down")
        ) as enqueue,
        patch('app_cron.clean_up_previous_main_tasks'),
    ):
        run_due_cron_jobs()

    enqueue.assert_called_once()
    assert enqueue.call_args[0][0] == 'tasks.analysis.run_analysis_task'
    assert enqueue.call_args[1]['task_type'] == 'main_analysis'
    assert not save.call_args_list, 'a failed queue write must leave no task row'
    db.rollback.assert_not_called()
    db.commit.assert_called_once()


@patch('app_cron.cron_matches_now', return_value=True)
@patch('app_cron.get_db')
def test_plugin_branch_always_runs_against_all_servers(mock_get_db, _matches):
    from app_cron import run_due_cron_jobs

    row = _make_cron_row(task_type='plugin.demo.sync')
    row['options'] = {'server_scope': 'default'}
    cur = MagicMock()
    cur.fetchall.return_value = [row]
    cur.fetchone.return_value = None
    cur.rowcount = 1
    db = MagicMock()
    db.cursor.return_value = cur
    mock_get_db.return_value = db

    plugin_manager = MagicMock()
    plugin_manager.get_cron_task.return_value = {
        'dotted': 'audiomuse_plugins.demo.tasks.sync', 'queue': 'default',
    }
    fake_plugin_module = MagicMock()
    fake_plugin_module.plugin_manager = plugin_manager

    with patch.dict('sys.modules', {'plugin.manager': fake_plugin_module}), \
            patch('app_cron.save_task_status'), \
            patch('app_cron.taskqueue.enqueue') as queue:
        run_due_cron_jobs()

    assert queue.called
    kwargs = queue.call_args.kwargs
    assert kwargs['args'] == ('audiomuse_plugins.demo.tasks.sync',)
    assert kwargs['kwargs'] == {'server_scope': 'all'}, (
        'the shared cancel check enforces the live claim for every task now; '
        'the old task_claim_required flag is no longer written into a payload'
    )


def _task_types_the_page_can_schedule():

    page = (pathlib.Path(__file__).resolve().parents[2] / 'templates' / 'cron.html').read_text(
        encoding='utf-8'
    )
    return sorted(set(re.findall(r"task_type:'([a-z_]+)'", page)))


@pytest.mark.parametrize('task_type', _task_types_the_page_can_schedule())
def test_every_row_the_page_can_save_reaches_a_real_branch(task_type):
    from app_cron import _dispatch_cron_row

    db = _open_db()
    summary = {'playlists_created': 0, 'failed': []}
    with (
        patch('app_cron.main_task_start_lock'),
        patch('app_cron.get_queue_blocking_task', return_value=None),
        patch('app_cron.clean_up_previous_main_tasks'),
        patch('app_cron.save_task_status'),
        patch('app_cron.taskqueue.enqueue') as enqueue,
        patch('tasks.radio_manager.run_radio_playlists', return_value=summary) as radio,
        patch(
            'tasks.sonic_fingerprint_manager.run_sonic_fingerprint_task',
            return_value=summary,
        ) as fingerprint,
        patch(
            'tasks.album_creation_manager.run_album_of_the_week_task',
            return_value=summary,
        ) as album,
    ):
        outcome = _dispatch_cron_row(db, {'task_type': task_type})
        queued = [call[0][0] for call in enqueue.call_args_list]
        ran_inline = radio.called or fingerprint.called or album.called

    inline = task_type in task_types.INLINE_FLASK_TASK_TYPES
    assert outcome == ('ran' if inline else 'enqueued'), (
        f'a {task_type} row can be saved from the Scheduled Tasks page and its '
        f'tick answered {outcome}; a row that dispatches to nothing fires for '
        'ever and never runs'
    )
    if inline:
        assert ran_inline and not queued
    else:
        assert not ran_inline and len(queued) == 1 and queued[0].startswith('tasks.')


class TestTheScheduledPlaylistTasksShareOneScaffold:
    def _task_sources(self):
        from tasks.album_creation_manager import run_album_of_the_week_task
        from tasks.sonic_fingerprint_manager import run_sonic_fingerprint_task

        return {
            'sonic_fingerprint': inspect.getsource(run_sonic_fingerprint_task),
            'album_of_the_week': inspect.getsource(run_album_of_the_week_task),
        }

    @pytest.mark.parametrize('task_type', ['sonic_fingerprint', 'album_of_the_week'])
    def test_the_task_body_is_the_track_ids_and_nothing_else(self, task_type):
        source = self._task_sources()[task_type]

        assert 'run_playlist_task_per_server' in source
        for copied in (
            'create_or_replace_playlist', 'row_heartbeat', 'cancel_guard',
            'make_task_reporter', 'for_each_server_in_scope', 'app_context',
        ):
            assert copied not in source, (
                f'{task_type} carries its own {copied} again. Both tasks used to '
                'hold a line-for-line copy of the same ninety-line scaffold, '
                'which is how the older per-task scaffolds drifted; the only '
                'thing a scheduled playlist task owns is its track ids'
            )
        assert len(source.splitlines()) < 20

    def test_both_reach_the_scaffold_with_the_dotted_path_the_registry_runs(self):
        assert task_types.CRON_INLINE_TASKS
        for cron_type, dotted in task_types.CRON_INLINE_TASKS.items():
            func = taskqueue.resolve_func(dotted)
            assert func.__name__ in self._task_sources()[cron_type]
            assert list(inspect.signature(func).parameters) == [
                'server_scope', 'inline_task_id',
            ], 'the shared inline branch passes a server scope and the row id only'


def _minute_expr(ts):
    local = time.localtime(ts)
    return f"{local.tm_min} {local.tm_hour} * * *"


class _FakeCronTable:
    def __init__(self, rows):
        self.rows = rows
        self.last_run = {row['id']: None for row in rows}
        self.claims = []
        self.disabled = set()

    def claim(self, _db, row_id, minute_start, cron_expr):
        self.claims.append((row_id, minute_start))
        last = self.last_run.get(row_id)
        if row_id in self.disabled or (last is not None and last >= minute_start):
            return False
        self.last_run[row_id] = minute_start
        return True


def _run_ticks(table, clock, dispatch, sleeps=None, on_connect=None):
    import app_cron

    def sleep(seconds):
        if sleeps is not None:
            sleeps.append(seconds)
        clock['now'] += seconds

    cur = MagicMock()
    cur.fetchall.return_value = table.rows
    db = MagicMock()
    db.cursor.return_value = cur

    def connect():
        if on_connect is not None:
            on_connect()
        return db

    with (
        patch('app_cron.get_db', side_effect=connect),
        patch.object(app_cron.time, 'time', lambda: clock['now']),
        patch.object(app_cron.time, 'monotonic', lambda: clock['now'] - clock.get('jump', 0)),
        patch.object(app_cron.time, 'sleep', sleep),
        patch('app_cron._claim_cron_minute', side_effect=table.claim),
        patch('app_cron._dispatch_cron_row', side_effect=dispatch),
        patch('app_cron.clear_cron_retry'),
        patch('app_cron._record_cron_retry'),
    ):
        yield app_cron.run_due_cron_jobs


class TestTheTickCatchesUpEveryMinuteItMissed:
    BASE = 1_900_000_020 - (1_900_000_020 % 60)

    def test_a_slow_inline_run_spanning_a_minute_does_not_swallow_a_batch_schedule(self):
        clock = {'now': self.BASE + 5}
        table = _FakeCronTable([
            {'id': 1, 'task_type': 'sonic_fingerprint', 'cron_expr': '* * * * *'},
            {'id': 2, 'task_type': 'analysis', 'cron_expr': _minute_expr(self.BASE + 60)},
        ])
        fired = []

        def dispatch(_db, row):
            fired.append((row['task_type'], clock['now']))
            if row['task_type'] == 'sonic_fingerprint' and len(fired) == 1:
                clock['now'] = self.BASE + 75
                return 'ran'
            return 'enqueued' if row['task_type'] == 'analysis' else 'ran'

        for tick in _run_ticks(table, clock, dispatch):
            tick()
            assert [name for name, _ in fired] == ['sonic_fingerprint']
            clock['now'] = self.BASE + 125
            tick()

        assert [name for name, _ in fired] == [
            'sonic_fingerprint', 'analysis', 'sonic_fingerprint',
        ], (
            'the inline run held the poll thread through the minute the analysis '
            'was due; evaluating only "now" on the next tick lost that schedule'
        )
        assert (2, self.BASE + 60) in table.claims, (
            'the late fire is claimed for ITS minute, so a second web process '
            'evaluating the same minute cannot fire it again'
        )

    def test_an_every_minute_row_fires_once_per_tick_not_once_per_missed_minute(self):
        clock = {'now': self.BASE + 5}
        table = _FakeCronTable([
            {'id': 1, 'task_type': 'sonic_fingerprint', 'cron_expr': '* * * * *'},
        ])
        fired = []

        def dispatch(_db, row):
            fired.append(clock['now'])
            return 'ran'

        for tick in _run_ticks(table, clock, dispatch):
            tick()
            clock['now'] = self.BASE + 600 + 5
            tick()

        assert len(fired) == 2
        assert table.claims == [(1, self.BASE), (1, self.BASE + 600)], (
            'one fire for the most recent matching minute of the window'
        )

    def test_the_first_tick_after_a_start_evaluates_only_the_current_minute(self):
        clock = {'now': self.BASE + 5}
        table = _FakeCronTable([
            {'id': 2, 'task_type': 'analysis', 'cron_expr': _minute_expr(self.BASE - 60)},
        ])
        dispatch = MagicMock(return_value='enqueued')

        for tick in _run_ticks(table, clock, dispatch):
            tick()

        dispatch.assert_not_called()

    def test_a_busy_thread_is_caught_up_for_the_whole_retry_window(self):
        import app_cron

        window = app_cron.CRON_RETRY_MAX_MINUTES
        clock = {'now': self.BASE + 5}
        with patch.object(app_cron.time, 'monotonic', lambda: clock['now']):
            app_cron._minutes_to_evaluate(clock['now'])
            clock['now'] = self.BASE + window * 60 + 5
            minutes, dropped = app_cron._minutes_to_evaluate(clock['now'])
            clock['now'] += (window + 10) * 60
            later, too_old = app_cron._minutes_to_evaluate(clock['now'])

        assert len(minutes) == window and dropped == [], (
            'a gap the monotonic clock also measured is a busy thread, caught up in full'
        )
        assert minutes[0] == self.BASE + 60 and minutes[-1] == self.BASE + window * 60
        assert len(later) == window and len(too_old) == 10, (
            'nothing is caught up past CRON_RETRY_MAX_MINUTES, the bound of every wait'
        )

    def test_an_hour_long_inline_run_loses_no_batch_schedule(self):
        clock = {'now': self.BASE + 5}
        table = _FakeCronTable([
            {'id': 1, 'task_type': 'sonic_fingerprint', 'cron_expr': _minute_expr(self.BASE)},
            {'id': 2, 'task_type': 'clustering', 'cron_expr': _minute_expr(self.BASE + 60 * 60)},
        ])
        fired = []

        def dispatch(_db, row):
            fired.append(row['task_type'])
            if row['task_type'] == 'sonic_fingerprint':
                clock['now'] = self.BASE + 65 * 60 + 5
                return 'ran'
            return 'enqueued'

        for tick in _run_ticks(table, clock, dispatch):
            tick()
            tick()

        assert fired == ['sonic_fingerprint', 'clustering'], (
            'a 65-minute online run is a busy thread, not a clock jump: the '
            'clustering due during it must still fire'
        )

    def test_a_batch_schedule_older_than_the_window_is_a_visible_skip(self):
        import app_cron

        window = app_cron.CRON_RETRY_MAX_MINUTES
        clock = {'now': self.BASE + 5}
        table = _FakeCronTable([
            {'id': 1, 'task_type': 'album_of_the_week', 'cron_expr': _minute_expr(self.BASE)},
            {'id': 2, 'task_type': 'analysis', 'cron_expr': _minute_expr(self.BASE + 60)},
            {'id': 3, 'task_type': 'sonic_fingerprint', 'cron_expr': _minute_expr(self.BASE + 120)},
        ])
        fired = []

        def dispatch(_db, row):
            fired.append(row['task_type'])
            if row['task_type'] == 'album_of_the_week':
                clock['now'] = self.BASE + (window + 10) * 60 + 5
            return 'ran' if row['task_type'] in task_types.INLINE_FLASK_TASK_TYPES else 'enqueued'

        with patch('app_cron._record_retry_expired') as expired:
            for tick in _run_ticks(table, clock, dispatch):
                tick()
                tick()

        assert fired == ['album_of_the_week']
        assert [call.args[0] for call in expired.call_args_list] == ['analysis'], (
            'the analysis was due before the window: a visible skip, never a silent '
            'loss; the online row missed the same way is simply skipped'
        )
        texts = expired.call_args.kwargs
        assert 'scheduler was busy' in texts['message'], (
            'nothing blocked this run: the skip must not claim a blocker kept it waiting'
        )
        assert 'busy' in texts['status_message']

    def test_a_gap_past_the_catch_up_is_a_clock_jump_not_a_replay(self, caplog):
        clock = {'now': self.BASE + 5}
        jump = 61 * 60
        table = _FakeCronTable([
            {'id': 2, 'task_type': 'analysis', 'cron_expr': _minute_expr(self.BASE + 120)},
            {'id': 3, 'task_type': 'clustering', 'cron_expr': _minute_expr(self.BASE + jump)},
        ])
        fired = []

        def dispatch(_db, row):
            fired.append(row['task_type'])
            return 'enqueued'

        with caplog.at_level(logging.WARNING, logger='app_cron'):
            for tick in _run_ticks(table, clock, dispatch):
                tick()
                clock['now'] = self.BASE + jump + 5
                clock['jump'] = jump
                tick()

        assert fired == ['clustering'], (
            'a wall-clock jump (NTP step, a host waking from sleep) that the '
            'monotonic clock did not see must not fire every schedule of the '
            'skipped span at once'
        )
        assert any('clock jump' in record.getMessage() for record in caplog.records)


class TestTheTickSurvivesWhatALongOnlineRunChanges:
    BASE = 1_900_000_020 - (1_900_000_020 % 60)

    def test_a_row_disabled_during_an_earlier_online_run_does_not_fire(self):
        clock = {'now': self.BASE + 5}
        table = _FakeCronTable([
            {'id': 1, 'task_type': 'album_of_the_week', 'cron_expr': '* * * * *'},
            {'id': 2, 'task_type': 'sonic_fingerprint', 'cron_expr': '* * * * *'},
        ])
        fired = []

        def dispatch(_db, row):
            fired.append(row['task_type'])
            table.disabled.add(2)
            return 'ran'

        for tick in _run_ticks(table, clock, dispatch):
            tick()

        assert fired == ['album_of_the_week'], (
            'the fingerprint was disabled on the page while the album ran: the '
            'stale SELECT of the tick start must not run it anyway'
        )

    def test_a_slow_database_connect_is_not_taken_for_a_clock_jump(self):
        clock = {'now': self.BASE + 5}
        table = _FakeCronTable([
            {'id': 1, 'task_type': 'sonic_fingerprint', 'cron_expr': _minute_expr(self.BASE)},
            {'id': 2, 'task_type': 'analysis', 'cron_expr': _minute_expr(self.BASE + 300)},
        ])
        slow = {'first': True}
        fired = []

        def on_connect():
            if slow['first']:
                slow['first'] = False
                clock['now'] += 180

        def dispatch(_db, row):
            fired.append(row['task_type'])
            if row['task_type'] == 'sonic_fingerprint':
                clock['now'] = self.BASE + 600 + 5
                return 'ran'
            return 'enqueued'

        for tick in _run_ticks(table, clock, dispatch, on_connect=on_connect):
            tick()
            tick()

        assert fired == ['sonic_fingerprint', 'analysis'], (
            'both clocks are read at the tick start: a 3-minute connect must not '
            'make the next tick see a clock jump and skip the analysis'
        )

    def test_a_schedule_saved_during_a_busy_run_starts_from_its_save(self):
        import app_cron

        clock = {'now': self.BASE + 5}
        table = _FakeCronTable([
            {'id': 1, 'task_type': 'sonic_fingerprint', 'cron_expr': _minute_expr(self.BASE)},
            {'id': 3, 'task_type': 'clustering', 'cron_expr': '0 4 1 1 *'},
        ])
        fired = []

        def dispatch(_db, row):
            fired.append(row['task_type'])
            if row['task_type'] == 'sonic_fingerprint':
                table.rows.append(
                    {'id': 2, 'task_type': 'analysis', 'cron_expr': _minute_expr(self.BASE + 120)}
                )
                table.rows[1]['cron_expr'] = _minute_expr(self.BASE + 300)
                app_cron._cron_saved_at['analysis'] = self.BASE + 200
                app_cron._cron_saved_at['clustering'] = self.BASE + 200
                clock['now'] = self.BASE + 600 + 5
                return 'ran'
            return 'enqueued'

        for tick in _run_ticks(table, clock, dispatch):
            tick()
            tick()

        assert fired == ['sonic_fingerprint', 'clustering'], (
            'saved at +200 s during the busy run: the analysis minute (+120 s) had '
            'already passed, so it must not fire retroactively, while the edited '
            'clustering minute (+300 s) came after the save and must still fire'
        )

    def test_a_schedule_turned_off_and_on_never_fires_for_the_minutes_it_was_off(self):
        import app_cron

        clock = {'now': self.BASE + 5}
        table = _FakeCronTable([
            {'id': 1, 'task_type': 'sonic_fingerprint', 'cron_expr': _minute_expr(self.BASE)},
            {'id': 2, 'task_type': 'analysis', 'cron_expr': _minute_expr(self.BASE + 120)},
        ])
        fired = []

        def dispatch(_db, row):
            fired.append(row['task_type'])
            if row['task_type'] == 'sonic_fingerprint':
                app_cron._cron_saved_at['analysis'] = self.BASE + 400
                clock['now'] = self.BASE + 600 + 5
                return 'ran'
            return 'enqueued'

        for tick in _run_ticks(table, clock, dispatch):
            tick()
            tick()

        assert fired == ['sonic_fingerprint'], (
            'switched off, then back on at +400 s with the same expression: its '
            '+120 s minute passed while it was off and must not fire on the catch-up'
        )

    def test_a_zero_retry_window_catches_up_nothing(self):
        import app_cron

        with patch('app_cron.CRON_RETRY_MAX_MINUTES', 0):
            app_cron._minutes_to_evaluate(self.BASE + 5, 1000.0)
            minutes, dropped = app_cron._minutes_to_evaluate(self.BASE + 600 + 5, 1600.0)

        assert minutes == [self.BASE + 600], 'the current minute is always evaluated'
        assert len(dropped) == 9, 'a zero window must never mean "the whole gap"'

    def test_windows_evaluates_only_the_current_minute_as_before(self):
        import app_cron

        with patch('app_cron._CATCH_UP_SUPPORTED', False):
            app_cron._minutes_to_evaluate(self.BASE + 5, 1000.0)
            minutes, dropped = app_cron._minutes_to_evaluate(self.BASE + 600 + 5, 1600.0)

        assert (minutes, dropped) == ([self.BASE + 600], []), (
            'the Windows monotonic clock counts through sleep, so a laptop wake '
            'would look like a busy thread and replay its schedules: there the '
            'scheduler keeps its old current-minute-only behaviour'
        )



class TestTheInlineRunSurvivesADatabaseDrop:
    def test_a_dead_connection_still_writes_the_failure_row(self):
        import app_cron

        db = _open_db()
        db.rollback.side_effect = psycopg2.InterfaceError('connection already closed')

        def run():
            raise RuntimeError('the database restarted mid-run')

        with patch('app_cron.save_task_status', return_value=True) as save:
            outcome = app_cron._run_inline(db, 'job-1', 'alchemy_radio', run, ERR_SEARCH_FAILED)

        assert outcome == 'ran'
        assert save.call_args_list[-1].args[2] == config.TASK_STATUS_FAILURE, (
            'a rollback on a dead connection must not skip the FAIL row'
        )

    def test_a_row_after_a_database_drop_uses_the_reconnected_connection(self):
        import app_cron

        dead = MagicMock()
        dead.closed = 2
        fresh = MagicMock()
        fresh.closed = 0
        claimed_on = []

        def claim(db, *_args):
            claimed_on.append(db)
            return True

        with (
            patch('app_cron.get_db', return_value=fresh),
            patch('app_cron._claim_cron_minute', side_effect=claim),
            patch('app_cron._dispatch_cron_row', return_value='enqueued'),
            patch('app_cron.clear_cron_retry') as clear,
        ):
            app_cron._fire_cron_row(
                dead, {'id': 1, 'task_type': 'analysis', 'cron_expr': '* * * * *'}, 0,
            )

        assert claimed_on == [fresh]
        assert clear.call_args.kwargs['conn'] is fresh

    def test_an_aborted_transaction_is_cleared_before_the_success_row(self):
        import app_cron

        order = MagicMock()
        db = order.db
        db.closed = 0
        db.get_transaction_status.return_value = TRANSACTION_STATUS_INERROR
        order.save.return_value = True

        with patch('app_cron.save_task_status', order.save):
            outcome = app_cron._run_inline(
                db, 'job-1', 'alchemy_radio', lambda: {'message': 'done'}, ERR_SEARCH_FAILED,
            )

        assert outcome == 'ran'
        names = [name for name, _args, _kwargs in order.mock_calls if name in ('db.rollback', 'save')]
        assert names[-2:] == ['db.rollback', 'save'], (
            'a failure the run caught internally left the transaction aborted; it '
            'must be cleared or the SUCCESS row is refused'
        )
        assert order.save.call_args.args[2] == config.TASK_STATUS_SUCCESS


class TestTheInlineTerminalRowIsTheWorkersOwnBuilder:
    def test_the_worker_keeps_collapsing_its_success_log(self):
        log = taskqueue.terminal_log(['[t] step'], config.TASK_STATUS_SUCCESS, 'done')

        assert log == ['Task completed successfully. Final status: done'], (
            'the worker output must stay byte-identical after the move'
        )

    def test_an_inline_success_keeps_the_steps_and_never_repeats_the_last_line(self):
        kept = taskqueue.terminal_log(
            ['[t] step'], config.TASK_STATUS_SUCCESS, 'done', keep_log=True,
        )
        again = taskqueue.terminal_log(kept, config.TASK_STATUS_SUCCESS, 'done', keep_log=True)

        assert kept[0] == '[t] step' and kept[-1].endswith('] done')
        assert again == kept

    def test_the_worker_uses_the_shared_builder(self):
        from taskqueue import worker

        assert worker._terminal_details is taskqueue.terminal_details
        assert worker._UNREAD is taskqueue.ERROR_AT_CLAIM_UNREAD


class TestOnlineRowsDueTogetherStartTenSecondsApart:
    BASE = 1_900_000_020 - (1_900_000_020 % 60)

    def _every_minute(self, *rows):
        return _FakeCronTable([
            {'id': row_id, 'task_type': task_type, 'cron_expr': '* * * * *'}
            for row_id, task_type in rows
        ])

    def test_three_online_rows_due_together_start_at_0_10_and_20_seconds(self):
        clock = {'now': self.BASE + 5}
        table = self._every_minute(
            (3, 'sonic_fingerprint'), (1, 'alchemy_radio'), (2, 'album_of_the_week'),
        )
        fired, sleeps = [], []

        def dispatch(_db, row):
            fired.append((row['task_type'], clock['now'] - (self.BASE + 5)))
            return 'ran'

        for tick in _run_ticks(table, clock, dispatch, sleeps):
            tick()

        assert fired == [
            ('album_of_the_week', 0), ('alchemy_radio', 10), ('sonic_fingerprint', 20),
        ], 'online rows sharing a tick start 10 seconds apart in a fixed order'
        assert sleeps == [10, 10]

    def test_a_batch_row_of_the_same_tick_is_dispatched_before_any_online_run(self):
        clock = {'now': self.BASE + 5}
        table = self._every_minute(
            (1, 'sonic_fingerprint'), (2, 'analysis'), (3, 'alchemy_radio'), (4, 'clustering'),
        )
        fired = []

        def dispatch(_db, row):
            fired.append((row['task_type'], clock['now'] - (self.BASE + 5)))
            return 'ran' if row['task_type'] in task_types.INLINE_FLASK_TASK_TYPES else 'enqueued'

        for tick in _run_ticks(table, clock, dispatch):
            tick()

        assert fired == [
            ('analysis', 0), ('clustering', 0), ('alchemy_radio', 0), ('sonic_fingerprint', 10),
        ], 'a batch row only enqueues, so it never waits behind an online run'

    def test_a_slow_online_run_lets_the_next_start_at_once(self):
        clock = {'now': self.BASE + 5}
        table = self._every_minute((1, 'alchemy_radio'), (2, 'sonic_fingerprint'))
        fired, sleeps = [], []

        def dispatch(_db, row):
            fired.append((row['task_type'], clock['now'] - (self.BASE + 5)))
            clock['now'] += 15 if row['task_type'] == 'alchemy_radio' else 1
            return 'ran'

        for tick in _run_ticks(table, clock, dispatch, sleeps):
            tick()

        assert fired == [('alchemy_radio', 0), ('sonic_fingerprint', 15)]
        assert sleeps == [], 'the gap already passed during the slow run; no extra wait'

    def test_the_gap_counts_from_the_previous_start_so_it_is_never_shorter(self):
        clock = {'now': self.BASE + 5}
        table = self._every_minute(
            (1, 'album_of_the_week'), (2, 'alchemy_radio'), (3, 'sonic_fingerprint'),
        )
        fired, sleeps = [], []

        def dispatch(_db, row):
            fired.append((row['task_type'], clock['now'] - (self.BASE + 5)))
            clock['now'] += 12 if row['task_type'] == 'album_of_the_week' else 3
            return 'ran'

        for tick in _run_ticks(table, clock, dispatch, sleeps):
            tick()

        assert fired == [
            ('album_of_the_week', 0), ('alchemy_radio', 12), ('sonic_fingerprint', 22),
        ], (
            'slots fixed to the tick start would start the third run at +20, only '
            '8 seconds after the second one'
        )
        assert sleeps == [7]

    def test_a_row_another_web_process_claimed_uses_no_slot(self):
        clock = {'now': self.BASE + 5}
        table = self._every_minute((1, 'album_of_the_week'), (2, 'alchemy_radio'))
        table.last_run[1] = self.BASE
        fired, sleeps = [], []

        def dispatch(_db, row):
            fired.append((row['task_type'], clock['now'] - (self.BASE + 5)))
            return 'ran'

        for tick in _run_ticks(table, clock, dispatch, sleeps):
            tick()

        assert fired == [('alchemy_radio', 0)]
        assert sleeps == [], 'nothing started before it, so nothing to wait for'

    def test_the_gap_is_measured_on_the_monotonic_clock(self):
        import app_cron

        source = inspect.getsource(app_cron.run_due_cron_jobs)
        assert 'time.monotonic()' in source
        stagger = source.split('previous_start = None', 1)[1]
        assert 'time.time()' not in stagger, (
            'a wall-clock step must never stretch the sleep of the cron thread'
        )

    @patch('app_cron.get_db')
    def test_the_retry_path_never_waits(self, mock_get_db):
        import app_cron

        mock_get_db.return_value = MagicMock()
        entries = [
            {'task_type': name, 'retry_until': 10**12, 'attempts': 0,
             'first_blocked_at': None, 'blocker_task_id': None, 'blocker_task_type': None}
            for name in ('analysis', 'clustering')
        ]
        dispatch = MagicMock(return_value='enqueued')
        with (
            patch('app_cron.list_pending_cron_retries', return_value=entries),
            patch('app_cron._cron_row_for_retry', side_effect=lambda _db, name: {'id': 1, 'task_type': name}),
            patch('app_cron.cron_retry_task_already_done', return_value=False),
            patch('app_cron.clear_cron_retry'),
            patch('app_cron._touch_cron_last_run'),
            patch('app_cron._dispatch_cron_row', dispatch),
            patch.object(app_cron.time, 'sleep') as sleep,
        ):
            assert app_cron.retry_due_cron_jobs() == 2

        assert dispatch.call_count == 2
        sleep.assert_not_called()


class TestTheOneInlineScaffold:
    def _details_row(self, details):
        return {'details': json.dumps(details)}

    def test_success_merges_into_the_reported_details(self):
        import app_cron

        reported = {
            'message': 'Building the sonic fingerprint for one (1/1)...',
            'status_message': 'Building the sonic fingerprint for one (1/1)...',
            'log': ['[t] Building the sonic fingerprint playlist...'],
        }
        summary = {'message': 'Created 1 sonic fingerprint playlist(s).', 'playlists_created': 1}
        with (
            patch('app_cron.save_task_status', return_value=True) as save,
            patch('app_cron.get_task_info_from_db', return_value=self._details_row(reported)),
        ):
            outcome = app_cron._run_inline(
                _open_db(), 'job-1', 'sonic_fingerprint', lambda: summary, 6005,
            )

        assert outcome == 'ran'
        status, details = save.call_args[0][2], save.call_args[1]['details']
        assert status == config.TASK_STATUS_SUCCESS
        assert details['log'][0] == '[t] Building the sonic fingerprint playlist...', (
            'the terminal write used to REPLACE the details and threw the log away'
        )
        assert details['log'][-1].endswith(summary['message'])
        assert details['status_message'] == summary['message']
        assert details['playlists_created'] == 1

    def test_a_playlist_failure_carries_its_dotted_paths_classified_error(self):
        import app_cron

        reported = {'status_message': 'Building...', 'log': ['[t] Building...']}
        dotted = task_types.CRON_INLINE_TASKS['album_of_the_week']
        with (
            patch('app_cron.save_task_status', return_value=True) as save,
            patch('app_cron.get_task_info_from_db', return_value=self._details_row(reported)),
            patch(
                'tasks.album_creation_manager.run_album_of_the_week_task',
                side_effect=RuntimeError('every server failed'),
            ),
        ):
            outcome = app_cron._run_playlist_inline(_open_db(), 'job-2', 'album_of_the_week', 'all')

        assert outcome == 'ran', 'a run that executed and failed is never retried'
        assert save.call_args[0][2] == config.TASK_STATUS_FAILURE
        details = save.call_args[1]['details']
        assert details['error']['error_code'] == taskqueue.TASK_FUNC_ERROR_CODES[dotted]
        assert {'error_code', 'error_class', 'error_message'} <= set(details['error'])
        assert details['log'][0] == '[t] Building...'

    def test_a_cancelled_run_writes_no_verdict_over_the_revoked_row(self):
        import app_cron

        def cancelled():
            raise TaskCancelled('revoked')

        with patch('app_cron.save_task_status', return_value=True) as save:
            outcome = app_cron._run_inline(_open_db(), 'job-3', 'sonic_fingerprint', cancelled, 6005)

        assert outcome == 'ran'
        assert len(save.call_args_list) == 1, 'only the STARTED row; REVOKED is the verdict'

    def test_a_row_that_cannot_be_started_runs_nothing(self):
        import app_cron

        run = MagicMock()
        with patch('app_cron.save_task_status', side_effect=RuntimeError('db down')):
            outcome = app_cron._run_inline(_open_db(), 'job-4', 'sonic_fingerprint', run, 6005)

        assert outcome == 'failed'
        run.assert_not_called()

    def test_a_failed_terminal_write_is_logged_not_ignored(self, caplog):
        import app_cron

        writes = iter([True, RuntimeError('db down')])

        def save(*_args, **_kwargs):
            outcome = next(writes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with (
            patch('app_cron.save_task_status', side_effect=save),
            caplog.at_level(logging.ERROR, logger='app_cron'),
        ):
            app_cron._run_inline(_open_db(), 'job-5', 'sonic_fingerprint', lambda: {}, 6005)

        assert any('could not write' in record.getMessage() for record in caplog.records)

    def test_a_terminal_write_the_row_refused_is_reported(self, caplog):
        import app_cron

        with (
            patch('app_cron.save_task_status', side_effect=[True, False]),
            caplog.at_level(logging.WARNING, logger='app_cron'),
        ):
            app_cron._run_inline(_open_db(), 'job-6', 'sonic_fingerprint', lambda: {}, 6005)

        assert any('already terminal' in record.getMessage() for record in caplog.records)

    @patch('app_cron.cron_matches_now', return_value=True)
    @patch('app_cron.get_db')
    def test_an_executed_inline_run_clears_a_retry_and_records_none(self, mock_get_db, _matches):
        from app_cron import run_due_cron_jobs

        db, _cur = _setup_db_mock(task_type='album_of_the_week')
        mock_get_db.return_value = db
        with (
            patch('app_cron._run_inline', return_value='ran'),
            patch('app_cron.clear_cron_retry') as clear,
            patch('app_cron._record_cron_retry') as record,
        ):
            run_due_cron_jobs()

        clear.assert_called_once_with('album_of_the_week', conn=db)
        record.assert_not_called()


class TestTheInlinePlaylistRowIsWatchedAndBeatsInsideTheStaleSweep:
    def test_the_inline_cadence_is_well_inside_the_stale_sweep_and_never_disabled(
        self, monkeypatch
    ):
        from tasks import recovery
        from tasks.task_run import _playlist_heartbeat_cadence

        monkeypatch.setattr(config, 'QUEUE_INLINE_STALE_SECONDS', 1800.0)
        monkeypatch.setattr(config, 'QUEUE_WEDGED_MAIN_TASK_MINUTES', 0)

        every_minutes, stop_after = _playlist_heartbeat_cadence(inline=True)
        interval = recovery._heartbeat_interval_seconds(every_minutes)

        assert 0 < interval <= 1800.0 / 2, (
            'the old cadence came from QUEUE_WEDGED_MAIN_TASK_MINUTES: 2700 s by '
            'default, past the 1800 s after which fail_stale_inline_rows fails the '
            'row, and 0 there disabled the beat altogether'
        )
        assert stop_after is not None and stop_after * 60 > 1800.0

    def test_a_queued_run_keeps_the_wedged_main_cadence(self, monkeypatch):
        from tasks.task_run import _playlist_heartbeat_cadence

        monkeypatch.setattr(config, 'QUEUE_WEDGED_MAIN_TASK_MINUTES', 180)

        every_minutes, stop_after = _playlist_heartbeat_cadence(inline=False)

        assert every_minutes is None
        assert stop_after == 360

    def test_the_heartbeat_and_the_cancel_guard_watch_the_inline_row(self, monkeypatch):
        from tasks import task_run

        monkeypatch.setattr(config, 'QUEUE_INLINE_STALE_SECONDS', 1800.0)
        beats, guards = [], []

        @contextmanager
        def fake_heartbeat(task_id, label=None, every_minutes=None, stop_after_minutes=None):
            beats.append((task_id, every_minutes, stop_after_minutes))
            yield

        @contextmanager
        def fake_guard(task_id, parent_task_id=None, every_seconds=None):
            guards.append(task_id)
            yield lambda force=False: None

        with (
            patch('tasks.recovery.row_heartbeat', fake_heartbeat),
            patch('tasks.task_run.cancel_guard', fake_guard),
            patch('tasks.task_run.save_task_status', return_value=True),
            patch('tasks.task_run.taskqueue.current_task_id', return_value=None),
            patch('tasks.mediaserver.registry.servers_for_scope', return_value=[None]),
            patch('tasks.mediaserver.create_or_replace_playlist', return_value={'Id': 'p'}),
        ):
            summary = task_run.run_playlist_task_per_server(
                'sonic_fingerprint', 'sonic fingerprint', 'Name', 'Sonic Fingerprint',
                lambda: ['a'], 'all', 'inline-row-1',
            )

        assert summary['playlists_created'] == 1
        assert guards == ['inline-row-1']
        assert beats and beats[0][0] == 'inline-row-1'
        assert beats[0][1] == 30.0


def _cron_api_client():
    from flask import Flask
    from app_cron import cron_bp

    app = Flask(__name__)
    app.register_blueprint(cron_bp)
    app.config['TESTING'] = True
    return app.test_client()


def test_get_cron_entries_exposes_the_pending_retry_state():
    client = _cron_api_client()

    cur = MagicMock()
    cur.fetchall.return_value = [{
        'id': 1,
        'name': 'Clustering',
        'task_type': 'clustering',
        'cron_expr': '0 2 * * *',
        'enabled': True,
        'last_run': 123.0,
        'created_at': None,
        'options': {},
    }]
    db = MagicMock()
    db.cursor.return_value = cur
    retry_row = {
        'task_type': 'clustering',
        'retry_until': time.time() + 3600,
        'attempts': 3,
        'created_at': None,
        'blocker_task_id': 'live-1',
        'blocker_task_type': 'main_analysis',
    }

    with (
        patch('app_cron.get_db', return_value=db),
        patch('app_cron.list_pending_cron_retries', return_value=[retry_row]),
    ):
        response = client.get('/api/cron')

    assert response.status_code == 200
    entry = response.get_json()[0]
    assert entry['retry_pending'] is True
    assert entry['retry_attempts'] == 3
    assert entry['retry_blocker_task_type'] == 'main_analysis'
    assert entry['retry_until'] == retry_row['retry_until']


def test_get_cron_entries_marks_entries_without_a_retry_as_not_pending():
    client = _cron_api_client()

    cur = MagicMock()
    cur.fetchall.return_value = [{
        'id': 1,
        'name': 'Analysis',
        'task_type': 'analysis',
        'cron_expr': '0 2 * * *',
        'enabled': True,
        'last_run': 123.0,
        'created_at': None,
        'options': {},
    }]
    db = MagicMock()
    db.cursor.return_value = cur

    with (
        patch('app_cron.get_db', return_value=db),
        patch('app_cron.list_pending_cron_retries', return_value=[]),
    ):
        response = client.get('/api/cron')

    assert response.status_code == 200
    assert response.get_json()[0]['retry_pending'] is False


@pytest.mark.parametrize('before, enabled, recorded', [
    (('0 2 * * 6', False), True, True),
    (('0 3 * * 6', True), True, True),
    (('0 2 * * 6', True), True, False),
    (('0 2 * * 6', True), False, False),
], ids=['enabled-now', 'expression-changed', 'unchanged-resave', 'disabled'])
def test_a_save_records_when_a_schedule_starts_only_if_it_changed(before, enabled, recorded):
    import app_cron

    client = _cron_api_client()
    cur = MagicMock()
    cur.fetchone.return_value = before
    db = MagicMock()
    db.cursor.return_value = cur
    with (
        patch('app_cron.get_db', return_value=db),
        patch('app_cron.time.time', return_value=5000.0),
    ):
        response = client.post('/api/cron', json={
            'id': 7, 'name': 'Clustering', 'task_type': 'clustering',
            'cron_expr': '0 2 * * 6', 'enabled': enabled,
        })

    assert response.status_code == 200
    assert (app_cron._cron_saved_at.get('clustering') == 5000.0) is recorded, (
        'the page re-saves every row on each Save: only a schedule that was '
        'enabled or changed may restart its catch-up window'
    )


def _cron_page_script():
    page = (pathlib.Path(__file__).resolve().parents[2] / 'templates' / 'cron.html').read_text(
        encoding='utf-8'
    )
    return page, page.split('<script>', 1)[1].split('</script>', 1)[0]


def _js_function_body(script, name):
    return script.split('function %s(' % name, 1)[1].split('\n        }\n', 1)[0]


def test_the_overlap_warning_covers_every_row_the_save_sends():
    page, script = _cron_page_script()
    assert 'id="cron-overlap-warning"' in page
    assert 'class="status-warning cron-overlap-warning"' in page
    assert 'at your own risk' in script
    handler = script.split("document.getElementById('save-btn').addEventListener('click'", 1)[1]
    saved_ids = set(re.findall(r"document\.getElementById\('([a-z-]+)-cron'\)\.value", handler))
    checked_ids = set(re.findall(r"\['[^']+', '([a-z-]+)'\]", script.split('const CRON_BUILT_IN_ROWS', 1)[1].split('];', 1)[0]))
    assert saved_ids == checked_ids == {
        'analysis', 'clustering', 'sonic-fingerprint', 'album-of-the-week', 'alchemy-radio',
    }
    rows = _js_function_body(script, 'cronScheduleRows')
    assert 'window.pluginCronTasks' in rows and 'fields.enable.checked' in rows


def test_the_overlap_warning_is_live_and_never_blocks_the_save():
    _, script = _cron_page_script()
    assert "addEventListener('input', updateOverlapWarning)" in script
    assert "addEventListener('change', updateOverlapWarning)" in script
    assert re.search(r'await loadPluginTasks\(rows\);\s*updateOverlapWarning\(\);', script)
    warning = _js_function_body(script, 'updateOverlapWarning')
    assert 'save-btn' not in warning and 'disabled' not in warning and 'return false' not in warning
    handler = script.split("document.getElementById('save-btn').addEventListener('click'", 1)[1]
    assert 'updateOverlapWarning();' in handler
    returns = [line.strip() for line in handler.splitlines() if re.search(r'\breturn\b', line)]
    assert returns == ['if (saveBtn.disabled) return; // prevent double-click', 'return;'], returns
    guarded = [line for line in handler.splitlines() if re.search(r'\bif\s*\(', line)]
    assert not [line for line in guarded if 'verlap' in line], guarded


def test_the_overlap_warning_states_the_real_scheduling_behaviour():
    import app_cron

    _, script = _cron_page_script()
    risk = script.split('const CRON_OVERLAP_RISK_TEXT = ', 1)[1].split(';\n', 1)[0]
    assert 'at your own risk' in risk
    assert 'run one after another, each starting at least {{ inline_stagger_seconds }} seconds after the previous one' in risk, (
        'online rows due together run back to back, never less than the stagger '
        'apart, and the page must not promise they overlap'
    )
    assert 'skipped if it is still blocked after {{ cron_retry_max_minutes }} minutes' in risk, (
        'a blocked batch schedule expires after CRON_RETRY_MAX_MINUTES; the page '
        'must never promise it waits forever'
    )
    with patch('app_cron.render_template', return_value='') as render:
        app_cron.cron_page()
    assert render.call_args.kwargs['inline_stagger_seconds'] == app_cron._INLINE_STAGGER_SECONDS
    assert render.call_args.kwargs['cron_retry_max_minutes'] == config.CRON_RETRY_MAX_MINUTES
